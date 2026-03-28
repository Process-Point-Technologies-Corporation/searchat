//! Filesystem watcher for AI coding agent conversation directories.
//!
//! Monitors Claude Code, Codex, and Vibe directories for new and modified
//! conversation files, then sends batches of changed paths to the indexing
//! controller via an unbounded channel.
//!
//! Supported agents:
//! - Claude Code: `~/.claude/projects/**/*.jsonl`
//! - Codex: `~/.codex/sessions/**/*.jsonl`
//! - Mistral Vibe: `~/.vibe/logs/session/*.json` (flat, non-recursive)

pub mod error;

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use log::{debug, info, warn};
use notify_debouncer_mini::{new_debouncer, DebounceEventResult};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

pub use error::WatcherError;

// ---------------------------------------------------------------------------
// Filter constants (mirrors Python ConversationEventHandler)
// ---------------------------------------------------------------------------

/// Claude Code subdirectories that are never conversation storage.
const EXCLUDED_DIRS: &[&str] = &[
    "plugins",
    "hooks",
    "commands",
    "todos",
    "settings",
    "usage-sessions",
    "telemetry",
    "sessions",
    "_corrupt_jsonls",
    "plans",
    "worktrees",
];

/// Root-level config/settings filenames that must never be indexed.
const EXCLUDED_FILES: &[&str] = &[
    ".credentials.json",
    "stats-cache.json",
    "settings.json",
    "settings.local.json",
    "settings.backup.json",
    "history.jsonl",
    "sessions-index.json",
    "mcp-needs-auth-cache.json",
];

/// Minimum file size to consider (bytes). Files smaller than this are likely
/// empty or corrupt and are skipped.
const MIN_FILE_SIZE: u64 = 50;

/// Default debounce interval: ignore repeated events for the same file within
/// this window.
const DEFAULT_DEBOUNCE_SECS: u64 = 2;

/// Default batch delay: wait this long after the last event before flushing.
const DEFAULT_BATCH_DELAY_SECS: u64 = 5;

// ---------------------------------------------------------------------------
// Persistent cache structure (matches Python watcher_file_cache.json)
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize)]
struct WatcherCache {
    updated_at: String,
    files: Vec<String>,
}

// ---------------------------------------------------------------------------
// ConversationWatcher
// ---------------------------------------------------------------------------

/// Watches agent conversation directories and forwards batches of changed
/// file paths to an [`mpsc::UnboundedReceiver`].
///
/// # Usage
///
/// ```no_run
/// use searchat_watcher::ConversationWatcher;
/// use tokio::sync::mpsc;
///
/// #[tokio::main]
/// async fn main() {
///     let (watcher, mut rx) = ConversationWatcher::new(None, None);
///     let files = watcher.scan_all_files();
///     println!("Found {} conversation files", files.len());
///     watcher.start().unwrap();
///
///     while let Some(batch) = rx.recv().await {
///         println!("Batch of {} changed files", batch.len());
///     }
/// }
/// ```
pub struct ConversationWatcher {
    /// Dirs to watch, keyed by provider id ("claude", "codex", "vibe").
    provider_dirs: Vec<(String, PathBuf)>,

    /// Cached set of known conversation files (populated by `scan_all_files`).
    known_files: Arc<RwLock<HashSet<PathBuf>>>,

    /// Whether `scan_all_files` has been called at least once.
    scan_complete: Arc<RwLock<bool>>,

    /// Per-path last-event timestamps for debouncing.
    debounce_map: Arc<RwLock<HashMap<PathBuf, Instant>>>,

    /// Sender half of the batch channel.
    tx: mpsc::UnboundedSender<Vec<PathBuf>>,

    /// Debounce interval (how long to suppress repeated events for one path).
    debounce_interval: Duration,

    /// Batch delay (how long to wait after the last event before flushing).
    batch_delay: Duration,

    /// Path for persisting the file cache across restarts.
    cache_path: PathBuf,
}

impl ConversationWatcher {
    /// Create a new watcher.
    ///
    /// Returns `(watcher, receiver)`. The receiver yields `Vec<PathBuf>` batches
    /// when files are created or modified.
    ///
    /// - `debounce_interval`: suppress repeated events for the same path within
    ///   this window (default: 2 s).
    /// - `batch_delay`: wait this long after the last event before flushing
    ///   (default: 5 s).
    pub fn new(
        debounce_interval: Option<Duration>,
        batch_delay: Option<Duration>,
    ) -> (Self, mpsc::UnboundedReceiver<Vec<PathBuf>>) {
        let (tx, rx) = mpsc::unbounded_channel();

        // Collect (provider_id, dir) pairs from all registered providers.
        let provider_dirs = searchat_agents::iter_dirs();

        let cache_path = searchat_config::default_storage_dir().join("watcher_file_cache.json");

        let watcher = Self {
            provider_dirs,
            known_files: Arc::new(RwLock::new(HashSet::new())),
            scan_complete: Arc::new(RwLock::new(false)),
            debounce_map: Arc::new(RwLock::new(HashMap::new())),
            tx,
            debounce_interval: debounce_interval
                .unwrap_or(Duration::from_secs(DEFAULT_DEBOUNCE_SECS)),
            batch_delay: batch_delay.unwrap_or(Duration::from_secs(DEFAULT_BATCH_DELAY_SECS)),
            cache_path,
        };

        // Attempt to restore a previously persisted cache.
        watcher.load_cached_file_list();

        (watcher, rx)
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Walk all watched directories and cache every matching conversation file.
    ///
    /// Vibe directories are scanned flat (non-recursive); all others recursive.
    /// Results are persisted to `watcher_file_cache.json` so the next startup
    /// can bootstrap instantly.
    pub fn scan_all_files(&self) -> Vec<PathBuf> {
        let mut files: Vec<PathBuf> = Vec::new();

        for (provider_id, dir) in &self.provider_dirs {
            if !searchat_config::safe_exists(dir) {
                continue;
            }
            let found = if provider_id == "vibe" {
                scan_flat(dir, ".json")
            } else {
                scan_recursive(dir, ".jsonl")
            };
            files.extend(found);
        }

        {
            let mut known = self.known_files.write();
            *known = files.iter().cloned().collect();
        }
        *self.scan_complete.write() = true;

        self.persist_known_files();
        info!("File scan complete: {} files cached", files.len());
        files
    }

    /// Return the cached file set from the last scan, or `None` if
    /// `scan_all_files` has not been called yet (caller should fall back to
    /// rglob).
    pub fn get_known_files(&self) -> Option<Vec<PathBuf>> {
        if !*self.scan_complete.read() {
            return None;
        }
        Some(self.known_files.read().iter().cloned().collect())
    }

    /// Return `true` if `scan_all_files` has been called at least once.
    pub fn is_scan_complete(&self) -> bool {
        *self.scan_complete.read()
    }

    /// Number of files in the cache.
    pub fn known_file_count(&self) -> usize {
        self.known_files.read().len()
    }

    /// Return all directories this watcher monitors.
    pub fn watched_dirs(&self) -> Vec<PathBuf> {
        self.provider_dirs
            .iter()
            .map(|(_, dir)| dir.clone())
            .collect()
    }

    /// Start watching directories for filesystem events.
    ///
    /// Events are filtered, debounced, and batched before being sent via the
    /// channel returned by [`ConversationWatcher::new`].
    ///
    /// This method spawns a background Tokio task and returns immediately.
    /// Call `stop_tx.send(())` on the returned sender to request a graceful
    /// shutdown (the task will drain any pending batch first).
    pub fn start(&self) -> Result<tokio::sync::oneshot::Sender<()>, WatcherError> {
        let dirs: Vec<PathBuf> = self
            .provider_dirs
            .iter()
            .filter(|(_, d)| searchat_config::safe_exists(d))
            .map(|(_, d)| d.clone())
            .collect();

        if dirs.is_empty() {
            warn!("No watchable directories found — watcher not started");
            // Return a sender that does nothing when dropped/sent
            let (stop_tx, _stop_rx) = tokio::sync::oneshot::channel();
            return Ok(stop_tx);
        }

        let known_files = Arc::clone(&self.known_files);
        let scan_complete = Arc::clone(&self.scan_complete);
        let debounce_map = Arc::clone(&self.debounce_map);
        let tx_batch = self.tx.clone();
        let debounce_interval = self.debounce_interval;
        let batch_delay = self.batch_delay;
        let cache_path = self.cache_path.clone();

        let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();

        // The notify debouncer runs on a blocking thread internally. We bridge
        // its output to an mpsc channel, then drive the batch logic in a Tokio
        // task.
        let (event_tx, mut event_rx) = mpsc::unbounded_channel::<PathBuf>();

        // Build the notify debouncer (blocking / non-async).
        // The debouncer callback fires on a background thread managed by notify.
        let debouncer_delay = debounce_interval;
        let mut debouncer = new_debouncer(debouncer_delay, move |res: DebounceEventResult| {
            match res {
                Ok(events) => {
                    for event in events {
                        // DebouncedEvent has `path` (singular) and `kind` (Any/AnyContinuous)
                        let _ = event_tx.send(event.path);
                    }
                }
                Err(err) => {
                    warn!("Watch error: {err}");
                }
            }
        })?;

        for dir in &dirs {
            info!("Watching directory: {}", dir.display());
            debouncer
                .watcher()
                .watch(dir, notify::RecursiveMode::Recursive)?;
        }

        // Spawn the batch-processing task.
        tokio::spawn(async move {
            // Keep the debouncer alive for the lifetime of this task.
            let _debouncer = debouncer;

            let mut pending: HashSet<PathBuf> = HashSet::new();
            let mut last_event: Option<Instant> = None;

            let mut stop_rx = stop_rx;

            loop {
                // Poll for events or the stop signal with a short sleep so we
                // can check the batch delay without blocking indefinitely.
                let timeout = tokio::time::sleep(Duration::from_millis(500));
                tokio::pin!(timeout);

                tokio::select! {
                    // Stop signal
                    _ = &mut stop_rx => {
                        debug!("Watcher stop signal received");
                        break;
                    }
                    // Incoming filesystem event
                    Some(path) = event_rx.recv() => {
                        if should_process(&path, &debounce_map, debounce_interval) {
                            if is_valid_size(&path) {
                                debug!("Queuing changed file: {}", path.display());
                                // Update the known-files cache if scan is complete
                                if *scan_complete.read() {
                                    let mut known = known_files.write();
                                    if known.insert(path.clone()) {
                                        // New file added — persist asynchronously
                                        persist_known_files_sync(&known, &cache_path);
                                    }
                                }
                                pending.insert(path);
                                last_event = Some(Instant::now());
                            }
                        }
                    }
                    // Timeout tick — check if we should flush
                    _ = &mut timeout => {}
                }

                // Flush if we have pending files and the batch delay has elapsed
                if !pending.is_empty() {
                    let elapsed = last_event
                        .map(|t| t.elapsed())
                        .unwrap_or(Duration::MAX);
                    if elapsed >= batch_delay {
                        let batch: Vec<PathBuf> = pending.drain().collect();
                        info!("Flushing watcher batch: {} files", batch.len());
                        if tx_batch.send(batch).is_err() {
                            // Receiver dropped — shut down
                            debug!("Watcher channel receiver dropped, stopping");
                            break;
                        }
                        last_event = None;
                    }
                }
            }

            // Drain any remaining pending files on shutdown
            if !pending.is_empty() {
                let batch: Vec<PathBuf> = pending.into_iter().collect();
                info!("Watcher shutdown flush: {} files", batch.len());
                let _ = tx_batch.send(batch);
            }
        });

        Ok(stop_tx)
    }

    // -----------------------------------------------------------------------
    // Cache persistence
    // -----------------------------------------------------------------------

    fn load_cached_file_list(&self) {
        if !self.cache_path.exists() {
            return;
        }
        let text = match std::fs::read_to_string(&self.cache_path) {
            Ok(t) => t,
            Err(e) => {
                debug!("Failed to read watcher cache {}: {e}", self.cache_path.display());
                return;
            }
        };
        let cache: WatcherCache = match serde_json::from_str(&text) {
            Ok(c) => c,
            Err(e) => {
                debug!("Failed to parse watcher cache: {e}");
                return;
            }
        };

        let raw_count = cache.files.len();
        let mut known = self.known_files.write();
        *known = cache
            .files
            .into_iter()
            .map(PathBuf::from)
            .filter(|p| !is_excluded_path(p))
            .collect();
        *self.scan_complete.write() = true;

        let pruned = raw_count - known.len();
        if pruned > 0 {
            info!(
                "Restored watcher file cache: {} files ({} excluded by current filters)",
                known.len(),
                pruned
            );
        } else {
            info!("Restored watcher file cache: {} files", known.len());
        }
    }

    fn persist_known_files(&self) {
        let known = self.known_files.read();
        persist_known_files_sync(&known, &self.cache_path);
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Recursively walk `directory` collecting files with `extension`.
/// Skips directories listed in `EXCLUDED_DIRS` and excluded filenames.
pub(crate) fn scan_recursive(directory: &PathBuf, extension: &str) -> Vec<PathBuf> {
    let mut results = Vec::new();
    scan_recursive_inner(directory, extension, &mut results);
    results
}

fn scan_recursive_inner(dir: &PathBuf, extension: &str, out: &mut Vec<PathBuf>) {
    let read_dir = match std::fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(_) => return,
    };

    for entry in read_dir.flatten() {
        let path = entry.path();
        let file_type = match entry.file_type() {
            Ok(ft) => ft,
            Err(_) => continue,
        };

        if file_type.is_symlink() {
            continue;
        }

        if file_type.is_file() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.ends_with(extension) && !EXCLUDED_FILES.contains(&name_str.as_ref()) {
                out.push(path);
            }
        } else if file_type.is_dir() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if !EXCLUDED_DIRS.contains(&name_str.as_ref()) {
                scan_recursive_inner(&path, extension, out);
            }
        }
    }
}

/// Non-recursive scan of a single directory.
pub(crate) fn scan_flat(directory: &PathBuf, extension: &str) -> Vec<PathBuf> {
    let mut results = Vec::new();
    let read_dir = match std::fs::read_dir(directory) {
        Ok(rd) => rd,
        Err(_) => return results,
    };

    for entry in read_dir.flatten() {
        let file_type = match entry.file_type() {
            Ok(ft) => ft,
            Err(_) => continue,
        };
        if file_type.is_symlink() || !file_type.is_file() {
            continue;
        }
        let name = entry.file_name();
        if name.to_string_lossy().ends_with(extension) {
            results.push(entry.path());
        }
    }
    results
}

/// Return `true` if `path` passes all filter rules and its debounce window has
/// elapsed.
fn should_process(
    path: &PathBuf,
    debounce_map: &Arc<RwLock<HashMap<PathBuf, Instant>>>,
    debounce_interval: Duration,
) -> bool {
    // Extension check
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    if ext != "jsonl" && ext != "json" {
        return false;
    }

    // Filename exclusions
    if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
        if EXCLUDED_FILES.contains(&name) {
            return false;
        }
        if name.ends_with(".meta.json") {
            return false;
        }
    }

    // Directory exclusions
    if is_excluded_path(path) {
        return false;
    }

    // Debounce
    let now = Instant::now();
    let mut map = debounce_map.write();
    if let Some(last) = map.get(path) {
        if now.duration_since(*last) < debounce_interval {
            return false;
        }
    }
    map.insert(path.clone(), now);

    // Prune stale entries when the map grows large
    if map.len() > 5000 {
        let cutoff = now - debounce_interval * 10;
        map.retain(|_, v| *v > cutoff);
    }

    true
}

/// Return `true` if the file on disk is large enough to be a real conversation.
fn is_valid_size(path: &PathBuf) -> bool {
    match std::fs::metadata(path) {
        Ok(m) => m.len() >= MIN_FILE_SIZE,
        Err(_) => false,
    }
}

/// Return `true` if `path` falls under an excluded directory or has an
/// excluded filename.
fn is_excluded_path(path: &PathBuf) -> bool {
    if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
        if EXCLUDED_FILES.contains(&name) {
            return true;
        }
        if name.ends_with(".meta.json") {
            return true;
        }
    }

    // Normalise separators for cross-platform matching
    let norm = path.to_string_lossy().replace('\\', "/").to_lowercase();
    for excluded in EXCLUDED_DIRS {
        if norm.contains(&format!("/.claude/{excluded}/"))
            || norm.contains(&format!("/.codex/{excluded}/"))
        {
            return true;
        }
    }
    false
}

/// Persist `known` to `cache_path`. Called from both sync and async contexts,
/// so it uses plain blocking I/O.
fn persist_known_files_sync(known: &HashSet<PathBuf>, cache_path: &PathBuf) {
    let parent = match cache_path.parent() {
        Some(p) => p,
        None => return,
    };
    if let Err(e) = std::fs::create_dir_all(parent) {
        debug!("Failed to create watcher cache dir: {e}");
        return;
    }

    let mut files: Vec<String> = known
        .iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    files.sort();

    let cache = WatcherCache {
        updated_at: chrono::Utc::now().to_rfc3339(),
        files,
    };

    match serde_json::to_string(&cache) {
        Ok(json) => {
            if let Err(e) = std::fs::write(cache_path, json) {
                debug!("Failed to write watcher cache: {e}");
            }
        }
        Err(e) => debug!("Failed to serialize watcher cache: {e}"),
    }
}
