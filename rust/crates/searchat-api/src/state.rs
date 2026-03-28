use std::sync::Arc;

use parking_lot::RwLock;
use searchat_config::settings::Config;
use searchat_storage::UnifiedStorage;
use tokio::sync::Mutex as TokioMutex;

/// Shared application state injected into every Axum handler via `State<AppState>`.
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Config>,
    pub storage: Arc<UnifiedStorage>,
    /// Cached list of distinct project IDs; None means "needs refresh".
    pub projects_cache: Arc<RwLock<Option<Vec<String>>>>,
    /// Serialises concurrent index_missing calls.
    pub indexing_lock: Arc<TokioMutex<()>>,
    /// Whether the filesystem watcher is currently running.
    pub watcher_running: Arc<RwLock<bool>>,
    /// Number of conversations indexed since server start.
    pub indexed_since_start: Arc<RwLock<u64>>,
    /// ISO-8601 timestamp of the last watcher update (if any).
    pub last_watcher_update: Arc<RwLock<Option<String>>>,
    /// Serialised index.html content.
    pub index_html: Arc<String>,
    /// Serialised conversation.html content.
    pub conversation_html: Arc<String>,
}
