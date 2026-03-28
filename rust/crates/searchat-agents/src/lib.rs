//! Agent-specific conversation file parsers for Claude Code, Codex, and Vibe.
//!
//! Each provider implements the [`AgentProvider`] trait and handles discovery,
//! file matching, and JSONL/JSON parsing for its own transcript format.

pub mod claude;
pub mod codex;
pub mod registry;
pub mod vibe;

use std::path::{Path, PathBuf};

use searchat_models::ConversationRecord;
use thiserror::Error;

pub use claude::ClaudeProvider;
pub use codex::CodexProvider;
pub use registry::{all_providers, detect_provider, iter_dirs};
pub use vibe::VibeProvider;

/// Errors that can occur during conversation parsing.
#[derive(Debug, Error)]
pub enum AgentError {
    #[error("I/O error reading {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("JSON parse error in {path}: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("No valid messages found in {0}")]
    NoMessages(PathBuf),
    #[error("File has no valid JSON lines: {0}")]
    EmptyFile(PathBuf),
}

pub type Result<T> = std::result::Result<T, AgentError>;

/// Core interface implemented by each agent transcript provider.
///
/// Match priority in the registry is: Codex > Claude > Vibe, matching the
/// Python registry's `_PROVIDERS` ordering.
pub trait AgentProvider: Send + Sync {
    /// Short identifier for this provider ("claude", "codex", "vibe").
    fn id(&self) -> &str;

    /// Human-readable label ("Claude Code", "Codex", "Vibe").
    fn label(&self) -> &str;

    /// Return root directories where this provider stores transcripts.
    /// Directories that do not exist on the current system are omitted.
    fn discover_dirs(&self) -> Vec<PathBuf>;

    /// Return `true` when `path` belongs to this provider.
    ///
    /// For path-disambiguated providers (Codex under `/.codex/`, Vibe under
    /// `/.vibe/`) this is a pure path check.  For Claude, which shares the
    /// `.jsonl` extension with Codex, the first line of the file is inspected
    /// when the file exists.
    fn matches_file(&self, path: &Path) -> bool;

    /// Parse a transcript file into a [`ConversationRecord`].
    ///
    /// Malformed individual lines in JSONL files are skipped with a `log::warn`
    /// rather than propagating an error, unless the file contains no valid lines
    /// at all.
    fn parse_conversation(&self, path: &Path) -> Result<ConversationRecord>;

    /// Extract the working directory recorded in the session, if any.
    fn extract_cwd(&self, path: &Path) -> Option<String>;

    /// Build the CLI command used to resume this session.
    fn build_resume_command(&self, session_id: &str) -> String;
}

// ---------------------------------------------------------------------------
// Shared helpers used by multiple providers
// ---------------------------------------------------------------------------

/// Extract code fences from a markdown-ish string.
/// Returns the inner content of every ``` block.
pub(crate) fn extract_code_blocks(text: &str) -> Vec<String> {
    // Matches ```optional_lang\n ... ``` (non-greedy, DOTALL)
    static RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let re = RE.get_or_init(|| {
        regex::Regex::new(r"```(?:\w+)?\n([\s\S]*?)```").expect("static regex is valid")
    });
    re.captures_iter(text)
        .map(|cap| cap[1].to_owned())
        .collect()
}
