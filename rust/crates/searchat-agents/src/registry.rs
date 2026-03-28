//! Registry of all known agent providers.
//!
//! Match priority follows the Python registry order: Codex > Claude > Vibe.
//! Codex files (`.jsonl` under `/.codex/`) must be checked before Claude
//! because both use the `.jsonl` extension.

use std::path::{Path, PathBuf};

use crate::{AgentProvider, ClaudeProvider, CodexProvider, VibeProvider};

/// Return one instance of every registered provider in match-priority order.
///
/// Priority: Codex → Claude → Vibe
pub fn all_providers() -> Vec<Box<dyn AgentProvider>> {
    vec![
        Box::new(CodexProvider),
        Box::new(ClaudeProvider),
        Box::new(VibeProvider),
    ]
}

/// Return the provider that claims `path`, or `None` if no provider matches.
///
/// Providers are checked in priority order (Codex > Claude > Vibe).
pub fn detect_provider(path: &Path) -> Option<Box<dyn AgentProvider>> {
    for provider in all_providers() {
        if provider.matches_file(path) {
            return Some(provider);
        }
    }
    None
}

/// Return all discovered `(provider_id, directory)` pairs across every provider.
///
/// Directories that don't exist on the current machine are included for Codex
/// WSL paths (they may still be walkable) but skipped for others.
pub fn iter_dirs() -> Vec<(String, PathBuf)> {
    all_providers()
        .into_iter()
        .flat_map(|p| {
            let id = p.id().to_owned();
            p.discover_dirs().into_iter().map(move |dir| (id.clone(), dir))
        })
        .collect()
}
