//! Codex session JSONL transcript provider.
//!
//! Transcript files live under `~/.codex/sessions/**/*.jsonl`.
//! File structure:
//!   - First line: `{"type": "session_meta", "payload": {"id": "...", "cwd": "...", "timestamp": "..."}}`
//!   - Subsequent lines: `{"type": "response_item", "timestamp": "...", "payload": {"type": "message", "role": "user"|"assistant", "content": [...]}}`
//!
//! Content blocks use `"type": "input_text"` or `"type": "output_text"` (not `"text"`).

use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use searchat_models::{ConversationRecord, MessageRecord};
use serde_json::Value;

use crate::{claude::parse_jsonl_lines, extract_code_blocks, AgentError, AgentProvider, Result};

/// Provider for Codex session JSONL transcripts.
pub struct CodexProvider;

impl AgentProvider for CodexProvider {
    fn id(&self) -> &str {
        "codex"
    }

    fn label(&self) -> &str {
        "Codex"
    }

    fn discover_dirs(&self) -> Vec<PathBuf> {
        let mut dirs = Vec::new();

        // Primary: ~/.codex/sessions
        if let Some(home) = dirs::home_dir() {
            let sessions = home.join(".codex").join("sessions");
            if sessions.exists() {
                dirs.push(sessions);
            }
        }

        // On Windows also probe WSL paths
        #[cfg(windows)]
        {
            let username = std::env::var("USERNAME")
                .or_else(|_| std::env::var("USER"))
                .unwrap_or_else(|_| "user".to_owned());
            let wsl_paths = [
                format!("\\\\wsl.localhost\\Ubuntu\\home\\{username}\\.codex\\sessions"),
                format!("\\\\wsl$\\Ubuntu\\home\\{username}\\.codex\\sessions"),
            ];
            for p in wsl_paths {
                let pb = PathBuf::from(&p);
                if !dirs.contains(&pb) {
                    dirs.push(pb);
                }
            }
        }

        dirs
    }

    fn matches_file(&self, path: &Path) -> bool {
        if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
            return false;
        }
        // Path-based: anything under /.codex/ is Codex
        let norm = path.to_string_lossy().replace('\\', "/").to_lowercase();
        if norm.contains("/.codex/") {
            return true;
        }
        // Content-based: first valid line has `"type": "session_meta"`
        if !path.exists() {
            return false;
        }
        if let Ok(raw) = std::fs::read(path) {
            let entries = parse_jsonl_lines(&raw, path);
            if let Some(first) = entries.first() {
                return first
                    .get("type")
                    .and_then(|v| v.as_str())
                    .map(|t| t == "session_meta")
                    .unwrap_or(false);
            }
        }
        false
    }

    fn parse_conversation(&self, path: &Path) -> Result<ConversationRecord> {
        let raw = std::fs::read(path).map_err(|e| AgentError::Io {
            path: path.to_owned(),
            source: e,
        })?;
        let file_size = raw.len() as u64;
        let mtime_ns = std::fs::metadata(path)
            .ok()
            .and_then(|m| {
                use std::time::SystemTime;
                m.modified()
                    .ok()
                    .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
                    .map(|d| d.as_nanos() as u64)
            })
            .unwrap_or(0);

        let entries = parse_jsonl_lines(&raw, path);
        if entries.is_empty() {
            return Err(AgentError::EmptyFile(path.to_owned()));
        }

        // Extract session metadata from the first session_meta line
        let session_meta = entries
            .iter()
            .find(|e| e.get("type").and_then(|v| v.as_str()) == Some("session_meta"))
            .and_then(|e| e.get("payload"))
            .cloned()
            .unwrap_or(Value::Object(serde_json::Map::new()));

        let conversation_id = session_meta
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or_else(|| {
                path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
            })
            .to_owned();

        let cwd = session_meta
            .get("cwd")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_owned();
        let project_name = if cwd.is_empty() {
            "codex-session".to_owned()
        } else {
            Path::new(&cwd)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("codex-session")
                .to_owned()
        };
        let project_id = format!("codex-{project_name}");

        let meta_timestamp = session_meta
            .get("timestamp")
            .and_then(|v| v.as_str())
            .and_then(parse_timestamp);
        let now = Utc::now();
        let mut created_at = meta_timestamp.unwrap_or(now);
        let mut updated_at = created_at;

        let mut title = String::from("Untitled Codex Session");
        let mut messages: Vec<MessageRecord> = Vec::new();
        let mut full_text_parts: Vec<String> = Vec::new();

        for entry in &entries {
            if entry.get("type").and_then(|v| v.as_str()) != Some("response_item") {
                continue;
            }
            let payload = match entry.get("payload") {
                Some(p) => p,
                None => continue,
            };
            if payload.get("type").and_then(|v| v.as_str()) != Some("message") {
                continue;
            }
            let role = match payload.get("role").and_then(|v| v.as_str()) {
                Some(r) if r == "user" || r == "assistant" => r,
                _ => continue,
            };
            let content_blocks = payload
                .get("content")
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default();
            let content = extract_codex_text(&content_blocks);
            if content.is_empty() {
                continue;
            }

            if role == "user" && title == "Untitled Codex Session" {
                title = content
                    .chars()
                    .take(100)
                    .collect::<String>()
                    .replace('\n', " ")
                    .trim()
                    .to_owned();
            }

            let timestamp = entry
                .get("timestamp")
                .and_then(|v| v.as_str())
                .and_then(parse_timestamp)
                .unwrap_or(now);
            updated_at = timestamp;

            let code_blocks = extract_code_blocks(&content);
            let has_code = !code_blocks.is_empty();

            full_text_parts.push(content.clone());
            messages.push(MessageRecord {
                sequence: messages.len() as i64,
                role: role.to_owned(),
                content,
                timestamp,
                has_code,
                code_blocks,
            });
        }

        if let Some(first) = messages.first() {
            created_at = first.timestamp;
        }
        if let Some(last) = messages.last() {
            updated_at = last.timestamp;
        }

        let message_count = messages.len() as i64;

        Ok(ConversationRecord {
            conversation_id,
            project_id,
            file_path: path.to_string_lossy().into_owned(),
            title,
            created_at,
            updated_at,
            message_count,
            messages,
            full_text: full_text_parts.join("\n\n"),
            embedding_id: -1,
            file_hash: String::new(),
            indexed_at: now,
            file_size: file_size as i64,
            mtime_ns: mtime_ns as i64,
        })
    }

    fn extract_cwd(&self, path: &Path) -> Option<String> {
        let raw = std::fs::read(path).ok()?;
        for entry in parse_jsonl_lines(&raw, path) {
            if entry.get("type").and_then(|v| v.as_str()) == Some("session_meta") {
                return entry
                    .get("payload")
                    .and_then(|p| p.get("cwd"))
                    .and_then(|v| v.as_str())
                    .map(str::to_owned);
            }
        }
        None
    }

    fn build_resume_command(&self, session_id: &str) -> String {
        format!("codex resume {session_id}")
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Parse a Codex content block array.
/// Blocks use `"type": "input_text"` or `"type": "output_text"`.
fn extract_codex_text(blocks: &[Value]) -> String {
    let parts: Vec<String> = blocks
        .iter()
        .filter_map(|block| {
            let obj = block.as_object()?;
            let block_type = obj.get("type").and_then(|v| v.as_str())?;
            if block_type == "input_text" || block_type == "output_text" {
                obj.get("text")
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .map(str::to_owned)
            } else {
                None
            }
        })
        .collect();
    parts.join("\n\n").trim().to_owned()
}

/// Parse an ISO-8601 / RFC-3339 timestamp, accepting the `Z` suffix.
pub(crate) fn parse_timestamp(s: &str) -> Option<DateTime<Utc>> {
    let normalized = s.replace('Z', "+00:00");
    DateTime::parse_from_rfc3339(&normalized)
        .ok()
        .map(|dt| dt.with_timezone(&Utc))
}
