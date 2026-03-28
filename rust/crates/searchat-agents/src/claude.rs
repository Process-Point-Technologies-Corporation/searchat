//! Claude Code JSONL transcript provider.
//!
//! Transcript files live under `~/.claude/projects/**/*.jsonl`.
//! Each line is a JSON object. Lines with `"type": "session_meta"` are
//! skipped; all other lines may carry `"type": "user"` or `"type": "assistant"`
//! messages.
//!
//! The `content` field inside a message can be either:
//!   - a plain string, or
//!   - an array of content blocks `[{"type": "text", "text": "..."}]`

use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use searchat_models::{ConversationRecord, MessageRecord};
use serde_json::Value;

use crate::{extract_code_blocks, AgentError, AgentProvider, Result};

/// Provider for Claude Code JSONL transcripts.
pub struct ClaudeProvider;

impl AgentProvider for ClaudeProvider {
    fn id(&self) -> &str {
        "claude"
    }

    fn label(&self) -> &str {
        "Claude Code"
    }

    fn discover_dirs(&self) -> Vec<PathBuf> {
        let mut dirs = Vec::new();

        // Primary: ~/.claude/projects
        if let Some(home) = dirs::home_dir() {
            let projects = home.join(".claude").join("projects");
            if projects.exists() {
                dirs.push(projects);
            } else {
                // Fallback: ~/.claude itself
                let base = home.join(".claude");
                if base.exists() {
                    dirs.push(base);
                }
            }
        }

        // Additional dirs from environment variable
        if let Ok(extra) = std::env::var("SEARCHAT_ADDITIONAL_DIRS") {
            for part in extra.split(if cfg!(windows) { ';' } else { ':' }) {
                let p = PathBuf::from(part.trim());
                if p.exists() && !dirs.contains(&p) {
                    dirs.push(p);
                }
            }
        }

        dirs
    }

    fn matches_file(&self, path: &Path) -> bool {
        // Must be a .jsonl file
        if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
            return false;
        }
        // If the path contains /.codex/ it belongs to Codex, not Claude
        let norm = path.to_string_lossy().replace('\\', "/").to_lowercase();
        if norm.contains("/.codex/") {
            return false;
        }
        // If file doesn't exist yet, assume Claude (caller knows what they're doing)
        if !path.exists() {
            return true;
        }
        // Peek at first valid line: Codex files start with `"type": "session_meta"`
        if let Ok(first) = first_valid_line(path) {
            return first
                .get("type")
                .and_then(|v| v.as_str())
                .map(|t| t != "session_meta")
                .unwrap_or(true);
        }
        true
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

        let conversation_id = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_owned();
        let project_id = path
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_owned();

        let mut title = String::from("Untitled");
        let mut messages: Vec<MessageRecord> = Vec::new();
        let mut full_text_parts: Vec<String> = Vec::new();

        for entry in &entries {
            let msg_type = entry.get("type").and_then(|v| v.as_str()).unwrap_or("");
            if msg_type != "user" && msg_type != "assistant" {
                continue;
            }

            let content_raw = entry
                .get("message")
                .and_then(|m| m.get("content"))
                .cloned()
                .unwrap_or(Value::String(String::new()));
            let content = extract_text(&content_raw);

            // First non-empty user/assistant message is the title
            if title == "Untitled" && !content.is_empty() {
                title = content.chars().take(100).collect();
            }

            let timestamp = entry
                .get("timestamp")
                .and_then(|v| v.as_str())
                .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(Utc::now);

            let code_blocks = extract_code_blocks(&content);
            let has_code = !code_blocks.is_empty();

            full_text_parts.push(content.clone());
            messages.push(MessageRecord {
                sequence: messages.len() as i64,
                role: msg_type.to_owned(),
                content,
                timestamp,
                has_code,
                code_blocks,
            });
        }

        let now = Utc::now();
        let created_at = messages.first().map(|m| m.timestamp).unwrap_or(now);
        let updated_at = messages.last().map(|m| m.timestamp).unwrap_or(now);
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
            if let Some(cwd) = entry.get("cwd").and_then(|v| v.as_str()) {
                return Some(cwd.to_owned());
            }
        }
        None
    }

    fn build_resume_command(&self, session_id: &str) -> String {
        format!("claude --resume {session_id}")
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Parse every line of a JSONL byte buffer; skip and warn on malformed lines.
pub(crate) fn parse_jsonl_lines(raw: &[u8], path: &Path) -> Vec<Value> {
    let text = match std::str::from_utf8(raw) {
        Ok(s) => s,
        Err(e) => {
            log::warn!("UTF-8 decode error in {}: {e}", path.display());
            return Vec::new();
        }
    };
    let mut out = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        match serde_json::from_str::<Value>(trimmed) {
            Ok(v) => out.push(v),
            Err(e) => {
                log::warn!(
                    "Skipping malformed JSON at {}:{}: {e}",
                    path.display(),
                    lineno + 1
                );
            }
        }
    }
    out
}

/// Read and parse only the first valid line of a JSONL file.
fn first_valid_line(path: &Path) -> std::result::Result<Value, ()> {
    let raw = std::fs::read(path).map_err(|_| ())?;
    let text = std::str::from_utf8(&raw).map_err(|_| ())?;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        return serde_json::from_str::<Value>(trimmed).map_err(|_| ());
    }
    Err(())
}

/// Extract plain text from a Claude `content` value.
///
/// Handles both the string form and the array-of-blocks form.
pub(crate) fn extract_text(raw: &Value) -> String {
    match raw {
        Value::String(s) => s.clone(),
        Value::Array(blocks) => {
            let parts: Vec<String> = blocks
                .iter()
                .filter_map(|block| {
                    let obj = block.as_object()?;
                    if obj.get("type").and_then(|v| v.as_str()) == Some("text") {
                        obj.get("text").and_then(|v| v.as_str()).map(str::to_owned)
                    } else {
                        None
                    }
                })
                .collect();
            parts.join("\n\n")
        }
        _ => String::new(),
    }
}
