//! Mistral Vibe JSON transcript provider.
//!
//! Transcript files live under `~/.vibe/logs/session/*.json`.
//! Unlike Claude/Codex they are regular JSON files (not JSONL) with the shape:
//!
//! ```json
//! {
//!   "metadata": {
//!     "session_id": "...",
//!     "start_time": "...",
//!     "end_time": "...",
//!     "environment": { "working_directory": "..." }
//!   },
//!   "messages": [
//!     { "role": "user", "content": "..." },
//!     { "role": "assistant", "content": "..." }
//!   ]
//! }
//! ```

use std::path::{Path, PathBuf};

use chrono::Utc;
use searchat_models::{ConversationRecord, MessageRecord};
use serde_json::Value;

use crate::{codex::parse_timestamp, extract_code_blocks, AgentError, AgentProvider, Result};

/// Provider for Vibe JSON session transcripts.
pub struct VibeProvider;

impl AgentProvider for VibeProvider {
    fn id(&self) -> &str {
        "vibe"
    }

    fn label(&self) -> &str {
        "Vibe"
    }

    fn discover_dirs(&self) -> Vec<PathBuf> {
        let mut dirs = Vec::new();

        // Primary: ~/.vibe/logs/session
        if let Some(home) = dirs::home_dir() {
            let session_dir = home.join(".vibe").join("logs").join("session");
            if session_dir.exists() {
                dirs.push(session_dir);
            }
        }

        // Optional: $VIBE_HOME/logs/session
        if let Ok(vibe_home) = std::env::var("VIBE_HOME") {
            let custom = PathBuf::from(vibe_home).join("logs").join("session");
            if custom.exists() && !dirs.contains(&custom) {
                dirs.push(custom);
            }
        }

        dirs
    }

    fn matches_file(&self, path: &Path) -> bool {
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            return false;
        }
        // Path-based: anything under /.vibe/ is Vibe
        let norm = path.to_string_lossy().replace('\\', "/").to_lowercase();
        if norm.contains("/.vibe/") {
            return true;
        }
        // Content-based: JSON object with both "metadata" and "messages" keys
        if !path.exists() {
            return false;
        }
        if let Ok(text) = std::fs::read_to_string(path) {
            if let Ok(v) = serde_json::from_str::<Value>(&text) {
                return v.get("metadata").is_some() && v.get("messages").is_some();
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

        let data: Value =
            serde_json::from_slice(&raw).map_err(|e| AgentError::Json {
                path: path.to_owned(),
                source: e,
            })?;

        let metadata = data
            .get("metadata")
            .cloned()
            .unwrap_or(Value::Object(serde_json::Map::new()));

        let session_id = metadata
            .get("session_id")
            .and_then(|v| v.as_str())
            .unwrap_or_else(|| {
                path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
            })
            .to_owned();

        let working_dir = metadata
            .get("environment")
            .and_then(|e| e.get("working_directory"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_owned();
        let project_name = if working_dir.is_empty() {
            "vibe-session".to_owned()
        } else {
            Path::new(&working_dir)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("vibe-session")
                .to_owned()
        };
        let project_id = format!("vibe-{project_name}");

        let now = Utc::now();
        let created_at = metadata
            .get("start_time")
            .and_then(|v| v.as_str())
            .and_then(parse_timestamp)
            .unwrap_or(now);
        let updated_at = metadata
            .get("end_time")
            .and_then(|v| v.as_str())
            .and_then(parse_timestamp)
            .unwrap_or(created_at);

        let raw_messages = data
            .get("messages")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        let mut title = String::from("Untitled Vibe Session");
        let mut messages: Vec<MessageRecord> = Vec::new();
        let mut full_text_parts: Vec<String> = Vec::new();

        for msg in &raw_messages {
            let role = match msg.get("role").and_then(|v| v.as_str()) {
                Some(r) if r == "user" || r == "assistant" => r,
                _ => continue,
            };
            let content = msg
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_owned();
            if content.is_empty() {
                continue;
            }

            if role == "user" && title == "Untitled Vibe Session" {
                title = content
                    .chars()
                    .take(100)
                    .collect::<String>()
                    .replace('\n', " ")
                    .trim()
                    .to_owned();
            }

            let code_blocks = extract_code_blocks(&content);
            let has_code = !code_blocks.is_empty();

            full_text_parts.push(content.clone());
            messages.push(MessageRecord {
                sequence: messages.len() as i64,
                role: role.to_owned(),
                content,
                // Vibe sessions have no per-message timestamps; use session start
                timestamp: created_at,
                has_code,
                code_blocks,
            });
        }

        let message_count = messages.len() as i64;

        Ok(ConversationRecord {
            conversation_id: session_id,
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
        let text = std::fs::read_to_string(path).ok()?;
        let data: Value = serde_json::from_str(&text).ok()?;
        data.get("metadata")
            .and_then(|m| m.get("environment"))
            .and_then(|e| e.get("working_directory"))
            .and_then(|v| v.as_str())
            .map(str::to_owned)
    }

    fn build_resume_command(&self, session_id: &str) -> String {
        format!("vibe --resume {session_id}")
    }
}
