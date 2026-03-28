use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageRecord {
    pub sequence: i64,
    pub role: String,
    pub content: String,
    pub timestamp: DateTime<Utc>,
    pub has_code: bool,
    #[serde(default)]
    pub code_blocks: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationRecord {
    pub conversation_id: String,
    pub project_id: String,
    pub file_path: String,
    pub title: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub message_count: i64,
    pub messages: Vec<MessageRecord>,
    pub full_text: String,
    pub embedding_id: i64,
    pub file_hash: String,
    pub indexed_at: DateTime<Utc>,
    #[serde(default)]
    pub file_size: i64,
    #[serde(default)]
    pub mtime_ns: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchFilters {
    #[serde(default)]
    pub project_ids: Option<Vec<String>>,
    #[serde(default)]
    pub date_from: Option<DateTime<Utc>>,
    #[serde(default)]
    pub date_to: Option<DateTime<Utc>>,
    #[serde(default)]
    pub min_messages: i64,
    #[serde(default)]
    pub has_code: Option<bool>,
}

impl Default for SearchFilters {
    fn default() -> Self {
        Self {
            project_ids: None,
            date_from: None,
            date_to: None,
            min_messages: 0,
            has_code: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub conversation_id: String,
    pub project_id: String,
    pub title: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub message_count: i64,
    pub file_path: String,
    pub score: f64,
    pub snippet: String,
    #[serde(default)]
    pub message_start_index: Option<i64>,
    #[serde(default)]
    pub message_end_index: Option<i64>,
    #[serde(default)]
    pub bm25_score: Option<f64>,
    #[serde(default)]
    pub semantic_score: Option<f64>,
    #[serde(default)]
    pub exchange_id: Option<String>,
    #[serde(default)]
    pub exchange_text: Option<String>,
    /// "legacy" | "unified" | "both"
    #[serde(default)]
    pub match_source: Option<String>,
    #[serde(default)]
    pub palace_summary: Option<String>,
    #[serde(default)]
    pub palace_context: Option<String>,
    #[serde(default)]
    pub files_touched_raw: Option<Vec<HashMap<String, String>>>,
    #[serde(default)]
    pub object_id: Option<String>,
    #[serde(default)]
    pub search_metadata: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResults {
    pub results: Vec<SearchResult>,
    pub total_count: i64,
    pub search_time_ms: f64,
    pub mode_used: String,
    #[serde(default)]
    pub error: Option<String>,
}

/// Statistics returned by indexing operations.
/// All fields optional (mirrors Python `IndexingStats(TypedDict, total=False)`).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IndexingStats {
    #[serde(default)]
    pub new_conversations: Option<i64>,
    #[serde(default)]
    pub updated_conversations: Option<i64>,
    #[serde(default)]
    pub exchanges_created: Option<i64>,
    #[serde(default)]
    pub embeddings_created: Option<i64>,
    #[serde(default)]
    pub skipped_already_indexed: Option<i64>,
    #[serde(default)]
    pub skipped_errors: Option<i64>,
    #[serde(default)]
    pub skipped_existing: Option<i64>,
    #[serde(default)]
    pub skipped_empty: Option<i64>,
    #[serde(default)]
    pub invalid_transcript_count: Option<i64>,
    #[serde(default)]
    pub invalid_transcript_examples: Option<Vec<String>>,
    #[serde(default)]
    pub skipped_known_invalid: Option<i64>,
    #[serde(default)]
    pub append_only_updates: Option<i64>,
    #[serde(default)]
    pub total_files: Option<i64>,
    #[serde(default)]
    pub changed_detected: Option<i64>,
    #[serde(default)]
    pub parse_seconds: Option<f64>,
    #[serde(default)]
    pub encode_seconds: Option<f64>,
    #[serde(default)]
    pub store_seconds: Option<f64>,
    #[serde(default)]
    pub time_seconds: Option<f64>,
    #[serde(default)]
    pub conversations_processed: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DateFilter {
    #[serde(default)]
    pub from_date: Option<DateTime<Utc>>,
    #[serde(default)]
    pub to_date: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedQuery {
    pub original: String,
    #[serde(default)]
    pub must_include: Vec<String>,
    #[serde(default)]
    pub should_include: Vec<String>,
    #[serde(default)]
    pub must_exclude: Vec<String>,
    #[serde(default)]
    pub exact_phrases: Vec<String>,
    #[serde(default)]
    pub date_filter: Option<DateFilter>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileTouched {
    pub path: String,
    /// One of: read | modified | created | deleted | discussed
    pub action: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistilledObject {
    pub object_id: String,
    pub project_id: String,
    pub conversation_id: String,
    pub ply_start: i64,
    pub ply_end: i64,
    pub files_touched: Vec<FileTouched>,
    pub exchange_core: String,
    pub specific_context: String,
    pub created_at: DateTime<Utc>,
    pub exchange_at: DateTime<Utc>,
    pub embedding_id: i64,
    pub distilled_text: String,
    #[serde(default)]
    pub conv_title: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Room {
    pub room_id: String,
    /// One of: file | module | concept | tool | workflow
    pub room_type: String,
    pub room_key: String,
    pub room_label: String,
    pub project_id: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub object_count: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoomObject {
    pub room_id: String,
    pub object_id: String,
    pub relevance: f64,
    pub placed_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PalaceSearchResult {
    pub object_id: String,
    pub conversation_id: String,
    pub project_id: String,
    pub ply_start: i64,
    pub ply_end: i64,
    pub exchange_core: String,
    pub specific_context: String,
    pub files_touched: Vec<FileTouched>,
    pub rooms: Vec<Room>,
    pub score: f64,
    #[serde(default)]
    pub keyword_score: f64,
    #[serde(default)]
    pub semantic_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedSearchResult {
    pub conversation_id: String,
    pub project_id: String,
    pub title: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub message_count: i64,
    pub file_path: String,
    pub combined_score: f64,

    // Palace layer (optional)
    #[serde(default)]
    pub palace_score: Option<f64>,
    #[serde(default)]
    pub palace_summary: Option<String>,
    #[serde(default)]
    pub palace_context: Option<String>,
    #[serde(default)]
    pub rooms: Vec<Room>,
    #[serde(default)]
    pub files_touched: Vec<FileTouched>,
    #[serde(default)]
    pub ply_start: Option<i64>,
    #[serde(default)]
    pub ply_end: Option<i64>,
    #[serde(default)]
    pub object_id: Option<String>,

    // Verbatim layer (optional)
    #[serde(default)]
    pub verbatim_score: Option<f64>,
    #[serde(default)]
    pub verbatim_snippet: Option<String>,
    #[serde(default)]
    pub message_start_index: Option<i64>,
    #[serde(default)]
    pub message_end_index: Option<i64>,

    // Sub-scores for analysis
    #[serde(default)]
    pub palace_bm25_score: Option<f64>,
    #[serde(default)]
    pub palace_semantic_score: Option<f64>,
    #[serde(default)]
    pub verbatim_bm25_score: Option<f64>,
    #[serde(default)]
    pub verbatim_semantic_score: Option<f64>,

    /// Progressive fallback tier: "scoped" | "related" | "unscoped" | None
    #[serde(default)]
    pub fallback_tier: Option<String>,
}

impl UnifiedSearchResult {
    pub fn has_palace(&self) -> bool {
        self.palace_score.is_some()
    }

    pub fn has_verbatim(&self) -> bool {
        self.verbatim_score.is_some()
    }

    pub fn is_intersection(&self) -> bool {
        self.has_palace() && self.has_verbatim()
    }
}
