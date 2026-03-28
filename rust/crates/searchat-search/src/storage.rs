/// Trait that `UnifiedSearchEngine` calls into for all data access.
///
/// The concrete implementation lives in `searchat-storage` (Phase 2).
/// This trait definition lets `searchat-search` compile independently
/// and lets unit tests provide mock implementations.
use std::collections::HashMap;

use crate::error::SearchError;

/// A raw row returned by palace semantic search.
/// Field names mirror the DuckDB column names used in the Python implementation.
#[derive(Debug, Clone)]
pub struct PalaceRow {
    pub exchange_id: String,
    pub conversation_id: String,
    pub project_id: String,
    pub score: f64,
    /// Distilled text summary.
    pub exchange_core: Option<String>,
    pub specific_context: Option<String>,
    /// JSON array of file-touch dicts (as raw serde_json::Value).
    pub files_touched: Option<serde_json::Value>,
    pub object_id: Option<String>,
    /// Optional ply positions (may be absent on palace-only rows).
    pub ply_start: Option<i64>,
    pub ply_end: Option<i64>,
    /// Conversation metadata — populated after enrichment.
    pub title: Option<String>,
    pub file_path: Option<String>,
    pub message_count: Option<i64>,
    pub updated_at: Option<chrono::DateTime<chrono::Utc>>,
    pub created_at: Option<chrono::DateTime<chrono::Utc>>,
}

/// A raw row returned by verbatim (BM25 or semantic) search.
/// Field names mirror the Python `search_verbatim_bm25` / `search_verbatim_semantic` returns.
#[derive(Debug, Clone)]
pub struct VerbatimRow {
    pub exchange_id: String,
    pub conversation_id: String,
    pub project_id: String,
    pub score: f64,
    pub exchange_text: Option<String>,
    pub title: Option<String>,
    pub file_path: Option<String>,
    pub message_count: Option<i64>,
    pub updated_at: Option<chrono::DateTime<chrono::Utc>>,
    pub created_at: Option<chrono::DateTime<chrono::Utc>>,
    pub ply_start: Option<i64>,
    pub ply_end: Option<i64>,
}

/// Batch conversation metadata returned by `get_conversations_batch`.
#[derive(Debug, Clone)]
pub struct ConversationMeta {
    pub title: String,
    pub file_path: String,
    pub message_count: i64,
    pub updated_at: Option<chrono::DateTime<chrono::Utc>>,
    pub created_at: Option<chrono::DateTime<chrono::Utc>>,
}

/// Storage backend trait consumed by `UnifiedSearchEngine`.
///
/// All methods are synchronous (DuckDB is synchronous; async is handled at
/// the API layer above this engine).
pub trait StorageBackend: Send + Sync {
    /// BM25 full-text search over exchange text.
    fn search_verbatim_bm25(
        &self,
        query: &str,
        limit: usize,
        project_ids: Option<&[String]>,
    ) -> Result<Vec<VerbatimRow>, SearchError>;

    /// Vector (HNSW) search over exchange embeddings.
    fn search_verbatim_semantic(
        &self,
        query_embedding: &[f32],
        limit: usize,
        project_ids: Option<&[String]>,
    ) -> Result<Vec<VerbatimRow>, SearchError>;

    /// Vector (HNSW) search over distilled palace object embeddings.
    fn search_palace_semantic(
        &self,
        query_embedding: &[f32],
        limit: usize,
        project_ids: Option<&[String]>,
    ) -> Result<Vec<PalaceRow>, SearchError>;

    /// Batch-fetch conversation metadata for a list of conversation IDs.
    fn get_conversations_batch(
        &self,
        conversation_ids: &[String],
    ) -> Result<HashMap<String, ConversationMeta>, SearchError>;

    /// Return basic statistics (count of conversations / exchanges / objects).
    fn get_stats(&self) -> Result<HashMap<String, serde_json::Value>, SearchError>;
}

/// Trait for query embedding.
pub trait EmbedderBackend: Send + Sync {
    /// Encode a text string into a dense embedding vector.
    fn encode(&self, text: &str) -> Result<Vec<f32>, SearchError>;
}
