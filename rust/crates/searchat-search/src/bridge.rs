/// Concrete implementations of [`StorageBackend`] and [`EmbedderBackend`]
/// that delegate to `searchat_storage::UnifiedStorage` and
/// `searchat_embed::Embedder` respectively.
///
/// These are thin adapters: they map the storage/embed result types and error
/// types into the search-crate's own types so that `UnifiedSearchEngine` can
/// be instantiated with the real backends.
use std::collections::HashMap;

use crate::error::SearchError;
use crate::storage::{
    ConversationMeta, EmbedderBackend, PalaceRow, StorageBackend, VerbatimRow,
};

// ---------------------------------------------------------------------------
// StorageBackend for UnifiedStorage
// ---------------------------------------------------------------------------

impl StorageBackend for searchat_storage::UnifiedStorage {
    fn search_verbatim_bm25(
        &self,
        query: &str,
        limit: usize,
        project_ids: Option<&[String]>,
    ) -> Result<Vec<VerbatimRow>, SearchError> {
        self.search_verbatim_bm25(query, limit, project_ids)
            .map(|rows| rows.into_iter().map(verbatim_row_from_storage).collect())
            .map_err(|e| SearchError::Storage(e.to_string()))
    }

    fn search_verbatim_semantic(
        &self,
        query_embedding: &[f32],
        limit: usize,
        project_ids: Option<&[String]>,
    ) -> Result<Vec<VerbatimRow>, SearchError> {
        self.search_verbatim_semantic(query_embedding, limit, project_ids)
            .map(|rows| rows.into_iter().map(verbatim_row_from_storage).collect())
            .map_err(|e| SearchError::Storage(e.to_string()))
    }

    fn search_palace_semantic(
        &self,
        query_embedding: &[f32],
        limit: usize,
        project_ids: Option<&[String]>,
    ) -> Result<Vec<PalaceRow>, SearchError> {
        self.search_palace_semantic(query_embedding, limit, project_ids)
            .map(|rows| rows.into_iter().map(palace_row_from_storage).collect())
            .map_err(|e| SearchError::Storage(e.to_string()))
    }

    fn get_conversations_batch(
        &self,
        conversation_ids: &[String],
    ) -> Result<HashMap<String, ConversationMeta>, SearchError> {
        self.get_conversations_batch(conversation_ids)
            .map(|map| {
                map.into_iter()
                    .map(|(id, row)| (id, conv_meta_from_storage(row)))
                    .collect()
            })
            .map_err(|e| SearchError::Storage(e.to_string()))
    }

    fn get_stats(&self) -> Result<HashMap<String, serde_json::Value>, SearchError> {
        self.get_stats()
            .map(storage_stats_to_map)
            .map_err(|e| SearchError::Storage(e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// EmbedderBackend for Embedder
// ---------------------------------------------------------------------------

impl EmbedderBackend for searchat_embed::Embedder {
    fn encode(&self, text: &str) -> Result<Vec<f32>, SearchError> {
        self.encode(text)
            .map_err(|e| SearchError::Embedding(e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// Row converters
// ---------------------------------------------------------------------------

fn verbatim_row_from_storage(r: searchat_storage::VerbatimSearchResult) -> VerbatimRow {
    VerbatimRow {
        exchange_id: r.exchange_id,
        conversation_id: r.conversation_id,
        project_id: r.project_id.unwrap_or_default(),
        score: r.score,
        exchange_text: if r.exchange_text.is_empty() {
            None
        } else {
            Some(r.exchange_text)
        },
        title: if r.title.is_empty() {
            None
        } else {
            Some(r.title)
        },
        file_path: if r.file_path.is_empty() {
            None
        } else {
            Some(r.file_path)
        },
        message_count: Some(r.message_count as i64),
        updated_at: Some(r.updated_at),
        created_at: Some(r.created_at),
        ply_start: Some(r.ply_start as i64),
        ply_end: Some(r.ply_end as i64),
    }
}

fn palace_row_from_storage(r: searchat_storage::PalaceSearchResult) -> PalaceRow {
    PalaceRow {
        exchange_id: r.exchange_id.unwrap_or_default(),
        conversation_id: r.conversation_id,
        project_id: r.project_id,
        score: r.score,
        exchange_core: if r.exchange_core.is_empty() {
            None
        } else {
            Some(r.exchange_core)
        },
        specific_context: if r.specific_context.is_empty() {
            None
        } else {
            Some(r.specific_context)
        },
        files_touched: if r.files_touched.is_empty() {
            None
        } else {
            Some(serde_json::Value::Array(r.files_touched))
        },
        object_id: Some(r.object_id),
        ply_start: Some(r.ply_start as i64),
        ply_end: Some(r.ply_end as i64),
        title: None,
        file_path: None,
        message_count: None,
        updated_at: None,
        created_at: None,
    }
}

fn conv_meta_from_storage(r: searchat_storage::ConversationRow) -> ConversationMeta {
    ConversationMeta {
        title: r.title,
        file_path: r.file_path,
        message_count: r.message_count as i64,
        updated_at: Some(r.updated_at),
        created_at: Some(r.created_at),
    }
}

fn storage_stats_to_map(
    s: searchat_storage::StorageStats,
) -> HashMap<String, serde_json::Value> {
    let mut m = HashMap::new();
    m.insert("conversations".into(), serde_json::Value::from(s.conversations));
    m.insert("messages".into(), serde_json::Value::from(s.messages));
    m.insert("exchanges".into(), serde_json::Value::from(s.exchanges));
    m.insert(
        "verbatim_embeddings".into(),
        serde_json::Value::from(s.verbatim_embeddings),
    );
    m.insert(
        "palace_objects".into(),
        serde_json::Value::from(s.palace_objects),
    );
    m.insert("rooms".into(), serde_json::Value::from(s.rooms));
    m.insert(
        "facet_embeddings".into(),
        serde_json::Value::from(s.facet_embeddings),
    );
    m.insert(
        "hierarchical_facets".into(),
        serde_json::Value::from(s.hierarchical_facets),
    );
    m.insert(
        "vss_available".into(),
        serde_json::Value::from(s.vss_available),
    );
    m.insert(
        "fts_available".into(),
        serde_json::Value::from(s.fts_available),
    );
    m
}
