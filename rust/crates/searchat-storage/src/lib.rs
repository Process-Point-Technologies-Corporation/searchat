/// Unified DuckDB storage — Rust port of `unified_storage.py`.
///
/// Provides the `UnifiedStorage` struct wrapping a single DuckDB connection
/// protected by a `parking_lot::Mutex`. Both reads and writes go through
/// the same connection; DuckDB's MVCC handles isolation.
pub mod error;
pub mod read;
pub mod schema;
pub mod write;

pub use error::{StorageError, StorageResult};
pub use read::{
    ConversationRow, ExchangeRow, FacetSearchResult, HierarchicalFacetSearchResult, MessageRow,
    PalaceSearchResult, StorageStats, VerbatimSearchResult,
};
pub use schema::EMBEDDING_DIM;
pub use write::ExchangeInput;

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use duckdb::Connection;
use parking_lot::Mutex;
use sha2::{Digest, Sha256};

use searchat_models::{ConversationRecord, DistilledObject, Room, RoomObject};

// ---------------------------------------------------------------------------
// Facet ID helper (mirrors Python's make_facet_id)
// ---------------------------------------------------------------------------

/// Deterministic facet ID from `type:text` — first 16 hex chars of SHA-256.
pub fn make_facet_id(facet_type: &str, facet_text: &str) -> String {
    let key = format!("{facet_type}:{facet_text}");
    let hash = Sha256::digest(key.as_bytes());
    hex::encode(&hash[..8])
}

// ---------------------------------------------------------------------------
// UnifiedStorage
// ---------------------------------------------------------------------------

/// Thread-safe DuckDB storage for all searchat data.
///
/// The inner `Connection` is protected by a `Mutex` because DuckDB's Rust
/// binding does not implement `Sync`. All public methods lock the mutex,
/// execute their query, and release immediately.
pub struct UnifiedStorage {
    conn: Mutex<Connection>,
    pub vss_available: bool,
    pub fts_available: bool,
    _db_path: Option<PathBuf>,
}

impl UnifiedStorage {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Open (or create) the database at `data_dir/searchat.duckdb`.
    pub fn open(data_dir: &Path) -> StorageResult<Self> {
        std::fs::create_dir_all(data_dir)?;
        let db_path = data_dir.join("searchat.duckdb");
        let conn = Connection::open(&db_path)?;
        Self::from_connection(conn, Some(db_path))
    }

    /// Use an existing connection (e.g. `:memory:` for tests).
    pub fn from_connection(conn: Connection, db_path: Option<PathBuf>) -> StorageResult<Self> {
        let (vss_available, fts_available) = load_extensions(&conn);

        if vss_available {
            // Disable HNSW persistence — the background thread it spawns
            // causes write-write conflicts with per-conversation transactions.
            let _ = conn.execute_batch("SET hnsw_enable_experimental_persistence = false");
        }

        let vss = vss_available;
        schema::ensure_tables(&conn, vss)?;

        Ok(Self {
            conn: Mutex::new(conn),
            vss_available,
            fts_available,
            _db_path: db_path,
        })
    }

    /// Create FTS indexes on exchanges and palace_objects.
    ///
    /// Uses `overwrite = 1` so it is safe to call after new data is indexed.
    pub fn create_fts_index(&self) -> StorageResult<()> {
        if !self.fts_available {
            return Err(StorageError::ExtensionUnavailable("fts".into()));
        }
        let conn = self.conn.lock();
        schema::create_fts_indexes(&conn)
    }

    // -----------------------------------------------------------------------
    // Conversation CRUD
    // -----------------------------------------------------------------------

    pub fn store_conversation(
        &self,
        record: &ConversationRecord,
        in_transaction: bool,
    ) -> StorageResult<()> {
        let conn = self.conn.lock();
        write::store_conversation(&conn, record, in_transaction)
    }

    pub fn get_conversation(
        &self,
        conversation_id: &str,
    ) -> StorageResult<Option<ConversationRow>> {
        let conn = self.conn.lock();
        read::get_conversation(&conn, conversation_id)
    }

    pub fn get_conversations_batch(
        &self,
        conversation_ids: &[String],
    ) -> StorageResult<HashMap<String, ConversationRow>> {
        let conn = self.conn.lock();
        read::get_conversations_batch(&conn, conversation_ids)
    }

    pub fn conversation_exists(&self, conversation_id: &str) -> StorageResult<bool> {
        let conn = self.conn.lock();
        read::conversation_exists(&conn, conversation_id)
    }

    pub fn get_all_conversation_ids(
        &self,
        project_id: Option<&str>,
    ) -> StorageResult<Vec<String>> {
        let conn = self.conn.lock();
        read::get_all_conversation_ids(&conn, project_id)
    }

    /// Returns `(file_hash, file_path, file_size, mtime_ns)` keyed by conversation_id.
    pub fn get_conversation_hashes(
        &self,
    ) -> StorageResult<HashMap<String, (String, String, i64, i64)>> {
        let conn = self.conn.lock();
        read::get_conversation_hashes(&conn)
    }

    pub fn get_indexed_file_paths(&self) -> StorageResult<HashSet<String>> {
        let conn = self.conn.lock();
        read::get_indexed_file_paths(&conn)
    }

    pub fn get_max_message_sequence(&self, conversation_id: &str) -> StorageResult<i32> {
        let conn = self.conn.lock();
        read::get_max_message_sequence(&conn, conversation_id)
    }

    pub fn get_conversation_messages(
        &self,
        conversation_id: &str,
    ) -> StorageResult<Vec<MessageRow>> {
        let conn = self.conn.lock();
        read::get_conversation_messages(&conn, conversation_id)
    }

    pub fn backfill_stat_columns(
        &self,
        rows: &[(i64, i64, String)],
    ) -> StorageResult<()> {
        let conn = self.conn.lock();
        write::backfill_stat_columns(&conn, rows)
    }

    // -----------------------------------------------------------------------
    // Exchange CRUD
    // -----------------------------------------------------------------------

    pub fn store_exchange(
        &self,
        ex: &ExchangeInput,
        created_at: DateTime<Utc>,
        skip_existing_check: bool,
    ) -> StorageResult<String> {
        let conn = self.conn.lock();
        write::store_exchange(&conn, ex, created_at, skip_existing_check)
    }

    pub fn store_exchanges_batch(
        &self,
        exchanges: &[ExchangeInput],
        created_at: DateTime<Utc>,
    ) -> StorageResult<()> {
        let conn = self.conn.lock();
        write::store_exchanges_batch(&conn, exchanges, created_at)
    }

    pub fn get_exchange(&self, exchange_id: &str) -> StorageResult<Option<ExchangeRow>> {
        let conn = self.conn.lock();
        read::get_exchange(&conn, exchange_id)
    }

    pub fn get_exchange_by_ply(
        &self,
        conversation_id: &str,
        ply_start: i32,
        ply_end: i32,
    ) -> StorageResult<Option<ExchangeRow>> {
        let conn = self.conn.lock();
        read::get_exchange_by_ply(&conn, conversation_id, ply_start, ply_end)
    }

    pub fn get_conversation_exchanges(
        &self,
        conversation_id: &str,
    ) -> StorageResult<Vec<ExchangeRow>> {
        let conn = self.conn.lock();
        read::get_conversation_exchanges(&conn, conversation_id)
    }

    pub fn get_existing_exchange_keys(
        &self,
        conversation_ids: Option<&[String]>,
    ) -> StorageResult<HashSet<(String, i32, i32)>> {
        let conn = self.conn.lock();
        read::get_existing_exchange_keys(&conn, conversation_ids)
    }

    pub fn get_exchange_ids_in_set(
        &self,
        exchange_ids: &HashSet<String>,
    ) -> StorageResult<HashSet<String>> {
        let conn = self.conn.lock();
        read::get_exchange_ids_in_set(&conn, exchange_ids)
    }

    // -----------------------------------------------------------------------
    // Verbatim embeddings
    // -----------------------------------------------------------------------

    pub fn store_verbatim_embedding(
        &self,
        exchange_id: &str,
        embedding: &[f32],
    ) -> StorageResult<()> {
        let conn = self.conn.lock();
        write::store_verbatim_embedding(&conn, exchange_id, embedding)
    }

    pub fn store_verbatim_embeddings_batch(
        &self,
        embeddings: &[(String, Vec<f32>)],
    ) -> StorageResult<()> {
        let conn = self.conn.lock();
        write::store_verbatim_embeddings_batch(&conn, embeddings)
    }

    // -----------------------------------------------------------------------
    // Palace objects
    // -----------------------------------------------------------------------

    pub fn store_palace_object(
        &self,
        obj: &DistilledObject,
        embedding: Option<&[f32]>,
    ) -> StorageResult<()> {
        let conn = self.conn.lock();
        write::store_palace_object(&conn, obj, embedding)
    }

    pub fn get_palace_object(
        &self,
        object_id: &str,
    ) -> StorageResult<Option<DistilledObject>> {
        let conn = self.conn.lock();
        read::get_palace_object(&conn, object_id)
    }

    pub fn get_existing_palace_keys(
        &self,
    ) -> StorageResult<HashSet<(String, i32, i32)>> {
        let conn = self.conn.lock();
        read::get_existing_palace_keys(&conn)
    }

    // -----------------------------------------------------------------------
    // Rooms
    // -----------------------------------------------------------------------

    pub fn store_room(&self, room: &Room) -> StorageResult<()> {
        let conn = self.conn.lock();
        write::store_room(&conn, room)
    }

    pub fn store_room_object(&self, junction: &RoomObject) -> StorageResult<()> {
        let conn = self.conn.lock();
        write::store_room_object(&conn, junction)
    }

    pub fn get_rooms_for_object(&self, object_id: &str) -> StorageResult<Vec<Room>> {
        let conn = self.conn.lock();
        read::get_rooms_for_object(&conn, object_id)
    }

    // -----------------------------------------------------------------------
    // Facet embeddings
    // -----------------------------------------------------------------------

    pub fn store_facet_embedding(
        &self,
        facet_id: &str,
        facet_type: &str,
        facet_text: &str,
        project_ids: &[String],
        embedding: &[f32],
        last_seen: Option<DateTime<Utc>>,
    ) -> StorageResult<()> {
        let conn = self.conn.lock();
        write::store_facet_embedding(
            &conn,
            facet_id,
            facet_type,
            facet_text,
            project_ids,
            embedding,
            last_seen,
        )
    }

    pub fn store_hierarchical_facet(
        &self,
        facet_id: &str,
        facet_type: &str,
        facet_level: &str,
        facet_text: &str,
        weight: f32,
        weighted_count: f32,
        project_ids: &[String],
        embedding: &[f32],
        last_seen: Option<DateTime<Utc>>,
    ) -> StorageResult<()> {
        let conn = self.conn.lock();
        write::store_hierarchical_facet(
            &conn,
            facet_id,
            facet_type,
            facet_level,
            facet_text,
            weight,
            weighted_count,
            project_ids,
            embedding,
            last_seen,
        )
    }

    pub fn get_facet_project_ids(
        &self,
        facet_id: &str,
    ) -> StorageResult<Option<Vec<String>>> {
        let conn = self.conn.lock();
        read::get_facet_project_ids(&conn, facet_id)
    }

    pub fn get_facet_project_ids_batch(
        &self,
        facet_ids: &[String],
    ) -> StorageResult<HashMap<String, Vec<String>>> {
        let conn = self.conn.lock();
        read::get_facet_project_ids_batch(&conn, facet_ids)
    }

    pub fn search_facet_embeddings(
        &self,
        query_embedding: &[f32],
        limit: usize,
        facet_types: Option<&[String]>,
        max_project_count: Option<i32>,
    ) -> StorageResult<Vec<FacetSearchResult>> {
        let conn = self.conn.lock();
        read::search_facet_embeddings(
            &conn,
            query_embedding,
            limit,
            facet_types,
            max_project_count,
            self.vss_available,
        )
    }

    pub fn search_hierarchical_facets(
        &self,
        query_embedding: &[f32],
        limit: usize,
        facet_types: Option<&[String]>,
        max_project_count: Option<i32>,
        min_weighted_count: Option<f32>,
    ) -> StorageResult<Vec<HierarchicalFacetSearchResult>> {
        let conn = self.conn.lock();
        read::search_hierarchical_facets(
            &conn,
            query_embedding,
            limit,
            facet_types,
            max_project_count,
            min_weighted_count,
            self.vss_available,
        )
    }

    // -----------------------------------------------------------------------
    // Search
    // -----------------------------------------------------------------------

    /// BM25 search over exchange text. Falls back to LIKE if FTS is unavailable.
    pub fn search_verbatim_bm25(
        &self,
        query: &str,
        limit: usize,
        project_ids: Option<&[String]>,
    ) -> StorageResult<Vec<VerbatimSearchResult>> {
        let conn = self.conn.lock();
        if self.fts_available {
            match read::search_verbatim_bm25(&conn, query, limit, project_ids) {
                Ok(results) => return Ok(results),
                Err(e) => {
                    log::warn!("BM25 search failed, falling back to LIKE: {e}");
                }
            }
        }
        read::search_verbatim_like(&conn, query, limit, project_ids)
    }

    /// BM25 search over palace distilled_text. Falls back to LIKE if FTS unavailable.
    pub fn search_palace_bm25(
        &self,
        query: &str,
        limit: usize,
        project_ids: Option<&[String]>,
    ) -> StorageResult<Vec<PalaceSearchResult>> {
        let conn = self.conn.lock();
        if self.fts_available {
            match read::search_palace_bm25(&conn, query, limit, project_ids) {
                Ok(results) => return Ok(results),
                Err(e) => {
                    log::warn!("Palace BM25 search failed, falling back to LIKE: {e}");
                }
            }
        }
        read::search_palace_like(&conn, query, limit, project_ids)
    }

    /// Semantic search over verbatim embeddings.
    pub fn search_verbatim_semantic(
        &self,
        query_embedding: &[f32],
        limit: usize,
        project_ids: Option<&[String]>,
    ) -> StorageResult<Vec<VerbatimSearchResult>> {
        let conn = self.conn.lock();
        read::search_verbatim_semantic(
            &conn,
            query_embedding,
            limit,
            project_ids,
            self.vss_available,
        )
    }

    /// Semantic search over palace objects.
    pub fn search_palace_semantic(
        &self,
        query_embedding: &[f32],
        limit: usize,
        project_ids: Option<&[String]>,
    ) -> StorageResult<Vec<PalaceSearchResult>> {
        let conn = self.conn.lock();
        read::search_palace_semantic(
            &conn,
            query_embedding,
            limit,
            project_ids,
            self.vss_available,
        )
    }

    // -----------------------------------------------------------------------
    // Source file state
    // -----------------------------------------------------------------------

    pub fn mark_source_file_invalid(
        &self,
        file_path: &str,
        conversation_id: &str,
        file_size: i64,
        mtime_ns: i64,
        error_message: &str,
    ) -> StorageResult<()> {
        let conn = self.conn.lock();
        write::mark_source_file_invalid(
            &conn,
            file_path,
            conversation_id,
            file_size,
            mtime_ns,
            error_message,
        )
    }

    pub fn clear_source_file_state(&self, file_path: &str) -> StorageResult<()> {
        let conn = self.conn.lock();
        write::clear_source_file_state(&conn, file_path)
    }

    pub fn get_source_file_state(
        &self,
        file_paths: Option<&[String]>,
    ) -> StorageResult<HashMap<String, serde_json::Value>> {
        let conn = self.conn.lock();
        read::get_source_file_state(&conn, file_paths)
    }

    // -----------------------------------------------------------------------
    // Deletion
    // -----------------------------------------------------------------------

    pub fn delete_exchange_data(&self, conversation_id: &str) -> StorageResult<()> {
        let conn = self.conn.lock();
        write::delete_exchange_data(&conn, conversation_id)
    }

    pub fn delete_exchange_data_from_ply(
        &self,
        conversation_id: &str,
        ply_start: i32,
    ) -> StorageResult<()> {
        let conn = self.conn.lock();
        write::delete_exchange_data_from_ply(&conn, conversation_id, ply_start)
    }

    pub fn delete_conversation(
        &self,
        conversation_id: &str,
    ) -> StorageResult<HashMap<String, usize>> {
        let conn = self.conn.lock();
        write::delete_conversation(&conn, conversation_id)
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    pub fn get_stats(&self) -> StorageResult<StorageStats> {
        let conn = self.conn.lock();
        read::get_stats(&conn, self.vss_available, self.fts_available)
    }

    // -----------------------------------------------------------------------
    // Batch distillation helper
    // -----------------------------------------------------------------------

    /// Store distillation results (palace objects + rooms + junctions + facets)
    /// in a single transaction.
    pub fn store_distillation_results(
        &self,
        objects: &[(DistilledObject, Option<Vec<f32>>)],
        rooms: &[Room],
        junctions: &[RoomObject],
        facets: &[FacetInput],
    ) -> StorageResult<()> {
        let conn = self.conn.lock();
        write::begin_transaction(&conn)?;
        let result = (|| -> StorageResult<()> {
            for (obj, embedding) in objects {
                write::store_palace_object(
                    &conn,
                    obj,
                    embedding.as_deref(),
                )?;
            }
            for room in rooms {
                write::store_room(&conn, room)?;
            }
            for junction in junctions {
                write::store_room_object(&conn, junction)?;
            }
            for f in facets {
                write::store_facet_embedding(
                    &conn,
                    &f.facet_id,
                    &f.facet_type,
                    &f.facet_text,
                    &f.project_ids,
                    &f.embedding,
                    f.last_seen,
                )?;
            }
            Ok(())
        })();

        match &result {
            Ok(_) => write::commit(&conn)?,
            Err(_) => write::rollback(&conn),
        }
        result
    }
}

// ---------------------------------------------------------------------------
// FacetInput helper struct
// ---------------------------------------------------------------------------

/// Input for a batch facet embedding store operation.
pub struct FacetInput {
    pub facet_id: String,
    pub facet_type: String,
    pub facet_text: String,
    pub project_ids: Vec<String>,
    pub embedding: Vec<f32>,
    pub last_seen: Option<DateTime<Utc>>,
}

// ---------------------------------------------------------------------------
// Extension loading
// ---------------------------------------------------------------------------

fn load_extensions(conn: &Connection) -> (bool, bool) {
    let vss = try_load_extension(conn, "vss");
    let fts = try_load_extension(conn, "fts");
    (vss, fts)
}

fn try_load_extension(conn: &Connection, name: &str) -> bool {
    if conn.execute_batch(&format!("LOAD {name}")).is_ok() {
        log::info!("DuckDB {name} extension loaded");
        return true;
    }
    // Try install + load (only needed on first use)
    if conn
        .execute_batch(&format!("INSTALL {name}; LOAD {name}"))
        .is_ok()
    {
        log::info!("DuckDB {name} extension installed and loaded");
        return true;
    }
    log::warn!("DuckDB {name} extension not available");
    false
}

// ---------------------------------------------------------------------------
// hex encoding helper (avoid adding a dep just for this)
// ---------------------------------------------------------------------------

mod hex {
    pub fn encode(bytes: &[u8]) -> String {
        bytes.iter().fold(String::new(), |mut s, b| {
            s.push_str(&format!("{b:02x}"));
            s
        })
    }
}
