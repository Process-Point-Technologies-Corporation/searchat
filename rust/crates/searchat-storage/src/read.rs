/// SELECT queries: search, get conversation, statistics.
use std::collections::{HashMap, HashSet};

use chrono::{DateTime, Utc};
use duckdb::Connection;
use serde_json;

use searchat_models::{DistilledObject, FileTouched, Room};

use crate::error::StorageResult;
use crate::schema::EMBEDDING_DIM;
use crate::write::embedding_to_value;

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ConversationRow {
    pub conversation_id: String,
    pub project_id: String,
    pub file_path: String,
    pub title: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub message_count: i32,
    pub full_text: String,
    pub file_hash: String,
    pub indexed_at: DateTime<Utc>,
    pub file_size: i64,
    pub mtime_ns: i64,
}

#[derive(Debug, Clone)]
pub struct MessageRow {
    pub sequence: i32,
    pub role: String,
    pub content: String,
    pub timestamp: Option<DateTime<Utc>>,
    pub has_code: bool,
}

#[derive(Debug, Clone)]
pub struct ExchangeRow {
    pub exchange_id: String,
    pub conversation_id: String,
    pub project_id: Option<String>,
    pub ply_start: i32,
    pub ply_end: i32,
    pub exchange_text: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct VerbatimSearchResult {
    pub exchange_id: String,
    pub conversation_id: String,
    pub project_id: Option<String>,
    pub ply_start: i32,
    pub ply_end: i32,
    pub exchange_text: String,
    pub title: String,
    pub file_path: String,
    pub message_count: i32,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub score: f64,
    /// Cosine distance (semantic searches only; 0.0 for BM25)
    pub distance: f64,
}

#[derive(Debug, Clone)]
pub struct PalaceSearchResult {
    pub object_id: String,
    pub exchange_id: Option<String>,
    pub conversation_id: String,
    pub project_id: String,
    pub ply_start: i32,
    pub ply_end: i32,
    pub files_touched: Vec<serde_json::Value>,
    pub exchange_core: String,
    pub specific_context: String,
    pub distilled_text: String,
    pub created_at: DateTime<Utc>,
    pub exchange_at: DateTime<Utc>,
    pub score: f64,
    pub distance: f64,
    pub exchange_text: Option<String>,
}

#[derive(Debug, Clone)]
pub struct FacetSearchResult {
    pub facet_id: String,
    pub facet_type: String,
    pub facet_text: String,
    pub project_ids: Vec<String>,
    pub project_count: i32,
    pub distance: f64,
    pub score: f64,
    pub last_seen: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone)]
pub struct HierarchicalFacetSearchResult {
    pub facet_id: String,
    pub facet_type: String,
    pub facet_level: String,
    pub facet_text: String,
    pub weight: f32,
    pub weighted_count: f32,
    pub project_ids: Vec<String>,
    pub project_count: i32,
    pub distance: f64,
    pub score: f64,
    pub last_seen: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone)]
pub struct StorageStats {
    pub conversations: i64,
    pub messages: i64,
    pub exchanges: i64,
    pub verbatim_embeddings: i64,
    pub palace_objects: i64,
    pub rooms: i64,
    pub facet_embeddings: i64,
    pub hierarchical_facets: i64,
    pub vss_available: bool,
    pub fts_available: bool,
}

// ---------------------------------------------------------------------------
// Helper: parse files_touched JSON
// ---------------------------------------------------------------------------

fn parse_files_touched(raw: &Option<String>) -> Vec<serde_json::Value> {
    match raw {
        Some(s) => serde_json::from_str(s).unwrap_or_default(),
        None => vec![],
    }
}

fn parse_project_ids(raw: &str) -> Vec<String> {
    serde_json::from_str(raw).unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Conversation reads
// ---------------------------------------------------------------------------

pub fn get_conversation(
    conn: &Connection,
    conversation_id: &str,
) -> StorageResult<Option<ConversationRow>> {
    let mut stmt = conn.prepare(
        "SELECT conversation_id, project_id, file_path, title,
                created_at, updated_at, message_count, full_text,
                file_hash, indexed_at, file_size, mtime_ns
         FROM conversations
         WHERE conversation_id = ?",
    )?;

    let mut rows = stmt.query_map(duckdb::params![conversation_id], |row| {
        Ok(ConversationRow {
            conversation_id: row.get(0)?,
            project_id: row.get(1)?,
            file_path: row.get(2)?,
            title: row.get(3)?,
            created_at: row.get(4)?,
            updated_at: row.get(5)?,
            message_count: row.get(6)?,
            full_text: row.get(7)?,
            file_hash: row.get(8)?,
            indexed_at: row.get(9)?,
            file_size: row.get::<_, Option<i64>>(10)?.unwrap_or(0),
            mtime_ns: row.get::<_, Option<i64>>(11)?.unwrap_or(0),
        })
    })?;

    Ok(rows.next().and_then(|r| r.ok()))
}

pub fn get_conversations_batch(
    conn: &Connection,
    conversation_ids: &[String],
) -> StorageResult<HashMap<String, ConversationRow>> {
    if conversation_ids.is_empty() {
        return Ok(HashMap::new());
    }

    let placeholders = conversation_ids
        .iter()
        .map(|_| "?")
        .collect::<Vec<_>>()
        .join(", ");
    let sql = format!(
        "SELECT conversation_id, project_id, file_path, title,
                created_at, updated_at, message_count, full_text,
                file_hash, indexed_at, file_size, mtime_ns
         FROM conversations
         WHERE conversation_id IN ({placeholders})"
    );

    let mut stmt = conn.prepare(&sql)?;
    let params: Vec<&dyn duckdb::ToSql> = conversation_ids
        .iter()
        .map(|s| s as &dyn duckdb::ToSql)
        .collect();

    let rows = stmt.query_map(params.as_slice(), |row| {
        Ok(ConversationRow {
            conversation_id: row.get(0)?,
            project_id: row.get(1)?,
            file_path: row.get(2)?,
            title: row.get(3)?,
            created_at: row.get(4)?,
            updated_at: row.get(5)?,
            message_count: row.get(6)?,
            full_text: row.get(7)?,
            file_hash: row.get(8)?,
            indexed_at: row.get(9)?,
            file_size: row.get::<_, Option<i64>>(10)?.unwrap_or(0),
            mtime_ns: row.get::<_, Option<i64>>(11)?.unwrap_or(0),
        })
    })?;

    let mut map = HashMap::new();
    for row in rows.flatten() {
        map.insert(row.conversation_id.clone(), row);
    }
    Ok(map)
}

pub fn conversation_exists(conn: &Connection, conversation_id: &str) -> StorageResult<bool> {
    let result: bool = conn
        .query_row(
            "SELECT 1 FROM conversations WHERE conversation_id = ? LIMIT 1",
            duckdb::params![conversation_id],
            |_| Ok(true),
        )
        .unwrap_or(false);
    Ok(result)
}

pub fn get_all_conversation_ids(
    conn: &Connection,
    project_id: Option<&str>,
) -> StorageResult<Vec<String>> {
    match project_id {
        Some(pid) => {
            let mut stmt = conn.prepare(
                "SELECT conversation_id FROM conversations WHERE project_id = ?",
            )?;
            let rows = stmt.query_map(duckdb::params![pid], |row| row.get(0))?;
            Ok(rows.flatten().collect())
        }
        None => {
            let mut stmt =
                conn.prepare("SELECT conversation_id FROM conversations")?;
            let rows = stmt.query_map([], |row| row.get(0))?;
            Ok(rows.flatten().collect())
        }
    }
}

/// Returns `HashMap<conversation_id, (file_hash, file_path, file_size, mtime_ns)>`.
pub fn get_conversation_hashes(
    conn: &Connection,
) -> StorageResult<HashMap<String, (String, String, i64, i64)>> {
    let mut stmt = conn.prepare(
        "SELECT conversation_id, file_hash, file_path, file_size, mtime_ns FROM conversations",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, Option<i64>>(3)?.unwrap_or(0),
            row.get::<_, Option<i64>>(4)?.unwrap_or(0),
        ))
    })?;

    let mut map = HashMap::new();
    for row in rows.flatten() {
        map.insert(row.0, (row.1, row.2, row.3, row.4));
    }
    Ok(map)
}

pub fn get_indexed_file_paths(conn: &Connection) -> StorageResult<HashSet<String>> {
    let mut stmt = conn.prepare("SELECT file_path FROM conversations")?;
    let rows = stmt.query_map([], |row| row.get(0))?;
    Ok(rows.flatten().collect())
}

// ---------------------------------------------------------------------------
// Message reads
// ---------------------------------------------------------------------------

pub fn get_conversation_messages(
    conn: &Connection,
    conversation_id: &str,
) -> StorageResult<Vec<MessageRow>> {
    let mut stmt = conn.prepare(
        "SELECT sequence, role, content, timestamp, has_code
         FROM messages
         WHERE conversation_id = ?
         ORDER BY sequence ASC",
    )?;
    let rows = stmt.query_map(duckdb::params![conversation_id], |row| {
        Ok(MessageRow {
            sequence: row.get(0)?,
            role: row.get(1)?,
            content: row.get(2)?,
            timestamp: row.get(3)?,
            has_code: row.get(4)?,
        })
    })?;
    Ok(rows.flatten().collect())
}

pub fn get_max_message_sequence(conn: &Connection, conversation_id: &str) -> StorageResult<i32> {
    let val: i32 = conn.query_row(
        "SELECT COALESCE(MAX(sequence), -1) FROM messages WHERE conversation_id = ?",
        duckdb::params![conversation_id],
        |row| row.get(0),
    )?;
    Ok(val)
}

// ---------------------------------------------------------------------------
// Exchange reads
// ---------------------------------------------------------------------------

pub fn get_exchange(conn: &Connection, exchange_id: &str) -> StorageResult<Option<ExchangeRow>> {
    let mut stmt = conn.prepare(
        "SELECT exchange_id, conversation_id, project_id, ply_start, ply_end,
                exchange_text, created_at
         FROM exchanges WHERE exchange_id = ?",
    )?;
    let mut rows = stmt.query_map(duckdb::params![exchange_id], |row| {
        Ok(ExchangeRow {
            exchange_id: row.get(0)?,
            conversation_id: row.get(1)?,
            project_id: row.get(2)?,
            ply_start: row.get(3)?,
            ply_end: row.get(4)?,
            exchange_text: row.get(5)?,
            created_at: row.get(6)?,
        })
    })?;
    Ok(rows.next().and_then(|r| r.ok()))
}

pub fn get_exchange_by_ply(
    conn: &Connection,
    conversation_id: &str,
    ply_start: i32,
    ply_end: i32,
) -> StorageResult<Option<ExchangeRow>> {
    let mut stmt = conn.prepare(
        "SELECT exchange_id, conversation_id, project_id, ply_start, ply_end,
                exchange_text, created_at
         FROM exchanges
         WHERE conversation_id = ? AND ply_start = ? AND ply_end = ?",
    )?;
    let mut rows = stmt.query_map(
        duckdb::params![conversation_id, ply_start, ply_end],
        |row| {
            Ok(ExchangeRow {
                exchange_id: row.get(0)?,
                conversation_id: row.get(1)?,
                project_id: row.get(2)?,
                ply_start: row.get(3)?,
                ply_end: row.get(4)?,
                exchange_text: row.get(5)?,
                created_at: row.get(6)?,
            })
        },
    )?;
    Ok(rows.next().and_then(|r| r.ok()))
}

pub fn get_conversation_exchanges(
    conn: &Connection,
    conversation_id: &str,
) -> StorageResult<Vec<ExchangeRow>> {
    let mut stmt = conn.prepare(
        "SELECT exchange_id, conversation_id, project_id, ply_start, ply_end,
                exchange_text, created_at
         FROM exchanges
         WHERE conversation_id = ?
         ORDER BY ply_start ASC, ply_end ASC",
    )?;
    let rows = stmt.query_map(duckdb::params![conversation_id], |row| {
        Ok(ExchangeRow {
            exchange_id: row.get(0)?,
            conversation_id: row.get(1)?,
            project_id: row.get(2)?,
            ply_start: row.get(3)?,
            ply_end: row.get(4)?,
            exchange_text: row.get(5)?,
            created_at: row.get(6)?,
        })
    })?;
    Ok(rows.flatten().collect())
}

/// Returns the set of (conversation_id, ply_start, ply_end) tuples already stored.
/// If `conversation_ids` is `Some([])`, returns an empty set immediately.
pub fn get_existing_exchange_keys(
    conn: &Connection,
    conversation_ids: Option<&[String]>,
) -> StorageResult<HashSet<(String, i32, i32)>> {
    match conversation_ids {
        Some(ids) if ids.is_empty() => return Ok(HashSet::new()),
        Some(ids) => {
            let placeholders = ids.iter().map(|_| "?").collect::<Vec<_>>().join(", ");
            let sql = format!(
                "SELECT conversation_id, ply_start, ply_end FROM exchanges \
                 WHERE conversation_id IN ({placeholders})"
            );
            let mut stmt = conn.prepare(&sql)?;
            let params: Vec<&dyn duckdb::ToSql> =
                ids.iter().map(|s| s as &dyn duckdb::ToSql).collect();
            let rows = stmt.query_map(params.as_slice(), |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?))
            })?;
            Ok(rows.flatten().collect())
        }
        None => {
            let mut stmt = conn.prepare(
                "SELECT conversation_id, ply_start, ply_end FROM exchanges",
            )?;
            let rows = stmt.query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?;
            Ok(rows.flatten().collect())
        }
    }
}

/// Returns the subset of `exchange_ids` that exist in the exchanges table.
pub fn get_exchange_ids_in_set(
    conn: &Connection,
    exchange_ids: &HashSet<String>,
) -> StorageResult<HashSet<String>> {
    if exchange_ids.is_empty() {
        return Ok(HashSet::new());
    }
    let ids: Vec<&String> = exchange_ids.iter().collect();
    let placeholders = ids.iter().map(|_| "?").collect::<Vec<_>>().join(", ");
    let sql = format!(
        "SELECT exchange_id FROM exchanges WHERE exchange_id IN ({placeholders})"
    );
    let mut stmt = conn.prepare(&sql)?;
    let params: Vec<&dyn duckdb::ToSql> =
        ids.iter().map(|s| *s as &dyn duckdb::ToSql).collect();
    let rows = stmt.query_map(params.as_slice(), |row| row.get(0))?;
    Ok(rows.flatten().collect())
}

// ---------------------------------------------------------------------------
// BM25 search (FTS)
// ---------------------------------------------------------------------------

/// BM25 keyword search over exchange text.
pub fn search_verbatim_bm25(
    conn: &Connection,
    query: &str,
    limit: usize,
    project_ids: Option<&[String]>,
) -> StorageResult<Vec<VerbatimSearchResult>> {
    let project_filter = project_filter_sql("e.project_id", project_ids);

    let sql = format!(
        "SELECT
             e.exchange_id, e.conversation_id, e.project_id,
             e.ply_start, e.ply_end, e.exchange_text,
             c.title, c.file_path, c.message_count, c.created_at, c.updated_at,
             fts_main_exchanges.match_bm25(e.exchange_id, ?) AS score
         FROM exchanges e
         JOIN conversations c ON e.conversation_id = c.conversation_id
         WHERE score IS NOT NULL {project_filter}
         ORDER BY score DESC
         LIMIT {limit}"
    );

    let mut stmt = conn.prepare(&sql)?;
    let mut params: Vec<Box<dyn duckdb::ToSql>> = vec![Box::new(query.to_owned())];
    if let Some(pids) = project_ids {
        for pid in pids {
            params.push(Box::new(pid.clone()));
        }
    }
    let param_refs: Vec<&dyn duckdb::ToSql> = params.iter().map(|p| p.as_ref()).collect();

    let rows = stmt.query_map(param_refs.as_slice(), |row| {
        Ok(VerbatimSearchResult {
            exchange_id: row.get(0)?,
            conversation_id: row.get(1)?,
            project_id: row.get(2)?,
            ply_start: row.get(3)?,
            ply_end: row.get(4)?,
            exchange_text: row.get(5)?,
            title: row.get(6)?,
            file_path: row.get(7)?,
            message_count: row.get(8)?,
            created_at: row.get(9)?,
            updated_at: row.get(10)?,
            score: row.get::<_, f64>(11)?,
            distance: 0.0,
        })
    })?;

    Ok(rows.flatten().collect())
}

/// BM25 keyword search over palace distilled_text.
pub fn search_palace_bm25(
    conn: &Connection,
    query: &str,
    limit: usize,
    project_ids: Option<&[String]>,
) -> StorageResult<Vec<PalaceSearchResult>> {
    let project_filter = project_filter_sql_literal("po.project_id", project_ids);

    let sql = format!(
        "SELECT
             po.object_id, po.exchange_id, po.conversation_id, po.project_id,
             po.ply_start, po.ply_end, po.files_touched, po.exchange_core,
             po.specific_context, po.distilled_text, po.created_at, po.exchange_at,
             fts_main_palace_objects.match_bm25(po.object_id, ?) AS score,
             e.exchange_text
         FROM palace_objects po
         LEFT JOIN exchanges e ON po.exchange_id = e.exchange_id
         WHERE score IS NOT NULL {project_filter}
         ORDER BY score DESC
         LIMIT {limit}"
    );

    let mut stmt = conn.prepare(&sql)?;

    let rows = stmt.query_map(duckdb::params![query], |row| {
        let ft_raw: Option<String> = row.get(6)?;
        Ok(PalaceSearchResult {
            object_id: row.get(0)?,
            exchange_id: row.get(1)?,
            conversation_id: row.get(2)?,
            project_id: row.get(3)?,
            ply_start: row.get(4)?,
            ply_end: row.get(5)?,
            files_touched: parse_files_touched(&ft_raw),
            exchange_core: row.get(7)?,
            specific_context: row.get(8)?,
            distilled_text: row.get(9)?,
            created_at: row.get(10)?,
            exchange_at: row.get(11)?,
            score: row.get::<_, f64>(12)?,
            distance: 0.0,
            exchange_text: row.get(13)?,
        })
    })?;

    Ok(rows.flatten().collect())
}

// ---------------------------------------------------------------------------
// Semantic search (vector)
// ---------------------------------------------------------------------------

/// Semantic search over verbatim embeddings using cosine distance.
pub fn search_verbatim_semantic(
    conn: &Connection,
    query_embedding: &[f32],
    limit: usize,
    project_ids: Option<&[String]>,
    use_hnsw: bool,
) -> StorageResult<Vec<VerbatimSearchResult>> {
    debug_assert_eq!(query_embedding.len(), EMBEDDING_DIM);
    let distance_fn = if use_hnsw {
        "array_cosine_distance"
    } else {
        "list_cosine_distance"
    };
    let project_filter = project_filter_sql("e.project_id", project_ids);

    let sql = format!(
        "SELECT
             e.exchange_id, e.conversation_id, e.project_id,
             e.ply_start, e.ply_end, e.exchange_text,
             c.title, c.file_path, c.message_count, c.created_at, c.updated_at,
             {distance_fn}(ve.embedding, ?::FLOAT[{EMBEDDING_DIM}]) AS distance
         FROM verbatim_embeddings ve
         JOIN exchanges e ON ve.exchange_id = e.exchange_id
         JOIN conversations c ON e.conversation_id = c.conversation_id
         WHERE 1=1 {project_filter}
         ORDER BY distance ASC
         LIMIT {limit}"
    );

    let emb_value = embedding_to_value(query_embedding);
    let mut params: Vec<Box<dyn duckdb::ToSql>> = vec![Box::new(emb_value)];
    if let Some(pids) = project_ids {
        for pid in pids {
            params.push(Box::new(pid.clone()));
        }
    }
    let param_refs: Vec<&dyn duckdb::ToSql> = params.iter().map(|p| p.as_ref()).collect();

    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(param_refs.as_slice(), |row| {
        let distance: f64 = row.get(11)?;
        Ok(VerbatimSearchResult {
            exchange_id: row.get(0)?,
            conversation_id: row.get(1)?,
            project_id: row.get(2)?,
            ply_start: row.get(3)?,
            ply_end: row.get(4)?,
            exchange_text: row.get(5)?,
            title: row.get(6)?,
            file_path: row.get(7)?,
            message_count: row.get(8)?,
            created_at: row.get(9)?,
            updated_at: row.get(10)?,
            distance,
            score: 1.0 / (1.0 + distance),
        })
    })?;

    Ok(rows.flatten().collect())
}

/// Semantic search over palace objects using cosine distance.
pub fn search_palace_semantic(
    conn: &Connection,
    query_embedding: &[f32],
    limit: usize,
    project_ids: Option<&[String]>,
    use_hnsw: bool,
) -> StorageResult<Vec<PalaceSearchResult>> {
    debug_assert_eq!(query_embedding.len(), EMBEDDING_DIM);
    let distance_fn = if use_hnsw {
        "array_cosine_distance"
    } else {
        "list_cosine_distance"
    };
    let project_filter = project_filter_sql("po.project_id", project_ids);

    let sql = format!(
        "SELECT
             po.object_id, po.exchange_id, po.conversation_id, po.project_id,
             po.ply_start, po.ply_end, po.files_touched, po.exchange_core,
             po.specific_context, po.distilled_text, po.created_at, po.exchange_at,
             {distance_fn}(po.embedding, ?::FLOAT[{EMBEDDING_DIM}]) AS distance,
             e.exchange_text
         FROM palace_objects po
         LEFT JOIN exchanges e ON po.exchange_id = e.exchange_id
         WHERE po.embedding IS NOT NULL {project_filter}
         ORDER BY distance ASC
         LIMIT {limit}"
    );

    let emb_value = embedding_to_value(query_embedding);
    let mut params: Vec<Box<dyn duckdb::ToSql>> = vec![Box::new(emb_value)];
    if let Some(pids) = project_ids {
        for pid in pids {
            params.push(Box::new(pid.clone()));
        }
    }
    let param_refs: Vec<&dyn duckdb::ToSql> = params.iter().map(|p| p.as_ref()).collect();

    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(param_refs.as_slice(), |row| {
        let ft_raw: Option<String> = row.get(6)?;
        let distance: f64 = row.get(12)?;
        Ok(PalaceSearchResult {
            object_id: row.get(0)?,
            exchange_id: row.get(1)?,
            conversation_id: row.get(2)?,
            project_id: row.get(3)?,
            ply_start: row.get(4)?,
            ply_end: row.get(5)?,
            files_touched: parse_files_touched(&ft_raw),
            exchange_core: row.get(7)?,
            specific_context: row.get(8)?,
            distilled_text: row.get(9)?,
            created_at: row.get(10)?,
            exchange_at: row.get(11)?,
            distance,
            score: 1.0 / (1.0 + distance),
            exchange_text: row.get(13)?,
        })
    })?;

    Ok(rows.flatten().collect())
}

// ---------------------------------------------------------------------------
// LIKE fallback searches
// ---------------------------------------------------------------------------

/// LIKE-based verbatim search (fallback when FTS unavailable).
pub fn search_verbatim_like(
    conn: &Connection,
    query: &str,
    limit: usize,
    project_ids: Option<&[String]>,
) -> StorageResult<Vec<VerbatimSearchResult>> {
    let terms: Vec<String> = query
        .split_whitespace()
        .map(|t| t.to_lowercase())
        .collect();
    if terms.is_empty() {
        return Ok(vec![]);
    }

    let like_conditions = terms
        .iter()
        .map(|t| format!("LOWER(e.exchange_text) LIKE '%{t}%'"))
        .collect::<Vec<_>>()
        .join(" AND ");

    let project_filter = project_filter_sql("e.project_id", project_ids);

    let sql = format!(
        "SELECT
             e.exchange_id, e.conversation_id, e.project_id,
             e.ply_start, e.ply_end, e.exchange_text,
             c.title, c.file_path, c.message_count, c.created_at, c.updated_at,
             1.0 AS score
         FROM exchanges e
         JOIN conversations c ON e.conversation_id = c.conversation_id
         WHERE {like_conditions} {project_filter}
         LIMIT {limit}"
    );

    let mut params: Vec<Box<dyn duckdb::ToSql>> = vec![];
    if let Some(pids) = project_ids {
        for pid in pids {
            params.push(Box::new(pid.clone()));
        }
    }
    let param_refs: Vec<&dyn duckdb::ToSql> = params.iter().map(|p| p.as_ref()).collect();

    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(param_refs.as_slice(), |row| {
        Ok(VerbatimSearchResult {
            exchange_id: row.get(0)?,
            conversation_id: row.get(1)?,
            project_id: row.get(2)?,
            ply_start: row.get(3)?,
            ply_end: row.get(4)?,
            exchange_text: row.get(5)?,
            title: row.get(6)?,
            file_path: row.get(7)?,
            message_count: row.get(8)?,
            created_at: row.get(9)?,
            updated_at: row.get(10)?,
            score: row.get(11)?,
            distance: 0.0,
        })
    })?;

    Ok(rows.flatten().collect())
}

/// LIKE-based palace search (fallback when FTS unavailable).
pub fn search_palace_like(
    conn: &Connection,
    query: &str,
    limit: usize,
    project_ids: Option<&[String]>,
) -> StorageResult<Vec<PalaceSearchResult>> {
    let terms: Vec<String> = query
        .split_whitespace()
        .map(|t| t.to_lowercase())
        .collect();
    if terms.is_empty() {
        return Ok(vec![]);
    }

    let like_conditions = terms
        .iter()
        .map(|t| format!("LOWER(po.distilled_text) LIKE '%{t}%'"))
        .collect::<Vec<_>>()
        .join(" AND ");

    let project_filter = project_filter_sql_literal("po.project_id", project_ids);

    let sql = format!(
        "SELECT
             po.object_id, po.exchange_id, po.conversation_id, po.project_id,
             po.ply_start, po.ply_end, po.files_touched, po.exchange_core,
             po.specific_context, po.distilled_text, po.created_at, po.exchange_at,
             1.0 AS score,
             e.exchange_text
         FROM palace_objects po
         LEFT JOIN exchanges e ON po.exchange_id = e.exchange_id
         WHERE {like_conditions} {project_filter}
         LIMIT {limit}"
    );

    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map([], |row| {
        let ft_raw: Option<String> = row.get(6)?;
        Ok(PalaceSearchResult {
            object_id: row.get(0)?,
            exchange_id: row.get(1)?,
            conversation_id: row.get(2)?,
            project_id: row.get(3)?,
            ply_start: row.get(4)?,
            ply_end: row.get(5)?,
            files_touched: parse_files_touched(&ft_raw),
            exchange_core: row.get(7)?,
            specific_context: row.get(8)?,
            distilled_text: row.get(9)?,
            created_at: row.get(10)?,
            exchange_at: row.get(11)?,
            score: row.get(12)?,
            distance: 0.0,
            exchange_text: row.get(13)?,
        })
    })?;

    Ok(rows.flatten().collect())
}

// ---------------------------------------------------------------------------
// Palace object reads
// ---------------------------------------------------------------------------

pub fn get_palace_object(
    conn: &Connection,
    object_id: &str,
) -> StorageResult<Option<DistilledObject>> {
    let mut stmt = conn.prepare(
        "SELECT object_id, project_id, conversation_id, ply_start, ply_end,
                files_touched, exchange_core, specific_context,
                created_at, exchange_at, distilled_text
         FROM palace_objects WHERE object_id = ?",
    )?;

    let mut rows = stmt.query_map(duckdb::params![object_id], |row| {
        let ft_raw: Option<String> = row.get(5)?;
        let files_touched_json = parse_files_touched(&ft_raw);
        let files_touched: Vec<FileTouched> = files_touched_json
            .into_iter()
            .filter_map(|v| {
                let path = v["path"].as_str()?.to_owned();
                let action = v["action"].as_str()?.to_owned();
                Some(FileTouched { path, action })
            })
            .collect();

        Ok(DistilledObject {
            object_id: row.get(0)?,
            project_id: row.get(1)?,
            conversation_id: row.get(2)?,
            ply_start: row.get::<_, i32>(3)? as i64,
            ply_end: row.get::<_, i32>(4)? as i64,
            files_touched,
            exchange_core: row.get(6)?,
            specific_context: row.get(7)?,
            created_at: row.get(8)?,
            exchange_at: row.get(9)?,
            embedding_id: -1,
            distilled_text: row.get(10)?,
            conv_title: None,
        })
    })?;

    Ok(rows.next().and_then(|r| r.ok()))
}

pub fn get_existing_palace_keys(
    conn: &Connection,
) -> StorageResult<HashSet<(String, i32, i32)>> {
    let mut stmt =
        conn.prepare("SELECT conversation_id, ply_start, ply_end FROM palace_objects")?;
    let rows = stmt.query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?;
    Ok(rows.flatten().collect())
}

// ---------------------------------------------------------------------------
// Room reads
// ---------------------------------------------------------------------------

pub fn get_rooms_for_object(conn: &Connection, object_id: &str) -> StorageResult<Vec<Room>> {
    let mut stmt = conn.prepare(
        "SELECT r.room_id, r.room_type, r.room_key, r.room_label,
                r.project_id, r.created_at, r.updated_at, r.object_count
         FROM room_objects ro
         JOIN rooms r ON ro.room_id = r.room_id
         WHERE ro.object_id = ?",
    )?;
    let rows = stmt.query_map(duckdb::params![object_id], |row| {
        Ok(Room {
            room_id: row.get(0)?,
            room_type: row.get(1)?,
            room_key: row.get(2)?,
            room_label: row.get(3)?,
            project_id: row.get(4)?,
            created_at: row.get(5)?,
            updated_at: row.get(6)?,
            object_count: row.get::<_, i32>(7)? as i64,
        })
    })?;
    Ok(rows.flatten().collect())
}

// ---------------------------------------------------------------------------
// Facet embedding reads
// ---------------------------------------------------------------------------

pub fn get_facet_project_ids(
    conn: &Connection,
    facet_id: &str,
) -> StorageResult<Option<Vec<String>>> {
    let result: Option<String> = conn
        .query_row(
            "SELECT project_ids FROM facet_embeddings WHERE facet_id = ?",
            duckdb::params![facet_id],
            |row| row.get(0),
        )
        .ok();
    Ok(result.map(|s| parse_project_ids(&s)))
}

pub fn get_facet_project_ids_batch(
    conn: &Connection,
    facet_ids: &[String],
) -> StorageResult<HashMap<String, Vec<String>>> {
    if facet_ids.is_empty() {
        return Ok(HashMap::new());
    }
    let placeholders = facet_ids.iter().map(|_| "?").collect::<Vec<_>>().join(", ");
    let sql = format!(
        "SELECT facet_id, project_ids FROM facet_embeddings \
         WHERE facet_id IN ({placeholders})"
    );
    let mut stmt = conn.prepare(&sql)?;
    let params: Vec<&dyn duckdb::ToSql> =
        facet_ids.iter().map(|s| s as &dyn duckdb::ToSql).collect();
    let rows = stmt.query_map(params.as_slice(), |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    })?;
    let mut map = HashMap::new();
    for row in rows.flatten() {
        map.insert(row.0, parse_project_ids(&row.1));
    }
    Ok(map)
}

/// Semantic search over flat facet embeddings.
pub fn search_facet_embeddings(
    conn: &Connection,
    query_embedding: &[f32],
    limit: usize,
    facet_types: Option<&[String]>,
    max_project_count: Option<i32>,
    use_hnsw: bool,
) -> StorageResult<Vec<FacetSearchResult>> {
    debug_assert_eq!(query_embedding.len(), EMBEDDING_DIM);
    let distance_fn = if use_hnsw {
        "array_cosine_distance"
    } else {
        "list_cosine_distance"
    };

    let mut where_clauses: Vec<String> = vec![];
    if let Some(types) = facet_types {
        if !types.is_empty() {
            let list = types
                .iter()
                .map(|t| format!("'{t}'"))
                .collect::<Vec<_>>()
                .join(", ");
            where_clauses.push(format!("facet_type IN ({list})"));
        }
    }
    if let Some(max) = max_project_count {
        where_clauses.push(format!("project_count <= {max}"));
    }
    let where_sql = if where_clauses.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", where_clauses.join(" AND "))
    };

    let sql = format!(
        "SELECT facet_id, facet_type, facet_text, project_ids, project_count,
                {distance_fn}(embedding, ?::FLOAT[{EMBEDDING_DIM}]) AS distance, last_seen
         FROM facet_embeddings
         {where_sql}
         ORDER BY distance ASC
         LIMIT {limit}"
    );

    let emb_value = embedding_to_value(query_embedding);
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(duckdb::params![emb_value], |row| {
        let pids_raw: String = row.get(3)?;
        let distance: f64 = row.get(5)?;
        Ok(FacetSearchResult {
            facet_id: row.get(0)?,
            facet_type: row.get(1)?,
            facet_text: row.get(2)?,
            project_ids: parse_project_ids(&pids_raw),
            project_count: row.get(4)?,
            distance,
            score: 1.0 / (1.0 + distance),
            last_seen: row.get(6)?,
        })
    })?;

    Ok(rows.flatten().collect())
}

/// Semantic search over hierarchical facet embeddings.
pub fn search_hierarchical_facets(
    conn: &Connection,
    query_embedding: &[f32],
    limit: usize,
    facet_types: Option<&[String]>,
    max_project_count: Option<i32>,
    min_weighted_count: Option<f32>,
    use_hnsw: bool,
) -> StorageResult<Vec<HierarchicalFacetSearchResult>> {
    debug_assert_eq!(query_embedding.len(), EMBEDDING_DIM);
    let distance_fn = if use_hnsw {
        "array_cosine_distance"
    } else {
        "list_cosine_distance"
    };

    let mut where_clauses: Vec<String> = vec![];
    if let Some(types) = facet_types {
        if !types.is_empty() {
            let list = types
                .iter()
                .map(|t| format!("'{t}'"))
                .collect::<Vec<_>>()
                .join(", ");
            where_clauses.push(format!("facet_type IN ({list})"));
        }
    }
    if let Some(max) = max_project_count {
        where_clauses.push(format!("project_count <= {max}"));
    }
    if let Some(min) = min_weighted_count {
        where_clauses.push(format!("weighted_count >= {min}"));
    }
    let where_sql = if where_clauses.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", where_clauses.join(" AND "))
    };

    let sql = format!(
        "SELECT facet_id, facet_type, facet_level, facet_text,
                weight, weighted_count, project_ids, project_count,
                {distance_fn}(embedding, ?::FLOAT[{EMBEDDING_DIM}]) AS distance, last_seen
         FROM hierarchical_facets
         {where_sql}
         ORDER BY distance ASC
         LIMIT {limit}"
    );

    let emb_value = embedding_to_value(query_embedding);
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(duckdb::params![emb_value], |row| {
        let pids_raw: String = row.get(6)?;
        let distance: f64 = row.get(8)?;
        Ok(HierarchicalFacetSearchResult {
            facet_id: row.get(0)?,
            facet_type: row.get(1)?,
            facet_level: row.get(2)?,
            facet_text: row.get(3)?,
            weight: row.get(4)?,
            weighted_count: row.get(5)?,
            project_ids: parse_project_ids(&pids_raw),
            project_count: row.get(7)?,
            distance,
            score: 1.0 / (1.0 + distance),
            last_seen: row.get(9)?,
        })
    })?;

    Ok(rows.flatten().collect())
}

// ---------------------------------------------------------------------------
// Source file state reads
// ---------------------------------------------------------------------------

pub fn get_source_file_state(
    conn: &Connection,
    file_paths: Option<&[String]>,
) -> StorageResult<HashMap<String, serde_json::Value>> {
    let rows: Vec<(String, Option<String>, String, i64, i64, Option<String>, DateTime<Utc>)> =
        match file_paths {
            Some(paths) if !paths.is_empty() => {
                let placeholders = paths.iter().map(|_| "?").collect::<Vec<_>>().join(", ");
                let sql = format!(
                    "SELECT file_path, conversation_id, status, file_size, mtime_ns,
                            error_message, updated_at
                     FROM source_file_state WHERE file_path IN ({placeholders})"
                );
                let mut stmt = conn.prepare(&sql)?;
                let params: Vec<&dyn duckdb::ToSql> =
                    paths.iter().map(|s| s as &dyn duckdb::ToSql).collect();
                stmt.query_map(params.as_slice(), |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get::<_, Option<i64>>(3)?.unwrap_or(0),
                        row.get::<_, Option<i64>>(4)?.unwrap_or(0),
                        row.get(5)?,
                        row.get(6)?,
                    ))
                })?
                .flatten()
                .collect()
            }
            _ => {
                let mut stmt = conn.prepare(
                    "SELECT file_path, conversation_id, status, file_size, mtime_ns,
                            error_message, updated_at
                     FROM source_file_state",
                )?;
                stmt.query_map([], |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get::<_, Option<i64>>(3)?.unwrap_or(0),
                        row.get::<_, Option<i64>>(4)?.unwrap_or(0),
                        row.get(5)?,
                        row.get(6)?,
                    ))
                })?
                .flatten()
                .collect()
            }
        };

    let mut map = HashMap::new();
    for (fp, conv_id, status, fs, mtime, err, updated) in rows {
        map.insert(
            fp.clone(),
            serde_json::json!({
                "file_path": fp,
                "conversation_id": conv_id,
                "status": status,
                "file_size": fs,
                "mtime_ns": mtime,
                "error_message": err,
                "updated_at": updated.to_rfc3339(),
            }),
        );
    }
    Ok(map)
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

pub fn get_stats(conn: &Connection, vss_available: bool, fts_available: bool) -> StorageResult<StorageStats> {
    let row = conn.query_row(
        "SELECT
            (SELECT COUNT(*) FROM conversations),
            (SELECT COUNT(*) FROM messages),
            (SELECT COUNT(*) FROM exchanges),
            (SELECT COUNT(*) FROM verbatim_embeddings),
            (SELECT COUNT(*) FROM palace_objects),
            (SELECT COUNT(*) FROM rooms),
            (SELECT COUNT(*) FROM facet_embeddings),
            (SELECT COUNT(*) FROM hierarchical_facets)",
        [],
        |row| {
            Ok(StorageStats {
                conversations: row.get(0)?,
                messages: row.get(1)?,
                exchanges: row.get(2)?,
                verbatim_embeddings: row.get(3)?,
                palace_objects: row.get(4)?,
                rooms: row.get(5)?,
                facet_embeddings: row.get(6)?,
                hierarchical_facets: row.get(7)?,
                vss_available,
                fts_available,
            })
        },
    )?;
    Ok(row)
}

// ---------------------------------------------------------------------------
// Internal: SQL fragment helpers
// ---------------------------------------------------------------------------

/// Returns an AND clause fragment for filtering by project IDs using `?` params.
/// Used where the project IDs are passed as bound parameters.
fn project_filter_sql(col: &str, project_ids: Option<&[String]>) -> String {
    match project_ids {
        Some(ids) if !ids.is_empty() => {
            let placeholders = ids.iter().map(|_| "?").collect::<Vec<_>>().join(", ");
            format!("AND {col} IN ({placeholders})")
        }
        _ => String::new(),
    }
}

/// Returns an AND clause fragment with project IDs inlined as string literals.
/// Used in palace BM25/LIKE queries that have a fixed `?` slot for the query term.
fn project_filter_sql_literal(col: &str, project_ids: Option<&[String]>) -> String {
    match project_ids {
        Some(ids) if !ids.is_empty() => {
            let list = ids
                .iter()
                .map(|p| format!("'{p}'"))
                .collect::<Vec<_>>()
                .join(", ");
            format!("AND {col} IN ({list})")
        }
        _ => String::new(),
    }
}
