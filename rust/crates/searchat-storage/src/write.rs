/// INSERT / UPDATE / DELETE operations.
use chrono::{DateTime, Utc};
use duckdb::{Connection, ToSql, types::Value};
use serde_json;

use searchat_models::{ConversationRecord, DistilledObject, Room, RoomObject};

use crate::error::{StorageError, StorageResult};
use crate::schema::EMBEDDING_DIM;

// ---------------------------------------------------------------------------
// Embedding helpers
// ---------------------------------------------------------------------------

/// Convert a `Vec<f32>` into a `duckdb::Value::List` of `Value::Float` items.
/// This is passed to DuckDB bound parameters as `?::FLOAT[]`.
pub(crate) fn embedding_to_value(embedding: &[f32]) -> Value {
    Value::List(embedding.iter().map(|&f| Value::Float(f)).collect())
}

// ---------------------------------------------------------------------------
// Transaction helpers
// ---------------------------------------------------------------------------

pub fn begin_transaction(conn: &Connection) -> StorageResult<()> {
    // Swallow any stale open transaction before starting fresh.
    let _ = conn.execute_batch("ROLLBACK");
    conn.execute_batch("BEGIN TRANSACTION")?;
    Ok(())
}

pub fn commit(conn: &Connection) -> StorageResult<()> {
    conn.execute_batch("COMMIT")?;
    Ok(())
}

pub fn rollback(conn: &Connection) {
    let _ = conn.execute_batch("ROLLBACK");
}

// ---------------------------------------------------------------------------
// Conversation
// ---------------------------------------------------------------------------

/// Upsert a conversation and its messages.
///
/// If `in_transaction` is false, wraps the operation in BEGIN/COMMIT/ROLLBACK.
pub fn store_conversation(
    conn: &Connection,
    record: &ConversationRecord,
    in_transaction: bool,
) -> StorageResult<()> {
    if !in_transaction {
        begin_transaction(conn)?;
    }

    let result = (|| -> StorageResult<()> {
        conn.execute(
            "INSERT INTO conversations (
                conversation_id, project_id, file_path, title,
                created_at, updated_at, message_count, full_text,
                file_hash, indexed_at, file_size, mtime_ns
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (conversation_id) DO UPDATE SET
                project_id    = EXCLUDED.project_id,
                file_path     = EXCLUDED.file_path,
                title         = EXCLUDED.title,
                updated_at    = EXCLUDED.updated_at,
                message_count = EXCLUDED.message_count,
                full_text     = EXCLUDED.full_text,
                file_hash     = EXCLUDED.file_hash,
                indexed_at    = EXCLUDED.indexed_at,
                file_size     = EXCLUDED.file_size,
                mtime_ns      = EXCLUDED.mtime_ns",
            duckdb::params![
                record.conversation_id,
                record.project_id,
                record.file_path,
                record.title,
                record.created_at,
                record.updated_at,
                record.message_count,
                record.full_text,
                record.file_hash,
                record.indexed_at,
                record.file_size,
                record.mtime_ns,
            ],
        )?;

        if !record.messages.is_empty() {
            let mut stmt = conn.prepare(
                "INSERT INTO messages (
                    conversation_id, sequence, role, content, timestamp, has_code
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (conversation_id, sequence) DO UPDATE SET
                    role      = EXCLUDED.role,
                    content   = EXCLUDED.content,
                    timestamp = EXCLUDED.timestamp,
                    has_code  = EXCLUDED.has_code",
            )?;
            for msg in &record.messages {
                stmt.execute(duckdb::params![
                    record.conversation_id,
                    msg.sequence,
                    msg.role,
                    msg.content,
                    msg.timestamp,
                    msg.has_code,
                ])?;
            }
        }

        Ok(())
    })();

    if !in_transaction {
        match &result {
            Ok(_) => commit(conn)?,
            Err(_) => rollback(conn),
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Exchange
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct ExchangeInput {
    pub exchange_id: String,
    pub conversation_id: String,
    pub project_id: String,
    pub ply_start: i32,
    pub ply_end: i32,
    pub exchange_text: String,
}

/// Store a single exchange. Returns the exchange_id actually stored
/// (existing one if a (conversation_id, ply_start, ply_end) record already exists).
pub fn store_exchange(
    conn: &Connection,
    ex: &ExchangeInput,
    created_at: DateTime<Utc>,
    skip_existing_check: bool,
) -> StorageResult<String> {
    if !skip_existing_check {
        let mut stmt = conn.prepare(
            "SELECT exchange_id FROM exchanges \
             WHERE conversation_id = ? AND ply_start = ? AND ply_end = ?",
        )?;
        let existing: Option<String> = stmt
            .query_map(
                duckdb::params![ex.conversation_id, ex.ply_start, ex.ply_end],
                |row| row.get(0),
            )?
            .next()
            .and_then(|r| r.ok());

        if let Some(existing_id) = existing {
            return Ok(existing_id);
        }
    }

    conn.execute(
        "INSERT INTO exchanges (
            exchange_id, conversation_id, project_id,
            ply_start, ply_end, exchange_text, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)",
        duckdb::params![
            ex.exchange_id,
            ex.conversation_id,
            ex.project_id,
            ex.ply_start,
            ex.ply_end,
            ex.exchange_text,
            created_at,
        ],
    )?;

    Ok(ex.exchange_id.clone())
}

/// Store multiple exchanges in one prepared-statement loop.
/// Caller must manage the surrounding transaction.
pub fn store_exchanges_batch(
    conn: &Connection,
    exchanges: &[ExchangeInput],
    created_at: DateTime<Utc>,
) -> StorageResult<()> {
    if exchanges.is_empty() {
        return Ok(());
    }

    let mut stmt = conn.prepare(
        "INSERT INTO exchanges (
            exchange_id, conversation_id, project_id,
            ply_start, ply_end, exchange_text, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (conversation_id, ply_start, ply_end) DO UPDATE SET
            exchange_text = EXCLUDED.exchange_text,
            created_at    = EXCLUDED.created_at",
    )?;

    for ex in exchanges {
        stmt.execute(duckdb::params![
            ex.exchange_id,
            ex.conversation_id,
            ex.project_id,
            ex.ply_start,
            ex.ply_end,
            ex.exchange_text,
            created_at,
        ])?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Verbatim embeddings
// ---------------------------------------------------------------------------

/// Upsert a single verbatim embedding.
pub fn store_verbatim_embedding(
    conn: &Connection,
    exchange_id: &str,
    embedding: &[f32],
) -> StorageResult<()> {
    debug_assert_eq!(
        embedding.len(),
        EMBEDDING_DIM,
        "embedding length must be {EMBEDDING_DIM}"
    );
    let emb_value = embedding_to_value(embedding);
    conn.execute(
        &format!(
            "INSERT INTO verbatim_embeddings (exchange_id, embedding) \
             VALUES (?, ?::FLOAT[{EMBEDDING_DIM}]) \
             ON CONFLICT (exchange_id) DO UPDATE SET embedding = EXCLUDED.embedding"
        ),
        duckdb::params![exchange_id, emb_value],
    )?;
    Ok(())
}

/// Upsert multiple verbatim embeddings in a prepared-statement loop.
/// Caller must manage the surrounding transaction.
pub fn store_verbatim_embeddings_batch(
    conn: &Connection,
    embeddings: &[(String, Vec<f32>)],
) -> StorageResult<()> {
    if embeddings.is_empty() {
        return Ok(());
    }

    let sql = format!(
        "INSERT INTO verbatim_embeddings (exchange_id, embedding) \
         VALUES (?, ?::FLOAT[{EMBEDDING_DIM}]) \
         ON CONFLICT (exchange_id) DO UPDATE SET embedding = EXCLUDED.embedding"
    );
    let mut stmt = conn.prepare(&sql)?;

    for (exchange_id, embedding) in embeddings {
        debug_assert_eq!(embedding.len(), EMBEDDING_DIM);
        let emb_value = embedding_to_value(embedding);
        stmt.execute(duckdb::params![exchange_id, emb_value])?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Palace objects
// ---------------------------------------------------------------------------

/// Upsert a palace object with an optional embedding.
pub fn store_palace_object(
    conn: &Connection,
    obj: &DistilledObject,
    embedding: Option<&[f32]>,
) -> StorageResult<()> {
    let ft_json = serde_json::to_string(
        &obj.files_touched
            .iter()
            .map(|f| serde_json::json!({"path": f.path, "action": f.action}))
            .collect::<Vec<_>>(),
    )?;

    let sql = format!(
        "INSERT INTO palace_objects (
            object_id, exchange_id, conversation_id, project_id,
            ply_start, ply_end, files_touched, exchange_core,
            specific_context, distilled_text, embedding,
            created_at, exchange_at
        ) VALUES (?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?::FLOAT[{EMBEDDING_DIM}], ?, ?)
        ON CONFLICT (object_id) DO UPDATE SET
            exchange_id      = EXCLUDED.exchange_id,
            files_touched    = EXCLUDED.files_touched,
            exchange_core    = EXCLUDED.exchange_core,
            specific_context = EXCLUDED.specific_context,
            distilled_text   = EXCLUDED.distilled_text,
            embedding        = EXCLUDED.embedding"
    );

    let emb_value: Box<dyn ToSql> = match embedding {
        Some(emb) => {
            debug_assert_eq!(emb.len(), EMBEDDING_DIM);
            Box::new(embedding_to_value(emb))
        }
        None => Box::new(duckdb::types::Value::Null),
    };

    conn.execute(
        &sql,
        duckdb::params![
            obj.object_id,
            obj.conversation_id,
            obj.project_id,
            obj.ply_start as i32,
            obj.ply_end as i32,
            ft_json,
            obj.exchange_core,
            obj.specific_context,
            obj.distilled_text,
            &*emb_value,
            obj.created_at,
            obj.exchange_at,
        ],
    )?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Rooms
// ---------------------------------------------------------------------------

/// Upsert a room record.
pub fn store_room(conn: &Connection, room: &Room) -> StorageResult<()> {
    conn.execute(
        "INSERT INTO rooms (
            room_id, room_type, room_key, room_label, project_id,
            created_at, updated_at, object_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (room_id) DO UPDATE SET
            updated_at   = EXCLUDED.updated_at,
            object_count = EXCLUDED.object_count",
        duckdb::params![
            room.room_id,
            room.room_type,
            room.room_key,
            room.room_label,
            room.project_id,
            room.created_at,
            room.updated_at,
            room.object_count,
        ],
    )?;
    Ok(())
}

/// Insert a room-object junction. Errors if room or object don't exist.
pub fn store_room_object(conn: &Connection, junction: &RoomObject) -> StorageResult<()> {
    // Existence checks mirror the Python implementation.
    let room_exists: bool = conn
        .query_row(
            "SELECT 1 FROM rooms WHERE room_id = ? LIMIT 1",
            duckdb::params![junction.room_id],
            |_| Ok(true),
        )
        .unwrap_or(false);

    if !room_exists {
        return Err(StorageError::RoomNotFound(junction.room_id.clone()));
    }

    let object_exists: bool = conn
        .query_row(
            "SELECT 1 FROM palace_objects WHERE object_id = ? LIMIT 1",
            duckdb::params![junction.object_id],
            |_| Ok(true),
        )
        .unwrap_or(false);

    if !object_exists {
        return Err(StorageError::ObjectNotFound(junction.object_id.clone()));
    }

    conn.execute(
        "INSERT INTO room_objects (room_id, object_id, relevance, placed_at)
         VALUES (?, ?, ?, ?)
         ON CONFLICT (room_id, object_id) DO NOTHING",
        duckdb::params![
            junction.room_id,
            junction.object_id,
            junction.relevance,
            junction.placed_at,
        ],
    )?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Facet embeddings
// ---------------------------------------------------------------------------

/// Upsert a single flat facet embedding.
pub fn store_facet_embedding(
    conn: &Connection,
    facet_id: &str,
    facet_type: &str,
    facet_text: &str,
    project_ids: &[String],
    embedding: &[f32],
    last_seen: Option<DateTime<Utc>>,
) -> StorageResult<()> {
    debug_assert_eq!(embedding.len(), EMBEDDING_DIM);
    let mut sorted = project_ids.to_vec();
    sorted.sort();
    let project_ids_json = serde_json::to_string(&sorted)?;
    let last_seen = last_seen.unwrap_or_else(Utc::now);
    let emb_value = embedding_to_value(embedding);

    conn.execute(
        &format!(
            "INSERT INTO facet_embeddings (
                facet_id, facet_type, facet_text, project_ids,
                project_count, embedding, last_seen
            ) VALUES (?, ?, ?, ?, ?, ?::FLOAT[{EMBEDDING_DIM}], ?)
            ON CONFLICT (facet_id) DO UPDATE SET
                project_ids   = EXCLUDED.project_ids,
                project_count = EXCLUDED.project_count,
                embedding     = EXCLUDED.embedding,
                last_seen     = EXCLUDED.last_seen"
        ),
        duckdb::params![
            facet_id,
            facet_type,
            facet_text,
            project_ids_json,
            project_ids.len() as i32,
            emb_value,
            last_seen,
        ],
    )?;

    Ok(())
}

/// Upsert a single hierarchical facet embedding.
#[allow(clippy::too_many_arguments)]
pub fn store_hierarchical_facet(
    conn: &Connection,
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
    debug_assert_eq!(embedding.len(), EMBEDDING_DIM);
    let mut sorted = project_ids.to_vec();
    sorted.sort();
    let project_ids_json = serde_json::to_string(&sorted)?;
    let last_seen = last_seen.unwrap_or_else(Utc::now);
    let emb_value = embedding_to_value(embedding);

    conn.execute(
        &format!(
            "INSERT INTO hierarchical_facets (
                facet_id, facet_type, facet_level, facet_text,
                weight, weighted_count, project_ids, project_count,
                embedding, last_seen
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?::FLOAT[{EMBEDDING_DIM}], ?)
            ON CONFLICT (facet_id) DO UPDATE SET
                weight         = EXCLUDED.weight,
                weighted_count = EXCLUDED.weighted_count,
                project_ids    = EXCLUDED.project_ids,
                project_count  = EXCLUDED.project_count,
                embedding      = EXCLUDED.embedding,
                last_seen      = EXCLUDED.last_seen"
        ),
        duckdb::params![
            facet_id,
            facet_type,
            facet_level,
            facet_text,
            weight,
            weighted_count,
            project_ids_json,
            project_ids.len() as i32,
            emb_value,
            last_seen,
        ],
    )?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Source file state
// ---------------------------------------------------------------------------

pub fn mark_source_file_invalid(
    conn: &Connection,
    file_path: &str,
    conversation_id: &str,
    file_size: i64,
    mtime_ns: i64,
    error_message: &str,
) -> StorageResult<()> {
    conn.execute(
        "INSERT INTO source_file_state (
            file_path, conversation_id, status, file_size, mtime_ns,
            error_message, updated_at
        ) VALUES (?, ?, 'invalid', ?, ?, ?, ?)
        ON CONFLICT (file_path) DO UPDATE SET
            conversation_id = EXCLUDED.conversation_id,
            status          = EXCLUDED.status,
            file_size       = EXCLUDED.file_size,
            mtime_ns        = EXCLUDED.mtime_ns,
            error_message   = EXCLUDED.error_message,
            updated_at      = EXCLUDED.updated_at",
        duckdb::params![
            file_path,
            conversation_id,
            file_size,
            mtime_ns,
            error_message,
            Utc::now(),
        ],
    )?;
    Ok(())
}

pub fn clear_source_file_state(conn: &Connection, file_path: &str) -> StorageResult<()> {
    conn.execute(
        "DELETE FROM source_file_state WHERE file_path = ?",
        duckdb::params![file_path],
    )?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Deletion
// ---------------------------------------------------------------------------

/// Delete all exchange-layer data for a conversation (exchanges, verbatim embeddings,
/// palace objects, room objects). Does NOT delete the conversation or messages rows.
pub fn delete_exchange_data(conn: &Connection, conversation_id: &str) -> StorageResult<()> {
    conn.execute(
        "DELETE FROM room_objects WHERE object_id IN (
             SELECT object_id FROM palace_objects WHERE conversation_id = ?
         )",
        duckdb::params![conversation_id],
    )?;
    conn.execute(
        "DELETE FROM palace_objects WHERE conversation_id = ?",
        duckdb::params![conversation_id],
    )?;
    conn.execute(
        "DELETE FROM verbatim_embeddings WHERE exchange_id IN (
             SELECT exchange_id FROM exchanges WHERE conversation_id = ?
         )",
        duckdb::params![conversation_id],
    )?;
    conn.execute(
        "DELETE FROM exchanges WHERE conversation_id = ?",
        duckdb::params![conversation_id],
    )?;
    Ok(())
}

/// Delete exchange-layer data from `ply_start` onward.
pub fn delete_exchange_data_from_ply(
    conn: &Connection,
    conversation_id: &str,
    ply_start: i32,
) -> StorageResult<()> {
    conn.execute(
        "DELETE FROM room_objects WHERE object_id IN (
             SELECT object_id FROM palace_objects
             WHERE conversation_id = ? AND ply_start >= ?
         )",
        duckdb::params![conversation_id, ply_start],
    )?;
    conn.execute(
        "DELETE FROM palace_objects WHERE conversation_id = ? AND ply_start >= ?",
        duckdb::params![conversation_id, ply_start],
    )?;
    conn.execute(
        "DELETE FROM verbatim_embeddings WHERE exchange_id IN (
             SELECT exchange_id FROM exchanges
             WHERE conversation_id = ? AND ply_start >= ?
         )",
        duckdb::params![conversation_id, ply_start],
    )?;
    conn.execute(
        "DELETE FROM exchanges WHERE conversation_id = ? AND ply_start >= ?",
        duckdb::params![conversation_id, ply_start],
    )?;
    Ok(())
}

/// Delete a conversation and all dependent rows. Returns per-table deleted counts.
pub fn delete_conversation(
    conn: &Connection,
    conversation_id: &str,
) -> StorageResult<std::collections::HashMap<String, usize>> {
    let mut counts = std::collections::HashMap::new();

    begin_transaction(conn)?;
    let result = (|| -> StorageResult<()> {
        counts.insert(
            "verbatim_embeddings".into(),
            conn.execute(
                "DELETE FROM verbatim_embeddings WHERE exchange_id IN (
                     SELECT exchange_id FROM exchanges WHERE conversation_id = ?
                 )",
                duckdb::params![conversation_id],
            )?,
        );
        counts.insert(
            "room_objects".into(),
            conn.execute(
                "DELETE FROM room_objects WHERE object_id IN (
                     SELECT object_id FROM palace_objects WHERE conversation_id = ?
                 )",
                duckdb::params![conversation_id],
            )?,
        );
        counts.insert(
            "palace_objects".into(),
            conn.execute(
                "DELETE FROM palace_objects WHERE conversation_id = ?",
                duckdb::params![conversation_id],
            )?,
        );
        counts.insert(
            "exchanges".into(),
            conn.execute(
                "DELETE FROM exchanges WHERE conversation_id = ?",
                duckdb::params![conversation_id],
            )?,
        );
        counts.insert(
            "messages".into(),
            conn.execute(
                "DELETE FROM messages WHERE conversation_id = ?",
                duckdb::params![conversation_id],
            )?,
        );
        counts.insert(
            "conversations".into(),
            conn.execute(
                "DELETE FROM conversations WHERE conversation_id = ?",
                duckdb::params![conversation_id],
            )?,
        );
        Ok(())
    })();

    match &result {
        Ok(_) => commit(conn)?,
        Err(_) => rollback(conn),
    }
    result?;

    Ok(counts)
}

/// Bulk-update file_size + mtime_ns after the migration backfill.
/// `rows` is `(file_size, mtime_ns, conversation_id)`.
pub fn backfill_stat_columns(
    conn: &Connection,
    rows: &[(i64, i64, String)],
) -> StorageResult<()> {
    let mut stmt = conn.prepare(
        "UPDATE conversations SET file_size = ?, mtime_ns = ? WHERE conversation_id = ?",
    )?;
    for (file_size, mtime_ns, conv_id) in rows {
        stmt.execute(duckdb::params![file_size, mtime_ns, conv_id])?;
    }
    Ok(())
}
