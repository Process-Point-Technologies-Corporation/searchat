/// All CREATE TABLE DDL, migrations, and index creation.
use duckdb::Connection;

use crate::error::StorageResult;

pub const EMBEDDING_DIM: usize = 384;

/// Create all tables if they don't exist, then run migrations.
pub fn ensure_tables(conn: &Connection, vss_available: bool) -> StorageResult<()> {
    conn.execute_batch(&format!(
        r#"
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id VARCHAR PRIMARY KEY,
            project_id      VARCHAR NOT NULL,
            file_path       VARCHAR NOT NULL,
            title           VARCHAR NOT NULL,
            created_at      TIMESTAMP NOT NULL,
            updated_at      TIMESTAMP NOT NULL,
            message_count   INTEGER NOT NULL,
            full_text       TEXT NOT NULL,
            file_hash       VARCHAR NOT NULL,
            indexed_at      TIMESTAMP NOT NULL
        );

        CREATE TABLE IF NOT EXISTS source_file_state (
            file_path       VARCHAR PRIMARY KEY,
            conversation_id VARCHAR,
            status          VARCHAR NOT NULL,
            file_size       BIGINT NOT NULL,
            mtime_ns        BIGINT NOT NULL,
            error_message   TEXT,
            updated_at      TIMESTAMP NOT NULL
        );

        CREATE TABLE IF NOT EXISTS messages (
            conversation_id VARCHAR NOT NULL,
            sequence        INTEGER NOT NULL,
            role            VARCHAR NOT NULL,
            content         TEXT NOT NULL,
            timestamp       TIMESTAMP,
            has_code        BOOLEAN DEFAULT FALSE,
            PRIMARY KEY (conversation_id, sequence)
        );

        CREATE TABLE IF NOT EXISTS exchanges (
            exchange_id     VARCHAR PRIMARY KEY,
            conversation_id VARCHAR NOT NULL,
            project_id      VARCHAR,
            ply_start       INTEGER NOT NULL,
            ply_end         INTEGER NOT NULL,
            exchange_text   TEXT NOT NULL,
            created_at      TIMESTAMP NOT NULL,
            UNIQUE(conversation_id, ply_start, ply_end)
        );

        CREATE TABLE IF NOT EXISTS verbatim_embeddings (
            exchange_id VARCHAR PRIMARY KEY,
            embedding   FLOAT[{dim}] NOT NULL
        );

        CREATE TABLE IF NOT EXISTS palace_objects (
            object_id        VARCHAR PRIMARY KEY,
            exchange_id      VARCHAR,
            conversation_id  VARCHAR NOT NULL,
            project_id       VARCHAR NOT NULL,
            ply_start        INTEGER NOT NULL,
            ply_end          INTEGER NOT NULL,
            files_touched    JSON,
            exchange_core    VARCHAR NOT NULL,
            specific_context VARCHAR NOT NULL,
            distilled_text   VARCHAR NOT NULL,
            embedding        FLOAT[{dim}],
            created_at       TIMESTAMP NOT NULL,
            exchange_at      TIMESTAMP NOT NULL
        );

        CREATE TABLE IF NOT EXISTS rooms (
            room_id      VARCHAR PRIMARY KEY,
            room_type    VARCHAR NOT NULL,
            room_key     VARCHAR NOT NULL,
            room_label   VARCHAR NOT NULL,
            project_id   VARCHAR,
            created_at   TIMESTAMP NOT NULL,
            updated_at   TIMESTAMP NOT NULL,
            object_count INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS room_objects (
            room_id    VARCHAR NOT NULL,
            object_id  VARCHAR NOT NULL,
            relevance  FLOAT NOT NULL,
            placed_at  TIMESTAMP NOT NULL,
            PRIMARY KEY (room_id, object_id)
        );

        CREATE TABLE IF NOT EXISTS facet_embeddings (
            facet_id      VARCHAR PRIMARY KEY,
            facet_type    VARCHAR NOT NULL,
            facet_text    VARCHAR NOT NULL,
            project_ids   JSON NOT NULL,
            project_count INTEGER NOT NULL,
            embedding     FLOAT[{dim}] NOT NULL,
            last_seen     TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS hierarchical_facets (
            facet_id       VARCHAR PRIMARY KEY,
            facet_type     VARCHAR NOT NULL,
            facet_level    VARCHAR NOT NULL,
            facet_text     VARCHAR NOT NULL,
            weight         FLOAT NOT NULL,
            weighted_count FLOAT NOT NULL,
            project_ids    JSON NOT NULL,
            project_count  INTEGER NOT NULL,
            embedding      FLOAT[{dim}] NOT NULL,
            last_seen      TIMESTAMP
        );
        "#,
        dim = EMBEDDING_DIM
    ))?;

    run_migrations(conn)?;

    if vss_available {
        create_hnsw_indexes(conn);
    }

    Ok(())
}

/// ALTER TABLE migrations — all idempotent.
fn run_migrations(conn: &Connection) -> StorageResult<()> {
    // file_size + mtime_ns columns on conversations
    let _ = conn.execute_batch(
        "ALTER TABLE conversations ADD COLUMN file_size BIGINT DEFAULT 0",
    );
    let _ = conn.execute_batch(
        "ALTER TABLE conversations ADD COLUMN mtime_ns BIGINT DEFAULT 0",
    );

    // last_seen columns (ADD COLUMN IF NOT EXISTS supported in DuckDB)
    let _ = conn.execute_batch(
        "ALTER TABLE facet_embeddings ADD COLUMN IF NOT EXISTS last_seen TIMESTAMP",
    );
    let _ = conn.execute_batch(
        "ALTER TABLE hierarchical_facets ADD COLUMN IF NOT EXISTS last_seen TIMESTAMP",
    );

    Ok(())
}

/// Create HNSW indexes for approximate vector search.
/// Each index creation is attempted independently; failures are logged, not fatal.
pub fn create_hnsw_indexes(conn: &Connection) {
    let existing: Vec<String> = {
        let mut stmt = match conn.prepare(
            "SELECT index_name FROM duckdb_indexes() \
             WHERE index_name IN ('verbatim_hnsw','palace_hnsw','facet_hnsw','hierarchical_facet_hnsw')",
        ) {
            Ok(s) => s,
            Err(e) => {
                log::warn!("Failed to query existing HNSW indexes: {e}");
                return;
            }
        };
        match stmt.query_map([], |row| row.get(0)) {
            Ok(rows) => rows.filter_map(|r| r.ok()).collect(),
            Err(e) => {
                log::warn!("Failed to fetch HNSW index names: {e}");
                return;
            }
        }
    };

    let indexes = [
        (
            "verbatim_hnsw",
            "CREATE INDEX verbatim_hnsw ON verbatim_embeddings \
             USING HNSW (embedding) WITH (metric = 'cosine')",
        ),
        (
            "palace_hnsw",
            "CREATE INDEX palace_hnsw ON palace_objects \
             USING HNSW (embedding) WITH (metric = 'cosine')",
        ),
        (
            "facet_hnsw",
            "CREATE INDEX facet_hnsw ON facet_embeddings \
             USING HNSW (embedding) WITH (metric = 'cosine')",
        ),
        (
            "hierarchical_facet_hnsw",
            "CREATE INDEX hierarchical_facet_hnsw ON hierarchical_facets \
             USING HNSW (embedding) WITH (metric = 'cosine')",
        ),
    ];

    for (name, ddl) in &indexes {
        if existing.contains(&name.to_string()) {
            continue;
        }
        if let Err(e) = conn.execute_batch(ddl) {
            log::warn!("Failed to create HNSW index {name}: {e}");
        } else {
            log::info!("Created HNSW index {name}");
        }
    }
}

/// Create FTS indexes on exchanges.exchange_text and palace_objects.distilled_text.
pub fn create_fts_indexes(conn: &Connection) -> StorageResult<()> {
    conn.execute_batch(
        "PRAGMA create_fts_index(\
            'exchanges', 'exchange_id', 'exchange_text', \
            stemmer = 'porter', stopwords = 'english', overwrite = 1\
        )",
    )?;
    log::info!("Created FTS index on exchanges.exchange_text");

    conn.execute_batch(
        "PRAGMA create_fts_index(\
            'palace_objects', 'object_id', 'distilled_text', \
            stemmer = 'porter', stopwords = 'english', overwrite = 1\
        )",
    )?;
    log::info!("Created FTS index on palace_objects.distilled_text");

    Ok(())
}
