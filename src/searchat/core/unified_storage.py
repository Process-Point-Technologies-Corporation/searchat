"""Unified DuckDB storage with native vector search (VSS) and full-text search (FTS).

This module provides a single database for all searchat data:
- Conversations and messages
- Exchanges (segmented conversation turns)
- Verbatim embeddings (exchange-level vectors)
- Palace objects (distilled exchanges)
- Rooms and room-object junctions
- Facet embeddings (vocabulary-level index for semantic facet resolution)

Uses DuckDB extensions:
- VSS: HNSW indexes for approximate nearest neighbor search
- FTS: Full-text search indexes for BM25 keyword search
"""
import hashlib
import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import duckdb
import numpy as np

from searchat.config.constants import DEFAULT_EXCLUDED_PROMPT_PREFIXES
from searchat.models.domain import (
    ConversationRecord,
    DistilledObject,
    FileTouched,
    Room,
    RoomObject,
)

logger = logging.getLogger(__name__)

# Embedding dimension for all-MiniLM-L6-v2
EMBEDDING_DIM = 384


def make_facet_id(facet_type: str, facet_text: str) -> str:
    """Deterministic facet ID from type + text."""
    key = f"{facet_type}:{facet_text}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


class UnifiedStorage:
    """Unified DuckDB storage for all searchat data with native vector search."""

    def __init__(
        self,
        data_dir: Path,
        conn: Optional[duckdb.DuckDBPyConnection] = None,
    ):
        """Initialize unified storage.

        Args:
            data_dir: Directory for searchat.duckdb file.
            conn: Optional pre-existing connection (for testing with :memory:).
        """
        self.data_dir = data_dir
        self._vss_available = False
        self._fts_available = False
        self._local = threading.local()
        self._fts_lock = threading.Lock()

        if conn is not None:
            self.conn = conn
            self._external_conn = True
        else:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = data_dir / "searchat.duckdb"
            self.conn = duckdb.connect(str(self.db_path))
            self._external_conn = False

        self._load_extensions()
        self._ensure_tables()

    def _get_cursor(self) -> duckdb.DuckDBPyConnection:
        """Get or create a thread-local WRITE cursor.

        Used for INSERT/UPDATE/DELETE/DDL operations. Routes through
        the read-write connection. Must be serialized by indexing_lock
        for concurrent safety.
        """
        if not hasattr(self._local, "cursor") or self._local.cursor is None:
            self._local.cursor = self.conn.cursor()
        return self._local.cursor

    def _get_read_cursor(self) -> duckdb.DuckDBPyConnection:
        """Get a fresh cursor for read-only queries.

        Creates a new cursor each call to avoid lingering read transactions
        that conflict with write COMMITs on other cursors. DuckDB cursor
        creation is cheap (~0.01ms). The cursor auto-closes when GC'd.
        """
        return self.conn.cursor()

    def _load_extensions(self) -> None:
        """Load VSS and FTS extensions on both write and read connections.

        Tries LOAD first (no network). Falls back to INSTALL + LOAD only
        when the extension has never been installed.
        """
        for ext_name, attr_name in [("vss", "_vss_available"), ("fts", "_fts_available")]:
            try:
                self._get_cursor().execute(f"LOAD {ext_name}")
                setattr(self, attr_name, True)
                logger.info("DuckDB %s extension loaded", ext_name)
            except Exception:
                try:
                    self._get_cursor().execute(f"INSTALL {ext_name}")
                    self._get_cursor().execute(f"LOAD {ext_name}")
                    setattr(self, attr_name, True)
                    logger.info("DuckDB %s extension installed and loaded", ext_name)
                except Exception as e:
                    logger.warning("DuckDB %s extension not available: %s", ext_name, e)
                    setattr(self, attr_name, False)


        if self._vss_available:
            # Disable HNSW persistence: the internal background thread it spawns
            # causes write-write conflicts with per-conversation transactions.
            # The HNSW index rebuilds from embeddings data on restart (~1s for 16K vectors).
            self._get_cursor().execute("SET hnsw_enable_experimental_persistence = false")

    def _ensure_tables(self) -> None:
        """Create all tables if they don't exist."""
        # Conversations table
        self._get_cursor().execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id VARCHAR PRIMARY KEY,
                project_id VARCHAR NOT NULL,
                file_path VARCHAR NOT NULL,
                title VARCHAR NOT NULL,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                message_count INTEGER NOT NULL,
                full_text TEXT NOT NULL,
                file_hash VARCHAR NOT NULL,
                indexed_at TIMESTAMP NOT NULL
            )
        """)

        # Migration: add file_size column for change detection
        try:
            self._get_cursor().execute(
                "ALTER TABLE conversations ADD COLUMN file_size BIGINT DEFAULT 0"
            )
        except Exception:
            pass  # Column already exists

        # Migration: add mtime_ns column for change detection
        try:
            self._get_cursor().execute(
                "ALTER TABLE conversations ADD COLUMN mtime_ns BIGINT DEFAULT 0"
            )
        except Exception:
            pass  # Column already exists

        self._get_cursor().execute("""
            CREATE TABLE IF NOT EXISTS source_file_state (
                file_path VARCHAR PRIMARY KEY,
                conversation_id VARCHAR,
                status VARCHAR NOT NULL,
                file_size BIGINT NOT NULL,
                mtime_ns BIGINT NOT NULL,
                error_message TEXT,
                updated_at TIMESTAMP NOT NULL
            )
        """)

        self._create_core_tables()

        # Facet embeddings — vocabulary-level index for semantic facet resolution.
        # One row per unique facet value (filename, room key, project fragment),
        # not per palace object. Used to resolve query terms to project scopes.
        # Includes last_seen for temporal decay weighting.
        self._get_cursor().execute(f"""
            CREATE TABLE IF NOT EXISTS facet_embeddings (
                facet_id VARCHAR PRIMARY KEY,
                facet_type VARCHAR NOT NULL,
                facet_text VARCHAR NOT NULL,
                project_ids JSON NOT NULL,
                project_count INTEGER NOT NULL,
                embedding FLOAT[{EMBEDDING_DIM}] NOT NULL,
                last_seen TIMESTAMP
            )
        """)

        # Migration: Add last_seen column if it doesn't exist
        self._get_cursor().execute("""
            ALTER TABLE facet_embeddings
            ADD COLUMN IF NOT EXISTS last_seen TIMESTAMP
        """)

        # Hierarchical facet embeddings — multi-level file facets with weighted distinctiveness.
        # Stores full path (weight 3x), directory (weight 2x), and basename (weight 1x).
        self._get_cursor().execute(f"""
            CREATE TABLE IF NOT EXISTS hierarchical_facets (
                facet_id VARCHAR PRIMARY KEY,
                facet_type VARCHAR NOT NULL,
                facet_level VARCHAR NOT NULL,
                facet_text VARCHAR NOT NULL,
                weight FLOAT NOT NULL,
                weighted_count FLOAT NOT NULL,
                project_ids JSON NOT NULL,
                project_count INTEGER NOT NULL,
                embedding FLOAT[{EMBEDDING_DIM}] NOT NULL,
                last_seen TIMESTAMP
            )
        """)

        # Migration: Add last_seen column if it doesn't exist
        self._get_cursor().execute("""
            ALTER TABLE hierarchical_facets
            ADD COLUMN IF NOT EXISTS last_seen TIMESTAMP
        """)

        # Migration: drop foreign key constraints from existing tables.
        # DuckDB has no ALTER TABLE DROP CONSTRAINT — recreate tables without FKs.
        self._migrate_drop_foreign_keys()

        # Create HNSW indexes if VSS is available
        if self._vss_available:
            self._create_hnsw_indexes()

    def _create_core_tables(self) -> None:
        """Create messages, exchanges, embeddings, palace, and room tables."""
        self._get_cursor().execute("""
            CREATE TABLE IF NOT EXISTS messages (
                conversation_id VARCHAR NOT NULL,
                sequence INTEGER NOT NULL,
                role VARCHAR NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP,
                has_code BOOLEAN DEFAULT FALSE,
                PRIMARY KEY (conversation_id, sequence)
            )
        """)

        self._get_cursor().execute("""
            CREATE TABLE IF NOT EXISTS exchanges (
                exchange_id VARCHAR PRIMARY KEY,
                conversation_id VARCHAR NOT NULL,
                project_id VARCHAR,
                ply_start INTEGER NOT NULL,
                ply_end INTEGER NOT NULL,
                exchange_text TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                UNIQUE(conversation_id, ply_start, ply_end)
            )
        """)

        self._get_cursor().execute(f"""
            CREATE TABLE IF NOT EXISTS verbatim_embeddings (
                exchange_id VARCHAR PRIMARY KEY,
                embedding FLOAT[{EMBEDDING_DIM}] NOT NULL
            )
        """)

        self._get_cursor().execute(f"""
            CREATE TABLE IF NOT EXISTS palace_objects (
                object_id VARCHAR PRIMARY KEY,
                exchange_id VARCHAR,
                conversation_id VARCHAR NOT NULL,
                project_id VARCHAR NOT NULL,
                ply_start INTEGER NOT NULL,
                ply_end INTEGER NOT NULL,
                files_touched JSON,
                exchange_core VARCHAR NOT NULL,
                specific_context VARCHAR NOT NULL,
                distilled_text VARCHAR NOT NULL,
                embedding FLOAT[{EMBEDDING_DIM}],
                created_at TIMESTAMP NOT NULL,
                exchange_at TIMESTAMP NOT NULL
            )
        """)

        self._get_cursor().execute("""
            CREATE TABLE IF NOT EXISTS rooms (
                room_id VARCHAR PRIMARY KEY,
                room_type VARCHAR NOT NULL,
                room_key VARCHAR NOT NULL,
                room_label VARCHAR NOT NULL,
                project_id VARCHAR,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                object_count INTEGER DEFAULT 0
            )
        """)

        self._get_cursor().execute("""
            CREATE TABLE IF NOT EXISTS room_objects (
                room_id VARCHAR NOT NULL,
                object_id VARCHAR NOT NULL,
                relevance FLOAT NOT NULL,
                placed_at TIMESTAMP NOT NULL,
                PRIMARY KEY (room_id, object_id)
            )
        """)

    def _migrate_drop_foreign_keys(self) -> None:
        """Drop FK constraints by recreating affected tables.

        DuckDB lacks ALTER TABLE DROP CONSTRAINT and ADD UNIQUE.
        Strategy: save data to tmp, drop originals (leaves first),
        let _ensure_tables recreate them without FKs, restore data.

        Idempotent: no-op if no FK constraints exist.
        """
        cursor = self._get_cursor()

        fk_count = cursor.execute("""
            SELECT count(*) FROM duckdb_constraints()
            WHERE constraint_type = 'FOREIGN KEY'
        """).fetchone()[0]
        if fk_count == 0:
            return

        logger.info("Migrating: dropping %d foreign key constraints", fk_count)

        tables = ["room_objects", "palace_objects", "verbatim_embeddings",
                   "exchanges", "messages"]

        # Save data and drop (leaves first to satisfy existing FKs)
        for t in tables:
            cursor.execute(f"CREATE TABLE {t}__bak AS SELECT * FROM {t}")
            cursor.execute(f"DROP TABLE {t}")

        # Recreate without FKs — rerun the CREATE TABLE IF NOT EXISTS
        # statements from _ensure_tables. They're idempotent.
        self._create_core_tables()

        # Restore data
        for t in tables:
            cursor.execute(f"INSERT INTO {t} SELECT * FROM {t}__bak")
            cursor.execute(f"DROP TABLE {t}__bak")

        logger.info("Migration complete: FK constraints removed")

    def _create_hnsw_indexes(self) -> None:
        """Create HNSW indexes for vector search."""
        # Check if indexes already exist
        existing_indexes = self._get_cursor().execute("""
            SELECT index_name FROM duckdb_indexes()
            WHERE index_name IN ('verbatim_hnsw', 'palace_hnsw', 'facet_hnsw', 'hierarchical_facet_hnsw')
        """).fetchall()
        existing_names = {row[0] for row in existing_indexes}

        if "verbatim_hnsw" not in existing_names:
            try:
                self._get_cursor().execute("""
                    CREATE INDEX verbatim_hnsw ON verbatim_embeddings
                    USING HNSW (embedding) WITH (metric = 'cosine')
                """)
                logger.info("Created HNSW index on verbatim_embeddings")
            except Exception as e:
                logger.warning("Failed to create verbatim_hnsw index: %s", e)

        if "palace_hnsw" not in existing_names:
            try:
                self._get_cursor().execute("""
                    CREATE INDEX palace_hnsw ON palace_objects
                    USING HNSW (embedding) WITH (metric = 'cosine')
                """)
                logger.info("Created HNSW index on palace_objects")
            except Exception as e:
                logger.warning("Failed to create palace_hnsw index: %s", e)

        if "facet_hnsw" not in existing_names:
            try:
                self._get_cursor().execute("""
                    CREATE INDEX facet_hnsw ON facet_embeddings
                    USING HNSW (embedding) WITH (metric = 'cosine')
                """)
                logger.info("Created HNSW index on facet_embeddings")
            except Exception as e:
                logger.warning("Failed to create facet_hnsw index: %s", e)

        if "hierarchical_facet_hnsw" not in existing_names:
            try:
                self._get_cursor().execute("""
                    CREATE INDEX hierarchical_facet_hnsw ON hierarchical_facets
                    USING HNSW (embedding) WITH (metric = 'cosine')
                """)
                logger.info("Created HNSW index on hierarchical_facets")
            except Exception as e:
                logger.warning("Failed to create hierarchical_facet_hnsw index: %s", e)

    def create_fts_index(self) -> None:
        """Create full-text search indexes on exchanges and palace_objects tables.

        Uses a threading lock to prevent concurrent FTS catalog writes
        (DuckDB cannot serialize concurrent PRAGMA create_fts_index calls).
        """
        if not self._fts_available:
            logger.warning("FTS extension not available, skipping FTS index creation")
            return

        if not self._fts_lock.acquire(blocking=False):
            logger.info("FTS index creation already in progress, skipping")
            return

        try:
            # FTS index on exchanges.exchange_text
            self._get_cursor().execute("""
                PRAGMA create_fts_index(
                    'exchanges', 'exchange_id', 'exchange_text',
                    stemmer = 'porter',
                    stopwords = 'english',
                    overwrite = 1
                )
            """)
            logger.info("Created FTS index on exchanges.exchange_text")

            # FTS index on palace_objects.distilled_text
            self._get_cursor().execute("""
                PRAGMA create_fts_index(
                    'palace_objects', 'object_id', 'distilled_text',
                    stemmer = 'porter',
                    stopwords = 'english',
                    overwrite = 1
                )
            """)
            logger.info("Created FTS index on palace_objects.distilled_text")
        except Exception as e:
            logger.error("Failed to create FTS index: %s", e)
            raise
        finally:
            self._fts_lock.release()

    @property
    def vss_available(self) -> bool:
        """Check if VSS extension is available."""
        return self._vss_available

    @property
    def fts_available(self) -> bool:
        """Check if FTS extension is available."""
        return self._fts_available

    # =========================================================================
    # Conversation CRUD
    # =========================================================================

    def store_conversation(
        self, record: ConversationRecord, in_transaction: bool = False,
    ) -> None:
        """Store a conversation and its messages.

        Args:
            record: The conversation record to store.
            in_transaction: If True, caller manages the transaction.
                Skip BEGIN/COMMIT/ROLLBACK here.
        """
        if not in_transaction:
            self._begin_transaction()
        try:
            # Insert conversation
            self._get_cursor().execute("""
                INSERT INTO conversations (
                    conversation_id, project_id, file_path, title,
                    created_at, updated_at, message_count, full_text,
                    file_hash, indexed_at, file_size, mtime_ns
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (conversation_id) DO UPDATE SET
                    project_id = EXCLUDED.project_id,
                    file_path = EXCLUDED.file_path,
                    title = EXCLUDED.title,
                    updated_at = EXCLUDED.updated_at,
                    message_count = EXCLUDED.message_count,
                    full_text = EXCLUDED.full_text,
                    file_hash = EXCLUDED.file_hash,
                    indexed_at = EXCLUDED.indexed_at,
                    file_size = EXCLUDED.file_size,
                    mtime_ns = EXCLUDED.mtime_ns
            """, [
                record.conversation_id, record.project_id, record.file_path,
                record.title, record.created_at, record.updated_at,
                record.message_count, record.full_text, record.file_hash,
                record.indexed_at, record.file_size, record.mtime_ns,
            ])

            # Insert messages (batched)
            if record.messages:
                message_data = [
                    [record.conversation_id, msg.sequence, msg.role,
                     msg.content, msg.timestamp, msg.has_code]
                    for msg in record.messages
                ]
                self._get_cursor().executemany("""
                    INSERT INTO messages (
                        conversation_id, sequence, role, content, timestamp, has_code
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT (conversation_id, sequence) DO UPDATE SET
                        role = EXCLUDED.role,
                        content = EXCLUDED.content,
                        timestamp = EXCLUDED.timestamp,
                        has_code = EXCLUDED.has_code
                """, message_data)

            if not in_transaction:
                self._commit()
        except Exception:
            if not in_transaction:
                self._rollback()
            raise

    def get_conversation(self, conversation_id: str) -> Optional[Dict]:
        """Get a conversation by ID."""
        row = self._get_read_cursor().execute("""
            SELECT conversation_id, project_id, file_path, title,
                   created_at, updated_at, message_count, full_text,
                   file_hash, indexed_at, file_size, mtime_ns
            FROM conversations
            WHERE conversation_id = ?
        """, [conversation_id]).fetchone()

        if row is None:
            return None

        return {
            "conversation_id": row[0],
            "project_id": row[1],
            "file_path": row[2],
            "title": row[3],
            "created_at": row[4],
            "updated_at": row[5],
            "message_count": row[6],
            "full_text": row[7],
            "file_hash": row[8],
            "indexed_at": row[9],
            "file_size": row[10],
            "mtime_ns": row[11],
        }

    def get_conversations_batch(self, conversation_ids: List[str]) -> Dict[str, Dict]:
        """Get multiple conversations by ID in a single query.

        Returns {conversation_id: conversation_dict} for found conversations.
        """
        if not conversation_ids:
            return {}
        placeholders = ",".join("?" for _ in conversation_ids)
        rows = self._get_read_cursor().execute(f"""
            SELECT conversation_id, project_id, file_path, title,
                   created_at, updated_at, message_count, full_text,
                   file_hash, indexed_at, file_size, mtime_ns
            FROM conversations
            WHERE conversation_id IN ({placeholders})
        """, conversation_ids).fetchall()
        return {
            row[0]: {
                "conversation_id": row[0], "project_id": row[1],
                "file_path": row[2], "title": row[3],
                "created_at": row[4], "updated_at": row[5],
                "message_count": row[6], "full_text": row[7],
                "file_hash": row[8], "indexed_at": row[9],
                "file_size": row[10], "mtime_ns": row[11],
            }
            for row in rows
        }

    def get_max_message_sequence(self, conversation_id: str) -> int:
        """Get the highest message sequence number for a conversation.

        Returns -1 if no messages exist (so starting_sequence = 0).
        """
        row = self._get_read_cursor().execute("""
            SELECT COALESCE(MAX(sequence), -1) FROM messages WHERE conversation_id = ?
        """, [conversation_id]).fetchone()
        return row[0]

    def get_conversation_messages(self, conversation_id: str) -> List[Dict]:
        """Get all messages for a conversation."""
        rows = self._get_read_cursor().execute("""
            SELECT sequence, role, content, timestamp, has_code
            FROM messages
            WHERE conversation_id = ?
            ORDER BY sequence ASC
        """, [conversation_id]).fetchall()

        return [
            {
                "sequence": r[0],
                "role": r[1],
                "content": r[2],
                "timestamp": r[3],
                "has_code": r[4],
            }
            for r in rows
        ]

    def get_all_conversation_ids(self, project_id: Optional[str] = None) -> List[str]:
        """Get all conversation IDs, optionally filtered by project."""
        if project_id:
            rows = self._get_read_cursor().execute("""
                SELECT conversation_id FROM conversations WHERE project_id = ?
            """, [project_id]).fetchall()
        else:
            rows = self._get_read_cursor().execute("""
                SELECT conversation_id FROM conversations
            """).fetchall()
        return [r[0] for r in rows]

    def get_conversation_hashes(self) -> Dict[str, Tuple[str, str, int, int]]:
        """Get (file_hash, file_path, file_size, mtime_ns) for all indexed conversations."""
        rows = self._get_read_cursor().execute("""
            SELECT conversation_id, file_hash, file_path, file_size, mtime_ns
            FROM conversations
        """).fetchall()
        return {r[0]: (r[1], r[2], r[3] or 0, r[4] or 0) for r in rows}

    def backfill_stat_columns(
        self, rows: List[Tuple[int, int, str]],
    ) -> None:
        """Bulk-update file_size and mtime_ns for conversations after migration.

        Args:
            rows: List of (file_size, mtime_ns, conversation_id) tuples.
        """
        self._get_cursor().executemany(
            "UPDATE conversations SET file_size = ?, mtime_ns = ? WHERE conversation_id = ?",
            rows,
        )

    def get_indexed_file_paths(self) -> Set[str]:
        """Get all file paths already indexed in the conversations table."""
        rows = self._get_read_cursor().execute("""
            SELECT file_path FROM conversations
        """).fetchall()
        return {r[0] for r in rows}

    def get_source_file_state(
        self, file_paths: Optional[List[str]] = None,
    ) -> Dict[str, Dict]:
        """Get cached source-file processing state keyed by file path."""
        if file_paths:
            placeholders = ",".join(["?"] * len(file_paths))
            rows = self._get_read_cursor().execute(f"""
                SELECT file_path, conversation_id, status, file_size, mtime_ns,
                       error_message, updated_at
                FROM source_file_state
                WHERE file_path IN ({placeholders})
            """, file_paths).fetchall()
        else:
            rows = self._get_read_cursor().execute("""
                SELECT file_path, conversation_id, status, file_size, mtime_ns,
                       error_message, updated_at
                FROM source_file_state
            """).fetchall()

        return {
            row[0]: {
                "file_path": row[0],
                "conversation_id": row[1],
                "status": row[2],
                "file_size": row[3] or 0,
                "mtime_ns": row[4] or 0,
                "error_message": row[5],
                "updated_at": row[6],
            }
            for row in rows
        }

    def mark_source_file_invalid(
        self,
        *,
        file_path: str,
        conversation_id: str,
        file_size: int,
        mtime_ns: int,
        error_message: str,
    ) -> None:
        """Persist invalid source-file state so unchanged bad files can be skipped."""
        self._get_cursor().execute("""
            INSERT INTO source_file_state (
                file_path, conversation_id, status, file_size, mtime_ns,
                error_message, updated_at
            ) VALUES (?, ?, 'invalid', ?, ?, ?, ?)
            ON CONFLICT (file_path) DO UPDATE SET
                conversation_id = EXCLUDED.conversation_id,
                status = EXCLUDED.status,
                file_size = EXCLUDED.file_size,
                mtime_ns = EXCLUDED.mtime_ns,
                error_message = EXCLUDED.error_message,
                updated_at = EXCLUDED.updated_at
        """, [
            file_path,
            conversation_id,
            file_size,
            mtime_ns,
            error_message,
            datetime.utcnow(),
        ])

    def clear_source_file_state(self, file_path: str) -> None:
        """Remove cached source-file state after a successful parse/index."""
        self._get_cursor().execute("""
            DELETE FROM source_file_state WHERE file_path = ?
        """, [file_path])

    def conversation_exists(self, conversation_id: str) -> bool:
        """Check if a conversation exists."""
        row = self._get_read_cursor().execute("""
            SELECT 1 FROM conversations WHERE conversation_id = ? LIMIT 1
        """, [conversation_id]).fetchone()
        return row is not None

    # =========================================================================
    # Exchange CRUD
    # =========================================================================

    def store_exchange(
        self,
        exchange_id: str,
        conversation_id: str,
        project_id: str,
        ply_start: int,
        ply_end: int,
        exchange_text: str,
        created_at: datetime,
        skip_existing_check: bool = False,
    ) -> str:
        """Store an exchange and return the actual exchange_id.

        Args:
            skip_existing_check: If True, skip the SELECT for existing exchange.
                Use when caller has already filtered duplicates (e.g. via
                get_existing_exchange_keys).

        Returns:
            The exchange_id in the database. If exchange already exists
            for (conversation_id, ply_start, ply_end), returns the existing ID.
        """
        if not skip_existing_check:
            # Check if exchange already exists
            existing = self._get_cursor().execute("""
                SELECT exchange_id FROM exchanges
                WHERE conversation_id = ? AND ply_start = ? AND ply_end = ?
            """, [conversation_id, ply_start, ply_end]).fetchone()

            if existing:
                return existing[0]

        # Insert new exchange — no internal error handling.
        # Caller manages the transaction and handles failures.
        self._get_cursor().execute("""
            INSERT INTO exchanges (
                exchange_id, conversation_id, project_id, ply_start, ply_end,
                exchange_text, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [
            exchange_id, conversation_id, project_id, ply_start, ply_end,
            exchange_text, created_at,
        ])
        return exchange_id

    def store_verbatim_embedding(
        self,
        exchange_id: str,
        embedding: np.ndarray,
    ) -> None:
        """Store a verbatim embedding for an exchange."""
        # Convert numpy array to list for DuckDB
        embedding_list = embedding.tolist()
        self._get_cursor().execute("""
            INSERT INTO verbatim_embeddings (exchange_id, embedding)
            VALUES (?, ?::FLOAT[])
            ON CONFLICT (exchange_id) DO UPDATE SET
                embedding = EXCLUDED.embedding
        """, [exchange_id, embedding_list])

    def store_exchanges_batch(
        self,
        exchanges: List[Dict],
        created_at: datetime,
    ) -> None:
        """Store multiple exchanges in a single executemany call.

        Caller manages the transaction. No duplicate checking — caller must
        pre-filter via get_existing_exchange_keys().

        Args:
            exchanges: List of dicts with exchange_id, conversation_id,
                project_id, ply_start, ply_end, exchange_text.
            created_at: Timestamp for all exchanges.
        """
        if not exchanges:
            return
        data = [
            [e["exchange_id"], e["conversation_id"], e["project_id"],
             e["ply_start"], e["ply_end"], e["exchange_text"], created_at]
            for e in exchanges
        ]
        self._get_cursor().executemany("""
            INSERT INTO exchanges (
                exchange_id, conversation_id, project_id, ply_start, ply_end,
                exchange_text, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (conversation_id, ply_start, ply_end) DO UPDATE SET
                exchange_text = EXCLUDED.exchange_text,
                created_at = EXCLUDED.created_at
        """, data)

    def store_verbatim_embeddings_batch(
        self,
        embeddings: List[Tuple],
    ) -> None:
        """Store multiple verbatim embeddings in a single executemany call.

        Args:
            embeddings: List of (exchange_id, embedding_array) tuples.
        """
        if not embeddings:
            return
        data = [
            [eid, emb.tolist()]
            for eid, emb in embeddings
        ]
        self._get_cursor().executemany("""
            INSERT INTO verbatim_embeddings (exchange_id, embedding)
            VALUES (?, ?::FLOAT[])
            ON CONFLICT (exchange_id) DO UPDATE SET
                embedding = EXCLUDED.embedding
        """, data)

    def get_exchange_ids_in_set(self, exchange_ids: Set[str]) -> Set[str]:
        """Return the subset of exchange_ids that exist in the exchanges table."""
        if not exchange_ids:
            return set()
        placeholders = ",".join("?" for _ in exchange_ids)
        rows = self._get_read_cursor().execute(f"""
            SELECT exchange_id FROM exchanges WHERE exchange_id IN ({placeholders})
        """, list(exchange_ids)).fetchall()
        return {r[0] for r in rows}

    def get_existing_exchange_keys(
        self, conversation_ids: Optional[List[str]] = None,
    ) -> Set[Tuple[str, int, int]]:
        """Get existing (conversation_id, ply_start, ply_end) tuples.

        Args:
            conversation_ids: If provided, only return keys for these conversations.
                Returns empty set immediately if the list is empty.
                If None, returns all exchange keys (full table scan).
        """
        if conversation_ids is not None:
            if not conversation_ids:
                return set()
            placeholders = ",".join("?" for _ in conversation_ids)
            rows = self._get_read_cursor().execute(f"""
                SELECT conversation_id, ply_start, ply_end FROM exchanges
                WHERE conversation_id IN ({placeholders})
            """, conversation_ids).fetchall()
        else:
            rows = self._get_read_cursor().execute("""
                SELECT conversation_id, ply_start, ply_end FROM exchanges
            """).fetchall()
        return {(r[0], r[1], r[2]) for r in rows}

    def get_exchange(self, exchange_id: str) -> Optional[Dict]:
        """Get an exchange by ID."""
        row = self._get_read_cursor().execute("""
            SELECT exchange_id, conversation_id, project_id, ply_start, ply_end,
                   exchange_text, created_at
            FROM exchanges
            WHERE exchange_id = ?
        """, [exchange_id]).fetchone()

        if row is None:
            return None

        return {
            "exchange_id": row[0],
            "conversation_id": row[1],
            "project_id": row[2],
            "ply_start": row[3],
            "ply_end": row[4],
            "exchange_text": row[5],
            "created_at": row[6],
        }

    def get_exchange_by_ply(
        self, conversation_id: str, ply_start: int, ply_end: int
    ) -> Optional[Dict]:
        """Get an exchange by conversation and ply range."""
        row = self._get_read_cursor().execute("""
            SELECT exchange_id, conversation_id, project_id, ply_start, ply_end,
                   exchange_text, created_at
            FROM exchanges
            WHERE conversation_id = ? AND ply_start = ? AND ply_end = ?
        """, [conversation_id, ply_start, ply_end]).fetchone()

        if row is None:
            return None

        return {
            "exchange_id": row[0],
            "conversation_id": row[1],
            "project_id": row[2],
            "ply_start": row[3],
            "ply_end": row[4],
            "exchange_text": row[5],
            "created_at": row[6],
        }

    def get_conversation_exchanges(self, conversation_id: str) -> List[Dict]:
        """Get all exchanges for a conversation ordered by ply_start."""
        rows = self._get_read_cursor().execute("""
            SELECT exchange_id, conversation_id, project_id, ply_start, ply_end,
                   exchange_text, created_at
            FROM exchanges
            WHERE conversation_id = ?
            ORDER BY ply_start ASC, ply_end ASC
        """, [conversation_id]).fetchall()

        return [
            {
                "exchange_id": row[0],
                "conversation_id": row[1],
                "project_id": row[2],
                "ply_start": row[3],
                "ply_end": row[4],
                "exchange_text": row[5],
                "created_at": row[6],
            }
            for row in rows
        ]

    # =========================================================================
    # Vector Search (VSS)
    # =========================================================================

    def search_verbatim_semantic(
        self,
        query_embedding: np.ndarray,
        limit: int = 50,
        project_ids: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Semantic search over verbatim embeddings using HNSW index.

        Args:
            query_embedding: Query vector (EMBEDDING_DIM dimensions)
            limit: Maximum results to return
            project_ids: Optional list of project IDs to filter

        Returns:
            List of dicts with exchange info and distance scores
        """
        if not self._vss_available:
            logger.warning("VSS not available, falling back to brute-force search")
            return self._search_verbatim_brute_force(query_embedding, limit, project_ids)

        embedding_list = query_embedding.tolist()

        if project_ids:
            project_filter = "AND e.project_id IN (SELECT UNNEST(?::VARCHAR[]))"
        else:
            project_filter = ""

        params = [embedding_list]
        if project_ids:
            params.append(project_ids)
        params.append(limit)

        rows = self._get_read_cursor().execute(f"""
            SELECT
                e.exchange_id,
                e.conversation_id,
                e.project_id,
                e.ply_start,
                e.ply_end,
                e.exchange_text,
                c.title,
                c.file_path,
                c.message_count,
                c.created_at,
                c.updated_at,
                array_cosine_distance(ve.embedding, ?::FLOAT[{EMBEDDING_DIM}]) AS distance
            FROM verbatim_embeddings ve
            JOIN exchanges e ON ve.exchange_id = e.exchange_id
            JOIN conversations c ON e.conversation_id = c.conversation_id
            WHERE 1=1 {project_filter}
            ORDER BY distance ASC
            LIMIT ?
        """, params).fetchall()

        return [
            {
                "exchange_id": r[0],
                "conversation_id": r[1],
                "project_id": r[2],
                "ply_start": r[3],
                "ply_end": r[4],
                "exchange_text": r[5],
                "title": r[6],
                "file_path": r[7],
                "message_count": r[8],
                "created_at": r[9],
                "updated_at": r[10],
                "distance": r[11],
                "score": 1.0 / (1.0 + r[11]),  # Convert distance to similarity
            }
            for r in rows
        ]

    def _search_verbatim_brute_force(
        self,
        query_embedding: np.ndarray,
        limit: int = 50,
        project_ids: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Fallback brute-force search when VSS is not available."""
        embedding_list = query_embedding.tolist()

        if project_ids:
            project_filter = "AND e.project_id IN (SELECT UNNEST(?::VARCHAR[]))"
        else:
            project_filter = ""

        # Use list_cosine_distance for brute-force search
        params = [embedding_list]
        if project_ids:
            params.append(project_ids)
        params.append(limit)

        rows = self._get_read_cursor().execute(f"""
            SELECT
                e.exchange_id,
                e.conversation_id,
                e.project_id,
                e.ply_start,
                e.ply_end,
                e.exchange_text,
                c.title,
                c.file_path,
                c.message_count,
                c.created_at,
                c.updated_at,
                list_cosine_distance(ve.embedding, ?::FLOAT[{EMBEDDING_DIM}]) AS distance
            FROM verbatim_embeddings ve
            JOIN exchanges e ON ve.exchange_id = e.exchange_id
            JOIN conversations c ON e.conversation_id = c.conversation_id
            WHERE 1=1 {project_filter}
            ORDER BY distance ASC
            LIMIT ?
        """, params).fetchall()

        return [
            {
                "exchange_id": r[0],
                "conversation_id": r[1],
                "project_id": r[2],
                "ply_start": r[3],
                "ply_end": r[4],
                "exchange_text": r[5],
                "title": r[6],
                "file_path": r[7],
                "message_count": r[8],
                "created_at": r[9],
                "updated_at": r[10],
                "distance": r[11],
                "score": 1.0 / (1.0 + r[11]),
            }
            for r in rows
        ]

    def search_palace_semantic(
        self,
        query_embedding: np.ndarray,
        limit: int = 50,
        project_ids: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Semantic search over palace objects using HNSW index.

        Searches on distilled_text embeddings but returns exchange_text
        for consistent evaluation against verbatim search.
        """
        if not self._vss_available:
            return self._search_palace_brute_force(query_embedding, limit, project_ids)

        embedding_list = query_embedding.tolist()

        if project_ids:
            project_filter = "AND po.project_id IN (SELECT UNNEST(?::VARCHAR[]))"
        else:
            project_filter = ""

        params = [embedding_list]
        if project_ids:
            params.append(project_ids)
        params.append(limit)

        rows = self._get_read_cursor().execute(f"""
            SELECT
                po.object_id,
                po.exchange_id,
                po.conversation_id,
                po.project_id,
                po.ply_start,
                po.ply_end,
                po.files_touched,
                po.exchange_core,
                po.specific_context,
                po.distilled_text,
                po.created_at,
                po.exchange_at,
                array_cosine_distance(po.embedding, ?::FLOAT[{EMBEDDING_DIM}]) AS distance,
                e.exchange_text
            FROM palace_objects po
            LEFT JOIN exchanges e ON po.exchange_id = e.exchange_id
            WHERE po.embedding IS NOT NULL {project_filter}
            ORDER BY distance ASC
            LIMIT ?
        """, params).fetchall()

        return [self._palace_row_to_dict(r) for r in rows]

    def _search_palace_brute_force(
        self,
        query_embedding: np.ndarray,
        limit: int = 50,
        project_ids: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Fallback brute-force search for palace objects."""
        embedding_list = query_embedding.tolist()

        if project_ids:
            project_filter = "AND po.project_id IN (SELECT UNNEST(?::VARCHAR[]))"
        else:
            project_filter = ""

        params = [embedding_list]
        if project_ids:
            params.append(project_ids)
        params.append(limit)

        rows = self._get_read_cursor().execute(f"""
            SELECT
                po.object_id,
                po.exchange_id,
                po.conversation_id,
                po.project_id,
                po.ply_start,
                po.ply_end,
                po.files_touched,
                po.exchange_core,
                po.specific_context,
                po.distilled_text,
                po.created_at,
                po.exchange_at,
                list_cosine_distance(po.embedding, ?::FLOAT[{EMBEDDING_DIM}]) AS distance,
                e.exchange_text
            FROM palace_objects po
            LEFT JOIN exchanges e ON po.exchange_id = e.exchange_id
            WHERE po.embedding IS NOT NULL {project_filter}
            ORDER BY distance ASC
            LIMIT ?
        """, params).fetchall()

        return [self._palace_row_to_dict(r) for r in rows]

    def _palace_row_to_dict(self, row: tuple) -> Dict:
        """Convert a palace object row to a dict.

        Row format: object_id, exchange_id, conversation_id, project_id,
        ply_start, ply_end, files_touched, exchange_core, specific_context,
        distilled_text, created_at, exchange_at, distance, exchange_text
        """
        ft_raw = row[6]
        if isinstance(ft_raw, str):
            ft_data = json.loads(ft_raw)
        elif ft_raw is None:
            ft_data = []
        else:
            ft_data = ft_raw

        result = {
            "object_id": row[0],
            "exchange_id": row[1],
            "conversation_id": row[2],
            "project_id": row[3],
            "ply_start": row[4],
            "ply_end": row[5],
            "files_touched": ft_data,
            "exchange_core": row[7],
            "specific_context": row[8],
            "distilled_text": row[9],
            "created_at": row[10],
            "exchange_at": row[11],
            "distance": row[12],
            "score": 1.0 / (1.0 + row[12]),
        }
        # exchange_text from JOIN (index 13) - for consistent evaluation
        if len(row) > 13:
            result["exchange_text"] = row[13]
        return result

    # =========================================================================
    # Full-Text Search (FTS)
    # =========================================================================

    def search_verbatim_bm25(
        self,
        query: str,
        limit: int = 50,
        project_ids: Optional[List[str]] = None,
    ) -> List[Dict]:
        """BM25 keyword search over exchange text.

        Args:
            query: Search query string
            limit: Maximum results to return
            project_ids: Optional list of project IDs to filter

        Returns:
            List of dicts with exchange info and BM25 scores
        """
        if not self._fts_available:
            logger.warning("FTS not available, falling back to LIKE search")
            return self._search_verbatim_like(query, limit, project_ids)

        if project_ids:
            project_filter = "AND e.project_id IN (SELECT UNNEST(?::VARCHAR[]))"
        else:
            project_filter = ""

        try:
            params = [query]
            if project_ids:
                params.append(project_ids)
            params.append(limit)

            rows = self._get_read_cursor().execute(f"""
                SELECT
                    e.exchange_id,
                    e.conversation_id,
                    e.project_id,
                    e.ply_start,
                    e.ply_end,
                    e.exchange_text,
                    c.title,
                    c.file_path,
                    c.message_count,
                    c.created_at,
                    c.updated_at,
                    fts_main_exchanges.match_bm25(e.exchange_id, ?) AS score
                FROM exchanges e
                JOIN conversations c ON e.conversation_id = c.conversation_id
                WHERE score IS NOT NULL {project_filter}
                ORDER BY score DESC
                LIMIT ?
            """, params).fetchall()
        except Exception as e:
            logger.warning("FTS search failed, falling back to LIKE: %s", e)
            return self._search_verbatim_like(query, limit, project_ids)

        return [
            {
                "exchange_id": r[0],
                "conversation_id": r[1],
                "project_id": r[2],
                "ply_start": r[3],
                "ply_end": r[4],
                "exchange_text": r[5],
                "title": r[6],
                "file_path": r[7],
                "message_count": r[8],
                "created_at": r[9],
                "updated_at": r[10],
                "score": r[11],
            }
            for r in rows
        ]

    def _search_verbatim_like(
        self,
        query: str,
        limit: int = 50,
        project_ids: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Fallback LIKE search when FTS is not available."""
        if project_ids:
            project_filter = "AND e.project_id IN (SELECT UNNEST(?::VARCHAR[]))"
        else:
            project_filter = ""

        # Split query into terms for LIKE matching
        terms = query.lower().split()
        if not terms:
            return []

        # Build LIKE conditions for each term
        like_conditions = " AND ".join(
            f"LOWER(e.exchange_text) LIKE '%{term}%'" for term in terms
        )

        params = []
        if project_ids:
            params.append(project_ids)
        params.append(limit)

        rows = self._get_read_cursor().execute(f"""
            SELECT
                e.exchange_id,
                e.conversation_id,
                e.project_id,
                e.ply_start,
                e.ply_end,
                e.exchange_text,
                c.title,
                c.file_path,
                c.message_count,
                c.created_at,
                c.updated_at,
                1.0 AS score
            FROM exchanges e
            JOIN conversations c ON e.conversation_id = c.conversation_id
            WHERE {like_conditions} {project_filter}
            LIMIT ?
        """, params).fetchall()

        return [
            {
                "exchange_id": r[0],
                "conversation_id": r[1],
                "project_id": r[2],
                "ply_start": r[3],
                "ply_end": r[4],
                "exchange_text": r[5],
                "title": r[6],
                "file_path": r[7],
                "message_count": r[8],
                "created_at": r[9],
                "updated_at": r[10],
                "score": r[11],
            }
            for r in rows
        ]

    def search_palace_bm25(
        self,
        query: str,
        limit: int = 50,
        project_ids: Optional[List[str]] = None,
    ) -> List[Dict]:
        """BM25 keyword search over palace distilled_text.

        Searches on distilled summaries but returns exchange_text
        for consistent evaluation against verbatim search.

        Args:
            query: Search query string
            limit: Maximum results to return
            project_ids: Optional list of project IDs to filter

        Returns:
            List of dicts with palace object info, BM25 scores, and exchange_text
        """
        if not self._fts_available:
            logger.warning("FTS not available, falling back to LIKE search for palace")
            return self._search_palace_like(query, limit, project_ids)

        if project_ids:
            project_filter = "AND po.project_id IN (" + ",".join(f"'{p}'" for p in project_ids) + ")"
        else:
            project_filter = ""

        try:
            rows = self._get_read_cursor().execute(f"""
                SELECT
                    po.object_id,
                    po.exchange_id,
                    po.conversation_id,
                    po.project_id,
                    po.ply_start,
                    po.ply_end,
                    po.files_touched,
                    po.exchange_core,
                    po.specific_context,
                    po.distilled_text,
                    po.created_at,
                    po.exchange_at,
                    fts_main_palace_objects.match_bm25(po.object_id, ?) AS score,
                    e.exchange_text
                FROM palace_objects po
                LEFT JOIN exchanges e ON po.exchange_id = e.exchange_id
                WHERE score IS NOT NULL {project_filter}
                ORDER BY score DESC
                LIMIT ?
            """, [query, limit]).fetchall()
        except Exception as e:
            logger.warning("Palace FTS search failed, falling back to LIKE: %s", e)
            return self._search_palace_like(query, limit, project_ids)

        return [self._palace_bm25_row_to_dict(r) for r in rows]

    def _search_palace_like(
        self,
        query: str,
        limit: int = 50,
        project_ids: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Fallback LIKE search on palace distilled_text."""
        if project_ids:
            project_filter = "AND po.project_id IN (" + ",".join(f"'{p}'" for p in project_ids) + ")"
        else:
            project_filter = ""

        terms = query.lower().split()
        if not terms:
            return []

        like_conditions = " AND ".join(
            f"LOWER(po.distilled_text) LIKE '%{term}%'" for term in terms
        )

        rows = self._get_read_cursor().execute(f"""
            SELECT
                po.object_id,
                po.exchange_id,
                po.conversation_id,
                po.project_id,
                po.ply_start,
                po.ply_end,
                po.files_touched,
                po.exchange_core,
                po.specific_context,
                po.distilled_text,
                po.created_at,
                po.exchange_at,
                1.0 AS score,
                e.exchange_text
            FROM palace_objects po
            LEFT JOIN exchanges e ON po.exchange_id = e.exchange_id
            WHERE {like_conditions} {project_filter}
            LIMIT ?
        """, [limit]).fetchall()

        return [self._palace_bm25_row_to_dict(r) for r in rows]

    def _palace_bm25_row_to_dict(self, row: tuple) -> Dict:
        """Convert a palace BM25 result row to a dict.

        Row format: object_id, exchange_id, conversation_id, project_id,
        ply_start, ply_end, files_touched, exchange_core, specific_context,
        distilled_text, created_at, exchange_at, score, exchange_text
        """
        ft_raw = row[6]
        if isinstance(ft_raw, str):
            ft_data = json.loads(ft_raw)
        elif ft_raw is None:
            ft_data = []
        else:
            ft_data = ft_raw

        return {
            "object_id": row[0],
            "exchange_id": row[1],
            "conversation_id": row[2],
            "project_id": row[3],
            "ply_start": row[4],
            "ply_end": row[5],
            "files_touched": ft_data,
            "exchange_core": row[7],
            "specific_context": row[8],
            "distilled_text": row[9],
            "created_at": row[10],
            "exchange_at": row[11],
            "score": row[12],
            "exchange_text": row[13] if len(row) > 13 else None,
        }

    # =========================================================================
    # Palace Object CRUD
    # =========================================================================

    def store_palace_object(
        self,
        obj: DistilledObject,
        embedding: Optional[np.ndarray] = None,
    ) -> None:
        """Store a palace object with optional embedding."""
        ft_json = json.dumps(
            [{"path": f.path, "action": f.action} for f in obj.files_touched]
        )
        embedding_list = embedding.tolist() if embedding is not None else None

        self._get_cursor().execute("""
            INSERT INTO palace_objects (
                object_id, exchange_id, conversation_id, project_id,
                ply_start, ply_end, files_touched, exchange_core,
                specific_context, distilled_text, embedding,
                created_at, exchange_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?::FLOAT[], ?, ?)
            ON CONFLICT (object_id) DO UPDATE SET
                exchange_id = EXCLUDED.exchange_id,
                files_touched = EXCLUDED.files_touched,
                exchange_core = EXCLUDED.exchange_core,
                specific_context = EXCLUDED.specific_context,
                distilled_text = EXCLUDED.distilled_text,
                embedding = EXCLUDED.embedding
        """, [
            obj.object_id, None, obj.conversation_id, obj.project_id,
            obj.ply_start, obj.ply_end, ft_json, obj.exchange_core,
            obj.specific_context, obj.distilled_text, embedding_list,
            obj.created_at, obj.exchange_at,
        ])

    def get_palace_object(self, object_id: str) -> Optional[DistilledObject]:
        """Get a palace object by ID."""
        row = self._get_read_cursor().execute("""
            SELECT object_id, project_id, conversation_id, ply_start, ply_end,
                   files_touched, exchange_core, specific_context,
                   created_at, exchange_at, distilled_text
            FROM palace_objects
            WHERE object_id = ?
        """, [object_id]).fetchone()

        if row is None:
            return None

        ft_raw = row[5]
        if isinstance(ft_raw, str):
            ft_data = json.loads(ft_raw)
        elif ft_raw is None:
            ft_data = []
        else:
            ft_data = ft_raw

        files = [FileTouched(path=f["path"], action=f["action"]) for f in ft_data]

        return DistilledObject(
            object_id=row[0],
            project_id=row[1],
            conversation_id=row[2],
            ply_start=row[3],
            ply_end=row[4],
            files_touched=files,
            exchange_core=row[6],
            specific_context=row[7],
            created_at=row[8],
            exchange_at=row[9],
            embedding_id=-1,  # Not used in unified storage
            distilled_text=row[10],
        )

    def get_existing_palace_keys(self) -> Set[Tuple[str, int, int]]:
        """Get all existing (conversation_id, ply_start, ply_end) for palace objects."""
        rows = self._get_read_cursor().execute("""
            SELECT conversation_id, ply_start, ply_end FROM palace_objects
        """).fetchall()
        return {(r[0], r[1], r[2]) for r in rows}

    # =========================================================================
    # Room CRUD
    # =========================================================================

    def store_room(self, room: Room) -> None:
        """Store or update a room."""
        self._get_cursor().execute("""
            INSERT INTO rooms (
                room_id, room_type, room_key, room_label, project_id,
                created_at, updated_at, object_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (room_id) DO UPDATE SET
                updated_at = EXCLUDED.updated_at,
                object_count = EXCLUDED.object_count
        """, [
            room.room_id, room.room_type, room.room_key, room.room_label,
            room.project_id, room.created_at, room.updated_at, room.object_count,
        ])

    def store_room_object(self, junction: RoomObject) -> None:
        """Store a room-object junction."""
        room_exists = self._get_cursor().execute(
            "SELECT 1 FROM rooms WHERE room_id = ? LIMIT 1",
            [junction.room_id],
        ).fetchone()
        if room_exists is None:
            raise KeyError(f"Room not found for junction: {junction.room_id}")

        object_exists = self._get_cursor().execute(
            "SELECT 1 FROM palace_objects WHERE object_id = ? LIMIT 1",
            [junction.object_id],
        ).fetchone()
        if object_exists is None:
            raise KeyError(f"Palace object not found for junction: {junction.object_id}")

        self._get_cursor().execute("""
            INSERT INTO room_objects (room_id, object_id, relevance, placed_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (room_id, object_id) DO NOTHING
        """, [junction.room_id, junction.object_id, junction.relevance, junction.placed_at])

    def get_rooms_for_object(self, object_id: str) -> List[Room]:
        """Get all rooms that contain a given object."""
        rows = self._get_read_cursor().execute("""
            SELECT r.room_id, r.room_type, r.room_key, r.room_label,
                   r.project_id, r.created_at, r.updated_at, r.object_count
            FROM room_objects ro
            JOIN rooms r ON ro.room_id = r.room_id
            WHERE ro.object_id = ?
        """, [object_id]).fetchall()

        return [
            Room(
                room_id=r[0],
                room_type=r[1],
                room_key=r[2],
                room_label=r[3],
                project_id=r[4],
                created_at=r[5],
                updated_at=r[6],
                object_count=r[7],
            )
            for r in rows
        ]

    # =========================================================================
    # Facet Embeddings
    # =========================================================================

    def store_facet_embedding(
        self,
        facet_id: str,
        facet_type: str,
        facet_text: str,
        project_ids: List[str],
        embedding: np.ndarray,
        last_seen: Optional[datetime] = None,
    ) -> None:
        """Store or update a facet embedding.

        UPSERT: if facet_id already exists, updates project_ids, embedding, and last_seen.
        """
        embedding_list = embedding.tolist()
        project_ids_json = json.dumps(sorted(project_ids))
        if last_seen is None:
            last_seen = datetime.utcnow()

        self._get_cursor().execute(f"""
            INSERT INTO facet_embeddings (
                facet_id, facet_type, facet_text, project_ids,
                project_count, embedding, last_seen
            ) VALUES (?, ?, ?, ?, ?, ?::FLOAT[{EMBEDDING_DIM}], ?)
            ON CONFLICT (facet_id) DO UPDATE SET
                project_ids = EXCLUDED.project_ids,
                project_count = EXCLUDED.project_count,
                embedding = EXCLUDED.embedding,
                last_seen = EXCLUDED.last_seen
        """, [
            facet_id, facet_type, facet_text, project_ids_json,
            len(project_ids), embedding_list, last_seen,
        ])

    def store_facet_embeddings_batch(
        self,
        facets: List[Dict],
    ) -> int:
        """Store multiple facet embeddings in a transaction.

        Each dict in facets must have: facet_id, facet_type, facet_text,
        project_ids (list), embedding (np.ndarray), and optionally last_seen.

        Returns number of facets stored.
        """
        if not facets:
            return 0

        self._begin_transaction()
        try:
            for f in facets:
                self.store_facet_embedding(
                    facet_id=f["facet_id"],
                    facet_type=f["facet_type"],
                    facet_text=f["facet_text"],
                    project_ids=f["project_ids"],
                    embedding=f["embedding"],
                    last_seen=f.get("last_seen"),
                )
            self._commit()
            return len(facets)
        except Exception:
            self._rollback()
            raise

    def update_facet_meta(
        self,
        facet_id: str,
        project_ids: List[str],
        last_seen: datetime,
    ) -> None:
        """Update project_ids, project_count, and last_seen in one query."""
        project_ids_json = json.dumps(sorted(project_ids))
        self._get_cursor().execute("""
            UPDATE facet_embeddings
            SET project_ids = ?, project_count = ?, last_seen = ?
            WHERE facet_id = ? AND (last_seen IS NULL OR last_seen < ?)
        """, [project_ids_json, len(project_ids), last_seen, facet_id, last_seen])

    def get_facet_project_ids(self, facet_id: str) -> Optional[List[str]]:
        """Get the project_ids for an existing facet, or None if not found."""
        row = self._get_read_cursor().execute("""
            SELECT project_ids FROM facet_embeddings WHERE facet_id = ?
        """, [facet_id]).fetchone()
        if row is None:
            return None
        raw = row[0]
        if isinstance(raw, str):
            return json.loads(raw)
        return raw

    def get_facet_project_ids_batch(self, facet_ids: List[str]) -> Dict[str, List[str]]:
        """Get project_ids for multiple facets in one query.

        Returns {facet_id: project_ids_list} for facets that exist.
        Missing facets are omitted from the result.
        """
        if not facet_ids:
            return {}
        placeholders = ",".join("?" for _ in facet_ids)
        rows = self._get_read_cursor().execute(f"""
            SELECT facet_id, project_ids FROM facet_embeddings
            WHERE facet_id IN ({placeholders})
        """, facet_ids).fetchall()
        result = {}
        for facet_id, raw in rows:
            if isinstance(raw, str):
                result[facet_id] = json.loads(raw)
            else:
                result[facet_id] = raw
        return result

    def search_facet_embeddings(
        self,
        query_embedding: np.ndarray,
        limit: int = 20,
        facet_types: Optional[List[str]] = None,
        max_project_count: Optional[int] = None,
        apply_temporal_decay: bool = False,
        decay_rate: float = 0.01,
    ) -> List[Dict]:
        """Semantic search over facet embeddings using HNSW index.

        Args:
            query_embedding: Query vector (384 dims).
            limit: Maximum results.
            facet_types: Optional filter on facet_type ('file', 'room', 'project').
            max_project_count: Optional distinctiveness filter — only return facets
                appearing in at most this many projects.
            apply_temporal_decay: If True, apply exponential decay to scores based on recency.
            decay_rate: Decay coefficient for exponential decay (default 0.01).

        Returns:
            List of dicts with facet info and distance scores.
        """
        embedding_list = query_embedding.tolist()

        where_clauses = []
        if facet_types:
            type_list = ",".join(f"'{t}'" for t in facet_types)
            where_clauses.append(f"facet_type IN ({type_list})")
        if max_project_count is not None:
            where_clauses.append(f"project_count <= {max_project_count}")

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        distance_fn = "array_cosine_distance" if self._vss_available else "list_cosine_distance"

        rows = self._get_read_cursor().execute(f"""
            SELECT
                facet_id,
                facet_type,
                facet_text,
                project_ids,
                project_count,
                {distance_fn}(embedding, ?::FLOAT[{EMBEDDING_DIM}]) AS distance,
                last_seen
            FROM facet_embeddings
            {where_sql}
            ORDER BY distance ASC
            LIMIT ?
        """, [embedding_list, limit * 2 if apply_temporal_decay else limit]).fetchall()

        results = []
        now = datetime.utcnow()

        for r in rows:
            pids_raw = r[3]
            if isinstance(pids_raw, str):
                pids = json.loads(pids_raw)
            else:
                pids = pids_raw

            base_score = 1.0 / (1.0 + r[5])
            last_seen = r[6]

            # Apply temporal decay if requested
            if apply_temporal_decay and last_seen:
                days_old = (now - last_seen).total_seconds() / 86400.0
                decay_factor = np.exp(-decay_rate * days_old)
                temporal_score = base_score * decay_factor
            else:
                decay_factor = 1.0
                temporal_score = base_score

            results.append({
                "facet_id": r[0],
                "facet_type": r[1],
                "facet_text": r[2],
                "project_ids": pids,
                "project_count": r[4],
                "distance": r[5],
                "score": base_score,
                "temporal_score": temporal_score,
                "decay_factor": decay_factor,
                "last_seen": last_seen,
            })

        # Re-rank by temporal score if decay was applied
        if apply_temporal_decay:
            results = sorted(results, key=lambda x: x["temporal_score"], reverse=True)[:limit]

        return results

    # =========================================================================
    # Hierarchical Facet Operations
    # =========================================================================

    def store_hierarchical_facet(
        self,
        facet_id: str,
        facet_type: str,
        facet_level: str,
        facet_text: str,
        weight: float,
        weighted_count: float,
        project_ids: List[str],
        embedding: np.ndarray,
        last_seen: Optional[datetime] = None,
    ) -> None:
        """Store a single hierarchical facet embedding.

        Args:
            facet_id: Unique facet identifier
            facet_type: 'file', 'room', or 'project'
            facet_level: 'full', 'directory', 'basename', or 'single'
            facet_text: The text that was embedded
            weight: Level weight (3.0 for full, 2.0 for dir, 1.0 for base)
            weighted_count: Distinctiveness score (weight / (1 + project_count))
            project_ids: List of project IDs containing this facet
            embedding: 384-dim embedding vector
            last_seen: Timestamp of last occurrence (defaults to now)
        """
        embedding_list = embedding.tolist()
        project_ids_json = json.dumps(sorted(project_ids))
        last_seen_ts = last_seen or datetime.utcnow()

        self._get_cursor().execute(f"""
            INSERT INTO hierarchical_facets (
                facet_id, facet_type, facet_level, facet_text,
                weight, weighted_count, project_ids, project_count,
                embedding, last_seen
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?::FLOAT[{EMBEDDING_DIM}], ?)
            ON CONFLICT (facet_id) DO UPDATE SET
                weight = EXCLUDED.weight,
                weighted_count = EXCLUDED.weighted_count,
                project_ids = EXCLUDED.project_ids,
                project_count = EXCLUDED.project_count,
                embedding = EXCLUDED.embedding,
                last_seen = EXCLUDED.last_seen
        """, [
            facet_id, facet_type, facet_level, facet_text,
            weight, weighted_count, project_ids_json, len(project_ids),
            embedding_list, last_seen_ts,
        ])

    def store_hierarchical_facets_batch(
        self,
        facets: List[Dict],
    ) -> int:
        """Store multiple hierarchical facet embeddings in a transaction.

        Each dict in facets must have: facet_id, facet_type, facet_level,
        facet_text, weight, weighted_count, project_ids (list), embedding (np.ndarray).

        Returns number of facets stored.
        """
        if not facets:
            return 0

        self._begin_transaction()
        try:
            for f in facets:
                self.store_hierarchical_facet(
                    facet_id=f["facet_id"],
                    facet_type=f["facet_type"],
                    facet_level=f["facet_level"],
                    facet_text=f["facet_text"],
                    weight=f["weight"],
                    weighted_count=f["weighted_count"],
                    project_ids=f["project_ids"],
                    embedding=f["embedding"],
                    last_seen=f.get("last_seen"),
                )
            self._commit()
            return len(facets)
        except Exception:
            self._rollback()
            raise

    def search_hierarchical_facets(
        self,
        query_embedding: np.ndarray,
        limit: int = 20,
        facet_types: Optional[List[str]] = None,
        max_project_count: Optional[int] = None,
        min_weighted_count: Optional[float] = None,
        apply_temporal_decay: bool = False,
        decay_rate: float = 0.01,
    ) -> List[Dict]:
        """Semantic search over hierarchical facet embeddings using HNSW index.

        Args:
            query_embedding: Query vector (384 dims).
            limit: Maximum results.
            facet_types: Optional filter on facet_type ('file', 'room', 'project').
            max_project_count: Optional distinctiveness filter.
            min_weighted_count: Optional minimum weighted distinctiveness score.
            apply_temporal_decay: If True, apply exponential decay based on recency.
            decay_rate: Decay coefficient for exponential decay.

        Returns:
            List of dicts with facet info, including weight, weighted_count, and scores.
        """
        embedding_list = query_embedding.tolist()

        where_clauses = []
        if facet_types:
            type_list = ",".join(f"'{t}'" for t in facet_types)
            where_clauses.append(f"facet_type IN ({type_list})")
        if max_project_count is not None:
            where_clauses.append(f"project_count <= {max_project_count}")
        if min_weighted_count is not None:
            where_clauses.append(f"weighted_count >= {min_weighted_count}")

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        distance_fn = "array_cosine_distance" if self._vss_available else "list_cosine_distance"

        rows = self._get_read_cursor().execute(f"""
            SELECT
                facet_id,
                facet_type,
                facet_level,
                facet_text,
                weight,
                weighted_count,
                project_ids,
                project_count,
                {distance_fn}(embedding, ?::FLOAT[{EMBEDDING_DIM}]) AS distance,
                last_seen
            FROM hierarchical_facets
            {where_sql}
            ORDER BY distance ASC
            LIMIT ?
        """, [embedding_list, limit * 2 if apply_temporal_decay else limit]).fetchall()

        results = []
        now = datetime.utcnow()

        for r in rows:
            pids_raw = r[6]
            if isinstance(pids_raw, str):
                pids = json.loads(pids_raw)
            else:
                pids = pids_raw

            base_score = 1.0 / (1.0 + r[8])
            last_seen = r[9]

            # Apply temporal decay if requested
            if apply_temporal_decay and last_seen:
                days_old = (now - last_seen).total_seconds() / 86400.0
                decay_factor = np.exp(-decay_rate * days_old)
                temporal_score = base_score * decay_factor
            else:
                decay_factor = 1.0
                temporal_score = base_score

            results.append({
                "facet_id": r[0],
                "facet_type": r[1],
                "facet_level": r[2],
                "facet_text": r[3],
                "weight": r[4],
                "weighted_count": r[5],
                "project_ids": pids,
                "project_count": r[7],
                "distance": r[8],
                "score": base_score,
                "temporal_score": temporal_score,
                "decay_factor": decay_factor,
                "last_seen": last_seen,
            })

        # Re-rank by temporal score if decay was applied
        if apply_temporal_decay:
            results = sorted(results, key=lambda x: x["temporal_score"], reverse=True)[:limit]

        return results

    # =========================================================================
    # Batch Operations
    # =========================================================================

    def store_distillation_results(
        self,
        objects: List[DistilledObject],
        rooms: List[Room],
        junctions: List[RoomObject],
        embeddings: Optional[np.ndarray] = None,
        facet_embeddings: Optional[List[Dict]] = None,
    ) -> None:
        """Store distillation results in a single transaction.

        Args:
            objects: Palace objects to store.
            rooms: Rooms to store.
            junctions: Room-object junctions.
            embeddings: Object embeddings (one per object).
            facet_embeddings: Optional list of facet embedding dicts, each with
                facet_id, facet_type, facet_text, project_ids, embedding.
        """
        self._begin_transaction()
        try:
            for i, obj in enumerate(objects):
                embedding = embeddings[i] if embeddings is not None else None
                self.store_palace_object(obj, embedding)

            for room in rooms:
                self.store_room(room)

            for junction in junctions:
                self.store_room_object(junction)

            if facet_embeddings:
                for f in facet_embeddings:
                    self.store_facet_embedding(
                        facet_id=f["facet_id"],
                        facet_type=f["facet_type"],
                        facet_text=f["facet_text"],
                        project_ids=f["project_ids"],
                        embedding=f["embedding"],
                    )

            self._commit()
        except Exception:
            self._rollback()
            raise

    # =========================================================================
    # Conversation Deletion + Internal Prompt Cleanup
    # =========================================================================

    def _delete_exchange_data(
        self,
        conversation_id: str,
        in_transaction: bool = False,
    ) -> None:
        """Delete exchange-level data for a conversation.

        Does NOT delete the conversation or messages rows.
        """
        if not in_transaction:
            self._begin_transaction()
        try:
            cursor = self._get_cursor()
            cursor.execute("""
                DELETE FROM room_objects WHERE object_id IN (
                    SELECT object_id FROM palace_objects WHERE conversation_id = ?
                )
            """, [conversation_id])
            cursor.execute("""
                DELETE FROM palace_objects WHERE conversation_id = ?
            """, [conversation_id])
            cursor.execute("""
                DELETE FROM verbatim_embeddings WHERE exchange_id IN (
                    SELECT exchange_id FROM exchanges WHERE conversation_id = ?
                )
            """, [conversation_id])
            cursor.execute("""
                DELETE FROM exchanges WHERE conversation_id = ?
            """, [conversation_id])
            if not in_transaction:
                self._commit()
        except Exception:
            if not in_transaction:
                self._rollback()
            raise

    def delete_exchange_data_from_ply(
        self,
        conversation_id: str,
        ply_start: int,
        in_transaction: bool = False,
    ) -> None:
        """Delete exchange-layer data for a conversation from a ply boundary onward."""
        if not in_transaction:
            self._begin_transaction()
        try:
            cursor = self._get_cursor()
            cursor.execute("""
                DELETE FROM room_objects WHERE object_id IN (
                    SELECT object_id
                    FROM palace_objects
                    WHERE conversation_id = ? AND ply_start >= ?
                )
            """, [conversation_id, ply_start])
            cursor.execute("""
                DELETE FROM palace_objects
                WHERE conversation_id = ? AND ply_start >= ?
            """, [conversation_id, ply_start])
            cursor.execute("""
                DELETE FROM verbatim_embeddings WHERE exchange_id IN (
                    SELECT exchange_id
                    FROM exchanges
                    WHERE conversation_id = ? AND ply_start >= ?
                )
            """, [conversation_id, ply_start])
            cursor.execute("""
                DELETE FROM exchanges
                WHERE conversation_id = ? AND ply_start >= ?
            """, [conversation_id, ply_start])
            if not in_transaction:
                self._commit()
        except Exception:
            if not in_transaction:
                self._rollback()
            raise

    def delete_conversation(self, conversation_id: str) -> Dict[str, int]:
        """Delete a conversation and all dependent rows in FK dependency order.

        Returns:
            Dict mapping table name to number of rows deleted.
        """
        counts: Dict[str, int] = {}

        self._begin_transaction()
        try:
            row = self._get_cursor().execute("""
                DELETE FROM verbatim_embeddings
                WHERE exchange_id IN (
                    SELECT exchange_id FROM exchanges WHERE conversation_id = ?
                )
                RETURNING 1
            """, [conversation_id]).fetchall()
            counts["verbatim_embeddings"] = len(row)

            row = self._get_cursor().execute("""
                DELETE FROM room_objects
                WHERE object_id IN (
                    SELECT object_id FROM palace_objects WHERE conversation_id = ?
                )
                RETURNING 1
            """, [conversation_id]).fetchall()
            counts["room_objects"] = len(row)

            row = self._get_cursor().execute("""
                DELETE FROM palace_objects WHERE conversation_id = ?
                RETURNING 1
            """, [conversation_id]).fetchall()
            counts["palace_objects"] = len(row)

            row = self._get_cursor().execute("""
                DELETE FROM exchanges WHERE conversation_id = ?
                RETURNING 1
            """, [conversation_id]).fetchall()
            counts["exchanges"] = len(row)

            row = self._get_cursor().execute("""
                DELETE FROM messages WHERE conversation_id = ?
                RETURNING 1
            """, [conversation_id]).fetchall()
            counts["messages"] = len(row)
            row = self._get_cursor().execute("""
                DELETE FROM conversations WHERE conversation_id = ?
                RETURNING 1
            """, [conversation_id]).fetchall()
            counts["conversations"] = len(row)

            self._commit()
        except Exception:
            self._rollback()
            raise

        return counts

    def find_internal_prompt_conversations(
        self, prefixes: tuple = DEFAULT_EXCLUDED_PROMPT_PREFIXES,
    ) -> List[str]:
        """Find conversations that are automated internal prompts.

        Detection checks (any match = contaminated):
        1. First non-empty user message starts with an excluded prefix.
        2. Conversation title starts with an excluded prefix (catches cases
           where the first user message is empty but the title contains
           the compaction/distillation prompt text).

        Args:
            prefixes: Tuple of prompt prefixes to match against. Defaults to
                DEFAULT_EXCLUDED_PROMPT_PREFIXES from constants.py. Callers with
                access to Config should pass config.indexing.excluded_prompt_prefixes.

        Returns:
            List of conversation_ids that are contaminated internal prompts.
        """
        contaminated: Set[str] = set()

        # Check 1: first non-empty user message per conversation
        rows = self._get_read_cursor().execute("""
            SELECT m.conversation_id, m.content, m.sequence
            FROM messages m
            WHERE m.role = 'user'
            ORDER BY m.conversation_id, m.sequence
        """).fetchall()

        checked_convs: Set[str] = set()
        for conv_id, content, _seq in rows:
            if conv_id in checked_convs:
                continue
            text = (content or "").strip()
            if not text:
                continue  # Skip empty — check next user message in this conversation
            checked_convs.add(conv_id)
            if any(text.startswith(p) for p in prefixes):
                contaminated.add(conv_id)

        # Check 2: conversation title matches excluded prefix
        title_rows = self._get_read_cursor().execute("""
            SELECT conversation_id, title FROM conversations
            WHERE title IS NOT NULL AND title != ''
        """).fetchall()

        for conv_id, title in title_rows:
            if conv_id not in contaminated:
                if any(title.startswith(p) for p in prefixes):
                    contaminated.add(conv_id)

        return list(contaminated)

    def cleanup_internal_prompts(self) -> Dict[str, int]:
        """Find and delete all internal prompt conversations from the database.

        Returns:
            Aggregate counts of deleted rows per table, plus 'conversations_deleted' count.
        """
        contaminated_ids = self.find_internal_prompt_conversations()
        if not contaminated_ids:
            return {"conversations_deleted": 0}

        logger.info(
            "Found %d internal prompt conversations to clean up",
            len(contaminated_ids),
        )

        aggregate: Dict[str, int] = {}
        for conv_id in contaminated_ids:
            counts = self.delete_conversation(conv_id)
            for table, n in counts.items():
                aggregate[table] = aggregate.get(table, 0) + n

        aggregate["conversations_deleted"] = len(contaminated_ids)
        return aggregate

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> Dict:
        """Get storage statistics in a single query."""
        row = self._get_read_cursor().execute("""
            SELECT
                (SELECT COUNT(*) FROM conversations),
                (SELECT COUNT(*) FROM messages),
                (SELECT COUNT(*) FROM exchanges),
                (SELECT COUNT(*) FROM verbatim_embeddings),
                (SELECT COUNT(*) FROM palace_objects),
                (SELECT COUNT(*) FROM rooms),
                (SELECT COUNT(*) FROM facet_embeddings),
                (SELECT COUNT(*) FROM hierarchical_facets)
        """).fetchone()

        return {
            "conversations": row[0],
            "messages": row[1],
            "exchanges": row[2],
            "verbatim_embeddings": row[3],
            "palace_objects": row[4],
            "rooms": row[5],
            "facet_embeddings": row[6],
            "hierarchical_facets": row[7],
            "vss_available": self._vss_available,
            "fts_available": self._fts_available,
        }

    # =========================================================================
    # Transaction Helpers
    # =========================================================================

    def _begin_transaction(self) -> None:
        """Begin a transaction on the current thread's cursor.

        Each thread gets its own cursor via _get_cursor(), so no cross-thread
        interference. MVCC handles isolation; appends never conflict.
        """
        cursor = self._get_cursor()
        try:
            cursor.execute("ROLLBACK")
        except Exception:
            pass
        cursor.execute("BEGIN TRANSACTION")

    def _commit(self) -> None:
        """Commit the current transaction on the current thread's cursor."""
        self._get_cursor().execute("COMMIT")

    def _rollback(self) -> None:
        """Rollback the current transaction on the current thread's cursor."""
        try:
            self._get_cursor().execute("ROLLBACK")
        except Exception:
            pass

    def _close_thread_cursor(self) -> None:
        """Close and discard the current thread's cached write cursor.

        Call after background operations (FTS rebuild) to prevent lingering
        implicit transactions from conflicting with subsequent writers.
        Read cursors are ephemeral (created fresh each call) — no cleanup needed.
        """
        if hasattr(self._local, "cursor") and self._local.cursor is not None:
            try:
                self._local.cursor.close()
            except Exception:
                pass
            self._local.cursor = None

    def close(self) -> None:
        """Close the database connection."""
        if not self._external_conn:
            self.conn.close()
