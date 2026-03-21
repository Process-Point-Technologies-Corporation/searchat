"""Unified indexer for populating the DuckDB-based unified storage.

This module indexes conversations into the unified DuckDB database:
1. Reads source JSONL/JSON files or existing parquet files
2. Parses conversations and segments into exchanges
3. Generates embeddings and stores in DuckDB with HNSW indexes

Reuses exchange segmentation logic from palace/distiller.py.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import duckdb
import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

from searchat.config import Config
from searchat.agents import detect_provider
from searchat.core.unified_storage import UnifiedStorage
from searchat.models.domain import ConversationRecord, IndexingStats, MessageRecord
from searchat.parquet import get_valid_parquet_files, duckdb_file_list
from searchat.utils.jsonl import load_jsonl_text

logger = logging.getLogger(__name__)


class UnifiedIndexer:
    """Indexes conversations into unified DuckDB storage with exchange-level granularity."""

    def __init__(
        self,
        search_dir: Path,
        config: Optional[Config] = None,
        storage: Optional[UnifiedStorage] = None,
        embedder: Optional[SentenceTransformer] = None,
    ):
        """Initialize the unified indexer.

        Args:
            search_dir: Root directory for searchat data (~/.searchat)
            config: Configuration object (loads default if None)
            storage: Optional pre-existing UnifiedStorage (for testing)
            embedder: Optional shared SentenceTransformer instance
        """
        self.search_dir = search_dir
        self.data_dir = search_dir / "data"
        self.conversations_dir = self.data_dir / "conversations"

        if config is None:
            config = Config.load()
        self.config = config

        if storage is not None:
            self.storage = storage
        else:
            self.storage = UnifiedStorage(self.data_dir)

        # Initialize embedder (use shared instance if provided)
        if embedder is not None:
            self.embedder = embedder
        else:
            from sentence_transformers import SentenceTransformer
            device = config.embedding.get_device()
            logger.info("Initializing embedding model on device: %s", device)
            self.embedder = SentenceTransformer(config.embedding.model, device=device)
        self.batch_size = config.embedding.batch_size
        self.parse_workers = min(8, max(1, os.cpu_count() or 4))

        # Exchange segmentation config
        self.min_exchange_chars = config.distillation.min_exchange_chars
        self.max_ply_length = config.distillation.max_ply_length

    def index_from_parquet(
        self,
        project_id: Optional[str] = None,
        force: bool = False,
    ) -> IndexingStats:
        """Index conversations from existing parquet files into unified storage.

        This is a READ-ONLY operation on parquet files. It:
        1. Reads conversations from parquet files
        2. Segments into exchanges
        3. Generates embeddings
        4. Stores in unified DuckDB

        Args:
            project_id: Optional project ID to filter (indexes all if None)
            force: If True, re-index even if already present

        Returns:
            Dict with indexing statistics
        """
        start_time = time.time()

        if not self.conversations_dir.exists():
            raise FileNotFoundError(
                f"Conversations directory not found: {self.conversations_dir}. "
                "Run the legacy indexer first to create parquet files."
            )

        parquet_files = list(self.conversations_dir.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(
                f"No parquet files found in {self.conversations_dir}. "
                "Run the legacy indexer first."
            )

        # Get existing exchange keys to skip duplicates
        existing_keys = self.storage.get_existing_exchange_keys() if not force else set()

        stats = {
            "conversations_processed": 0,
            "exchanges_created": 0,
            "embeddings_created": 0,
            "skipped_existing": 0,
            "skipped_empty": 0,
            "time_seconds": 0.0,
        }

        # Process each parquet file
        valid_files = get_valid_parquet_files(self.conversations_dir)
        if not valid_files:
            logger.warning("No readable parquet files in %s", self.conversations_dir)
            return stats

        conn = duckdb.connect()
        try:
            file_list = duckdb_file_list(valid_files)

            # Build query with optional project filter
            if project_id:
                query = f"""
                    SELECT conversation_id, project_id, file_path, title,
                           created_at, updated_at, message_count, messages, full_text,
                           file_hash, indexed_at
                    FROM read_parquet({file_list})
                    WHERE project_id = ?
                """
                rows = conn.execute(query, [project_id]).fetchall()
            else:
                query = f"""
                    SELECT conversation_id, project_id, file_path, title,
                           created_at, updated_at, message_count, messages, full_text,
                           file_hash, indexed_at
                    FROM read_parquet({file_list})
                """
                rows = conn.execute(query).fetchall()

            # Process conversations in batches
            stats["skipped_errors"] = 0
            for idx, row in enumerate(rows):
                conv_id = row[0]
                proj_id = row[1]
                messages_raw = row[7]
                title = row[3][:60] if row[3] else "Untitled"

                if idx == 0 or (idx + 1) % max(1, len(rows) // 4) == 0:
                    logger.info(
                        "Indexing progress: %d/%d conversations (%.0f%%)",
                        idx + 1,
                        len(rows),
                        (idx + 1) / len(rows) * 100 if rows else 0
                    )

                try:
                    # Parse messages
                    messages = self._parse_messages(messages_raw)

                    # Wrap all DB writes for this conversation in one transaction.
                    # Embedding generation (CPU) happens outside the transaction.
                    # If anything fails, the whole conversation is rolled back cleanly.

                    # Segment into exchanges (pure computation, no DB)
                    exchanges = self._segment_exchanges(messages)

                    if not exchanges:
                        stats["skipped_empty"] += 1
                        stats["conversations_processed"] += 1
                        continue

                    # Collect exchange data and generate embeddings BEFORE transaction
                    exchange_texts = []
                    exchange_ids = []
                    exchange_meta = []

                    for ply_start, ply_end in exchanges:
                        # Skip if already exists
                        if (conv_id, ply_start, ply_end) in existing_keys:
                            stats["skipped_existing"] += 1
                            continue

                        # Extract exchange text
                        exchange_msgs = [
                            m for m in messages
                            if ply_start <= m["sequence"] <= ply_end
                        ]
                        exchange_text = "\n\n".join(
                            f"{m['role']}: {m['content']}" for m in exchange_msgs
                        )

                        if not exchange_text.strip():
                            continue

                        exchange_id = str(uuid.uuid4())
                        exchange_ids.append(exchange_id)
                        exchange_texts.append(exchange_text)
                        exchange_meta.append({
                            "exchange_id": exchange_id,
                            "conversation_id": conv_id,
                            "project_id": proj_id,
                            "ply_start": ply_start,
                            "ply_end": ply_end,
                        })

                    # Generate embeddings OUTSIDE transaction (CPU-bound)
                    embeddings = None
                    if exchange_texts:
                        logger.debug(
                            "Encoding %d exchanges for conversation %s (total text length: %d chars)",
                            len(exchange_texts),
                            conv_id,
                            sum(len(t) for t in exchange_texts)
                        )
                        embeddings = self.embedder.encode(
                            exchange_texts,
                            batch_size=self.batch_size,
                            show_progress_bar=False,
                            convert_to_numpy=True,
                        )
                        logger.debug("Encoding complete for conversation %s", conv_id)

                    # Single transaction for all DB writes
                    self.storage._begin_transaction()

                    # Store conversation (UPSERT handles conflicts)
                    record = ConversationRecord(
                        conversation_id=conv_id,
                        project_id=proj_id,
                        file_path=row[2],
                        title=row[3],
                        created_at=row[4],
                        updated_at=row[5],
                        message_count=row[6],
                        messages=[
                            MessageRecord(
                                sequence=m["sequence"],
                                role=m["role"],
                                content=m["content"],
                                timestamp=m["timestamp"] or datetime.now(),
                                has_code=m.get("has_code", False),
                                code_blocks=m.get("code_blocks", []),
                            )
                            for m in messages
                        ],
                        full_text=row[8],
                        embedding_id=-1,
                        file_hash=row[9],
                        indexed_at=row[10],
                    )
                    self.storage.store_conversation(record, in_transaction=True)

                    # Store exchanges and embeddings
                    if exchange_texts and embeddings is not None:
                        for i, meta in enumerate(exchange_meta):
                            actual_exchange_id = self.storage.store_exchange(
                                exchange_id=meta["exchange_id"],
                                conversation_id=meta["conversation_id"],
                                project_id=meta["project_id"],
                                ply_start=meta["ply_start"],
                                ply_end=meta["ply_end"],
                                exchange_text=exchange_texts[i],
                                created_at=datetime.utcnow(),
                                skip_existing_check=True,
                            )
                            self.storage.store_verbatim_embedding(
                                exchange_id=actual_exchange_id,
                                embedding=embeddings[i],
                            )

                        stats["exchanges_created"] += len(exchange_meta)
                        stats["embeddings_created"] += len(embeddings)

                    self.storage._commit()
                    stats["conversations_processed"] += 1

                except Exception as e:
                    logger.error(
                        "Failed to index conversation %s ('%s...'): %s",
                        conv_id,
                        title,
                        str(e)
                    )
                    stats["skipped_errors"] += 1
                    self.storage._rollback()
                    continue

        finally:
            conn.close()

        # Create FTS index only if new exchanges were added
        if stats["exchanges_created"] > 0:
            self.storage.create_fts_index()

        stats["time_seconds"] = time.time() - start_time
        return stats

    def _parse_messages(self, messages_raw) -> List[Dict]:
        """Parse messages from parquet struct array format."""
        messages = []
        if messages_raw is None:
            return messages

        for m in messages_raw:
            if isinstance(m, dict):
                messages.append({
                    "sequence": m.get("sequence", 0),
                    "role": m.get("role", ""),
                    "content": m.get("content", ""),
                    "timestamp": m.get("timestamp"),
                    "has_code": m.get("has_code", False),
                    "code_blocks": m.get("code_blocks", []),
                })
            else:
                # Handle DuckDB struct objects
                messages.append({
                    "sequence": getattr(m, "sequence", 0),
                    "role": getattr(m, "role", ""),
                    "content": getattr(m, "content", ""),
                    "timestamp": getattr(m, "timestamp", None),
                    "has_code": getattr(m, "has_code", False),
                    "code_blocks": getattr(m, "code_blocks", []),
                })

        return messages

    def _segment_exchanges(self, messages: List[Dict]) -> List[Tuple[int, int]]:
        """Segment messages into exchanges.

        Reuses logic from palace/distiller.py:_segment_exchanges().
        An exchange boundary occurs when a user message follows a substantive
        assistant response (non-empty content).

        Returns:
            List of (ply_start, ply_end) tuples
        """
        if not messages:
            return []

        sorted_msgs = sorted(messages, key=lambda m: m.get("sequence", 0))

        # Build lookups for content length and role by sequence number
        content_by_seq = {
            m.get("sequence", 0): len(m.get("content", "") or "")
            for m in sorted_msgs
        }
        role_by_seq = {
            m.get("sequence", 0): m.get("role", "")
            for m in sorted_msgs
        }

        exchanges = []
        current_start = None
        current_end = None
        has_assistant_content = False

        for msg in sorted_msgs:
            seq = msg.get("sequence", 0)
            role = msg.get("role", "")
            content_len = content_by_seq.get(seq, 0)

            if role == "user":
                if current_start is not None and has_assistant_content:
                    # Close previous exchange
                    exchanges.append((current_start, current_end))
                    current_start = seq
                    current_end = seq
                    has_assistant_content = False
                elif current_start is None:
                    current_start = seq
                    current_end = seq
                else:
                    current_end = seq
            else:
                if current_start is None:
                    current_start = seq
                current_end = seq
                if content_len > 0:
                    has_assistant_content = True

        if current_start is not None:
            exchanges.append((current_start, current_end))

        # Filter out empty exchanges
        non_empty = []
        for start, end in exchanges:
            total_chars = sum(
                content_by_seq.get(seq, 0)
                for seq in range(start, end + 1)
            )
            user_chars = sum(
                content_by_seq.get(seq, 0)
                for seq in range(start, end + 1)
                if role_by_seq.get(seq) == "user"
            )

            if total_chars < self.min_exchange_chars:
                continue
            if user_chars == 0:
                continue

            non_empty.append((start, end))

        # Enforce max_ply_length
        bounded = []
        for start, end in non_empty:
            if end - start + 1 > self.max_ply_length:
                for chunk_start in range(start, end + 1, self.max_ply_length):
                    chunk_end = min(chunk_start + self.max_ply_length - 1, end)
                    bounded.append((chunk_start, chunk_end))
            else:
                bounded.append((start, end))

        return bounded

    def index_single_conversation(
        self,
        conversation_id: str,
        force: bool = False,
    ) -> IndexingStats:
        """Index a single conversation by ID.

        Args:
            conversation_id: The conversation ID to index
            force: If True, re-index even if already present

        Returns:
            Dict with indexing statistics
        """
        start_time = time.time()

        if not self.conversations_dir.exists():
            raise FileNotFoundError(
                f"Conversations directory not found: {self.conversations_dir}"
            )

        # Get existing exchange keys scoped to this conversation
        existing_keys = self.storage.get_existing_exchange_keys(
            conversation_ids=[conversation_id]
        ) if not force else set()

        stats = {
            "exchanges_created": 0,
            "embeddings_created": 0,
            "skipped_existing": 0,
            "time_seconds": 0.0,
        }

        # Read conversation from parquet
        valid_files = get_valid_parquet_files(self.conversations_dir)
        conn = duckdb.connect()
        try:
            file_list = duckdb_file_list(valid_files)
            row = conn.execute(f"""
                SELECT conversation_id, project_id, file_path, title,
                       created_at, updated_at, message_count, messages, full_text,
                       file_hash, indexed_at
                FROM read_parquet({file_list})
                WHERE conversation_id = ?
                LIMIT 1
            """, [conversation_id]).fetchone()

            if row is None:
                raise KeyError(f"Conversation not found: {conversation_id}")

            conv_id = row[0]
            proj_id = row[1]
            messages_raw = row[7]

            # Parse messages
            messages = self._parse_messages(messages_raw)

            # Segment into exchanges
            exchanges = self._segment_exchanges(messages)

            if not exchanges:
                stats["time_seconds"] = time.time() - start_time
                return stats

            # Collect exchange data and generate embeddings BEFORE transaction
            exchange_texts = []
            exchange_ids = []
            exchange_meta = []

            for ply_start, ply_end in exchanges:
                if (conv_id, ply_start, ply_end) in existing_keys:
                    stats["skipped_existing"] += 1
                    continue

                exchange_msgs = [
                    m for m in messages
                    if ply_start <= m["sequence"] <= ply_end
                ]
                exchange_text = "\n\n".join(
                    f"{m['role']}: {m['content']}" for m in exchange_msgs
                )

                if not exchange_text.strip():
                    continue

                exchange_id = str(uuid.uuid4())
                exchange_ids.append(exchange_id)
                exchange_texts.append(exchange_text)
                exchange_meta.append({
                    "exchange_id": exchange_id,
                    "conversation_id": conv_id,
                    "project_id": proj_id,
                    "ply_start": ply_start,
                    "ply_end": ply_end,
                })

            # Generate embeddings OUTSIDE transaction (CPU-bound)
            embeddings = None
            if exchange_texts:
                embeddings = self.embedder.encode(
                    exchange_texts,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )

            # Single transaction for all DB writes
            self.storage._begin_transaction()

            # Store conversation (UPSERT handles conflicts)
            record = ConversationRecord(
                conversation_id=conv_id,
                project_id=proj_id,
                file_path=row[2],
                title=row[3],
                created_at=row[4],
                updated_at=row[5],
                message_count=row[6],
                messages=[
                    MessageRecord(
                        sequence=m["sequence"],
                        role=m["role"],
                        content=m["content"],
                        timestamp=m["timestamp"] or datetime.now(),
                        has_code=m.get("has_code", False),
                        code_blocks=m.get("code_blocks", []),
                    )
                    for m in messages
                ],
                full_text=row[8],
                embedding_id=-1,
                file_hash=row[9],
                indexed_at=row[10],
            )
            self.storage.store_conversation(record, in_transaction=True)

            if exchange_texts and embeddings is not None:
                for i, meta in enumerate(exchange_meta):
                    self.storage.store_exchange(
                        exchange_id=meta["exchange_id"],
                        conversation_id=meta["conversation_id"],
                        project_id=meta["project_id"],
                        ply_start=meta["ply_start"],
                        ply_end=meta["ply_end"],
                        exchange_text=exchange_texts[i],
                        created_at=datetime.utcnow(),
                    )
                    self.storage.store_verbatim_embedding(
                        exchange_id=meta["exchange_id"],
                        embedding=embeddings[i],
                    )

                stats["exchanges_created"] = len(exchange_meta)
                stats["embeddings_created"] = len(embeddings)

            self.storage._commit()

        finally:
            conn.close()

        stats["time_seconds"] = time.time() - start_time
        return stats

    # =========================================================================
    # Change detection
    # =========================================================================

    def detect_changed_files(
        self, file_paths: List[str],
    ) -> Tuple[List[str], List[str]]:
        """Classify files into (new_files, changed_files).

        2-tier stat-only detection: mtime_ns → size. One stat() per file,
        no file reads, no hashing. ~25ms for 2000 files.

        mtime_ns is the OS-guaranteed write signal. Size confirms content
        actually grew (append-only JSONLs). Same-size rewrites are ignored
        — not a real scenario for append-only conversation files.

        Bootstrap (stored_mtime_ns=0): stamp current stat values into DB
        via bulk UPDATE, skip the file. Fast path activates next run.
        """
        known = self.storage.get_conversation_hashes()  # {conv_id: (hash, path, size, mtime_ns)}
        known_ids = set(known.keys())
        # Secondary index: file_path → conv_id for agents where conversation_id
        # differs from file stem (e.g. Codex: stem is rollout-..., id is UUID).
        known_by_path: Dict[str, str] = {
            data[1]: cid for cid, data in known.items() if data[1]
        }

        new_files: List[str] = []
        changed_files: List[str] = []
        # (file_size, mtime_ns, conversation_id) — stat backfill for migration
        backfill: List[Tuple[int, int, str]] = []

        for fp in file_paths:
            conv_id = Path(fp).stem
            if conv_id not in known_ids:
                # Try file_path lookup (Codex sessions have different stem vs conv_id)
                conv_id = known_by_path.get(fp)
                if conv_id is None:
                    new_files.append(fp)
                    continue

            _stored_hash, _, stored_size, stored_mtime_ns = known[conv_id]
            try:
                st = Path(fp).stat()
            except OSError:
                continue  # File disappeared between scan and check

            if stored_mtime_ns == 0:
                # Bootstrap: no mtime baseline yet. Record current stat
                # and trust the stored hash. Fast path activates next run.
                backfill.append((st.st_size, st.st_mtime_ns, conv_id))
                continue

            # Tier 1: mtime unchanged → no write happened
            if st.st_mtime_ns == stored_mtime_ns:
                continue

            # Tier 2: mtime changed + size changed → content grew/shrank
            if st.st_size != stored_size:
                changed_files.append(fp)
                continue

            # mtime changed but size didn't — metadata touch, backup tool,
            # or antivirus scan. Not a content change for append-only files.

        if backfill:
            self.storage.backfill_stat_columns(backfill)
            logger.info("Backfilled stat columns for %d conversations", len(backfill))

        return new_files, changed_files

    def find_orphaned_conversations(
        self, file_paths: List[str],
    ) -> List[Dict]:
        """Find DB conversations whose source JSONL/JSON no longer exists on disk.

        Compares conversation_ids in DB against file stems on disk.
        Orphaned records are intact in DuckDB but have no source file
        (e.g. JSONL was lost/deleted). These records are still searchable.

        Args:
            file_paths: All source files currently on disk.

        Returns:
            List of dicts with conversation_id, project_id, file_path,
            message_count for each orphaned conversation.
        """
        on_disk_ids = {Path(fp).stem for fp in file_paths}
        rows = self.storage._get_read_cursor().execute("""
            SELECT conversation_id, project_id, file_path, message_count
            FROM conversations
        """).fetchall()
        return [
            {
                "conversation_id": r[0],
                "project_id": r[1],
                "file_path": r[2],
                "message_count": r[3],
            }
            for r in rows
            if r[0] not in on_disk_ids
        ]

    # =========================================================================
    # Direct source file indexing (JSONL/JSON → DuckDB, no parquet intermediate)
    # =========================================================================

    def index_from_source_files(
        self,
        file_paths: List[str],
        changed_file_paths: Optional[List[str]] = None,
    ) -> IndexingStats:
        """Index conversation files directly into DuckDB.

        Reads source files from registered agent providers, parses them,
        segments into exchanges, generates embeddings, and stores in DuckDB.
        No parquet intermediate.

        Args:
            file_paths: List of source file paths to index

        Returns:
            Dict with indexing statistics
        """
        start_time = time.time()

        # Filter by conversation_id (file stem), not file_path.
        # The DB may store old file_paths from parquet-era indexing that
        # don't match the current JSONL scan paths.
        indexed_conv_ids = set(self.storage.get_all_conversation_ids())
        new_files = [
            f for f in file_paths if Path(f).stem not in indexed_conv_ids
        ]
        reindex_files = list(changed_file_paths or [])
        all_files_to_process = new_files + reindex_files

        stats = {
            "new_conversations": 0,
            "updated_conversations": 0,
            "exchanges_created": 0,
            "embeddings_created": 0,
            "skipped_already_indexed": len(file_paths) - len(new_files),
            "skipped_errors": 0,
            "invalid_transcript_count": 0,
            "invalid_transcript_examples": [],
            "skipped_known_invalid": 0,
            "append_only_updates": 0,
            "total_files": len(file_paths) + len(reindex_files),
            "parse_seconds": 0.0,
            "encode_seconds": 0.0,
            "store_seconds": 0.0,
            "time_seconds": 0.0,
        }

        if not all_files_to_process:
            stats["time_seconds"] = time.time() - start_time
            return stats

        logger.info(
            "Indexing %d new + %d changed source files (%d already indexed)",
            len(new_files), len(reindex_files), stats["skipped_already_indexed"],
        )

        # Get existing exchange keys scoped to all conversation IDs being processed
        all_conv_ids = [Path(f).stem for f in all_files_to_process]
        existing_keys = self.storage.get_existing_exchange_keys(conversation_ids=all_conv_ids)

        # ── Phase A: Parse all files (no DB, no embeddings) ──
        parsed_conversations = []
        all_exchange_texts = []
        source_state = self.storage.get_source_file_state(all_files_to_process)
        parse_results: Dict[int, Dict] = {}
        parse_started_at = time.perf_counter()

        max_workers = min(self.parse_workers, max(1, len(all_files_to_process)))
        with ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="SearchatParse",
        ) as executor:
            futures = {
                executor.submit(
                    self._parse_source_candidate,
                    file_path,
                    existing_keys,
                    source_state.get(file_path),
                ): idx
                for idx, file_path in enumerate(all_files_to_process)
            }

            completed = 0
            total = len(all_files_to_process)
            for future in as_completed(futures):
                idx = futures[future]
                parse_results[idx] = future.result()
                completed += 1

                if completed == 1 or completed % max(1, total // 4) == 0:
                    logger.info(
                        "Phase A (parse): %d/%d files (%.0f%%)",
                        completed, total,
                        completed / total * 100,
                    )

        reindex_stems = {Path(f).stem for f in reindex_files}

        for idx in range(len(all_files_to_process)):
            result = parse_results[idx]
            status = result["status"]

            if status == "missing":
                logger.warning("File not found, skipping: %s", result["file_path"])
                stats["skipped_errors"] += 1
                continue
            if status == "unknown":
                logger.debug("Unknown format, skipping: %s", result["file_path"])
                stats["skipped_errors"] += 1
                continue
            if status == "skip_known_invalid":
                logger.debug(
                    "Skipping unchanged invalid transcript %s: %s",
                    result["file_path"],
                    result.get("reason") or "previous parse failure",
                )
                stats["skipped_known_invalid"] += 1
                continue
            if status == "invalid":
                logger.debug(
                    "Skipping malformed transcript %s: %s",
                    result["file_path"],
                    result["reason"],
                )
                stats["skipped_errors"] += 1
                stats["invalid_transcript_count"] += 1
                self.storage.mark_source_file_invalid(
                    file_path=result["file_path"],
                    conversation_id=result["conversation_id"],
                    file_size=result["file_size"],
                    mtime_ns=result["mtime_ns"],
                    error_message=result["reason"],
                )
                if len(stats["invalid_transcript_examples"]) < 3:
                    stats["invalid_transcript_examples"].append(
                        {"file_path": result["file_path"], "reason": result["reason"]}
                    )
                continue
            if status == "error":
                logger.error(
                    "Failed to parse %s: %s\n%s",
                    result["file_path"],
                    result["reason"],
                    result["traceback"],
                )
                stats["skipped_errors"] += 1
                continue
            if status == "empty":
                continue

            exchange_texts = result["exchange_texts"]
            embedding_offset = len(all_exchange_texts)
            all_exchange_texts.extend(exchange_texts)
            parsed_conversations.append({
                "record": result["record"],
                "exchange_meta": result["exchange_meta"],
                "embedding_offset": embedding_offset,
                "delete_from_ply": result.get("delete_from_ply"),
            })
            if result.get("append_only"):
                stats["append_only_updates"] += 1

        stats["parse_seconds"] = time.perf_counter() - parse_started_at

        # ── Phase B: Single encode call across all files ──
        all_embeddings = None
        encode_started_at = time.perf_counter()
        if all_exchange_texts:
            logger.info(
                "Phase B (encode): %d exchange texts in single batch",
                len(all_exchange_texts),
            )
            all_embeddings = self.embedder.encode(
                all_exchange_texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        stats["encode_seconds"] = time.perf_counter() - encode_started_at

        # ── Phase C: Per-conversation transactional writes ──
        # HNSW experimental persistence is disabled at startup, so no
        # internal DuckDB background writer races with COMMIT.
        store_started_at = time.perf_counter()
        for cidx, conv_data in enumerate(parsed_conversations):
            record = conv_data["record"]
            exchange_meta = conv_data["exchange_meta"]
            offset = conv_data["embedding_offset"]
            n = len(exchange_meta)
            delete_from_ply = conv_data.get("delete_from_ply")

            try:
                conv_id = record.conversation_id
                self.storage._begin_transaction()
                # Only delete existing data for re-indexed conversations
                if conv_id in reindex_stems:
                    if delete_from_ply is None:
                        self.storage._delete_exchange_data(conv_id, in_transaction=True)
                    else:
                        self.storage.delete_exchange_data_from_ply(
                            conv_id, delete_from_ply, in_transaction=True,
                        )
                self.storage.store_conversation(record, in_transaction=True)

                if exchange_meta and all_embeddings is not None:
                    created_at = datetime.utcnow()
                    self.storage.store_exchanges_batch(exchange_meta, created_at)

                    embedding_pairs = [
                        (exchange_meta[i]["exchange_id"], all_embeddings[offset + i])
                        for i in range(n)
                    ]
                    self.storage.store_verbatim_embeddings_batch(embedding_pairs)

                    stats["exchanges_created"] += n
                    stats["embeddings_created"] += n

                self.storage._commit()
                self.storage.clear_source_file_state(record.file_path)
                if record.conversation_id in reindex_stems:
                    stats["updated_conversations"] += 1
                else:
                    stats["new_conversations"] += 1

            except Exception as e:
                logger.error(
                    "Failed to store %s: %s", record.conversation_id, e,
                    exc_info=True,
                )
                stats["skipped_errors"] += 1
                self.storage._rollback()
                continue

        stats["store_seconds"] = time.perf_counter() - store_started_at

        stats["time_seconds"] = time.time() - start_time
        logger.info(
            "Indexing timings: parse=%.2fs encode=%.2fs store=%.2fs total=%.2fs "
            "(append_only=%d)",
            stats["parse_seconds"],
            stats["encode_seconds"],
            stats["store_seconds"],
            stats["time_seconds"],
            stats["append_only_updates"],
        )
        return stats

    def _parse_source_candidate(
        self,
        file_path: str,
        existing_keys: Set[Tuple[str, int, int]],
        cached_state: Optional[Dict],
    ) -> IndexingStats:
        """Parse one source file candidate for Phase A in a worker thread."""
        path = Path(file_path)
        try:
            st = path.stat()
        except OSError:
            return {"status": "missing", "file_path": file_path}

        if (
            cached_state
            and cached_state["status"] == "invalid"
            and cached_state["file_size"] == st.st_size
            and cached_state["mtime_ns"] == st.st_mtime_ns
        ):
            return {
                "status": "skip_known_invalid",
                "file_path": file_path,
                "reason": cached_state.get("error_message"),
            }

        provider = detect_provider(path)
        if provider is None:
            return {"status": "unknown", "file_path": file_path}

        if cached_state is None and provider.agent_id in {"claude", "codex"}:
            incremental = self._try_parse_appended_conversation(path, provider.agent_id)
            if incremental is not None:
                return incremental

        if provider.agent_id == "vibe":
            project_id = "vibe-sessions"
        elif provider.agent_id == "codex":
            project_id = "codex-sessions"
        else:
            project_id = path.parent.name

        try:
            record = provider.parse_conversation(path, project_id)
        except ValueError as e:
            return {
                "status": "invalid",
                "file_path": file_path,
                "conversation_id": path.stem,
                "file_size": st.st_size,
                "mtime_ns": st.st_mtime_ns,
                "reason": str(e),
            }
        except Exception as e:
            return {
                "status": "error",
                "file_path": file_path,
                "reason": str(e),
                "traceback": traceback.format_exc(),
            }

        if record.message_count == 0:
            return {"status": "empty", "file_path": file_path}

        messages = [
            {
                "sequence": m.sequence,
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp,
            }
            for m in record.messages
        ]
        exchanges = self._segment_exchanges(messages)

        exchange_texts = []
        exchange_meta = []

        if exchanges:
            for ply_start, ply_end in exchanges:
                if (record.conversation_id, ply_start, ply_end) in existing_keys:
                    continue

                exchange_msgs = [
                    m for m in messages
                    if ply_start <= m["sequence"] <= ply_end
                ]
                exchange_text = "\n\n".join(
                    f"{m['role']}: {m['content']}" for m in exchange_msgs
                )

                if not exchange_text.strip():
                    continue

                exchange_id = str(uuid.uuid4())
                exchange_texts.append(exchange_text)
                exchange_meta.append({
                    "exchange_id": exchange_id,
                    "conversation_id": record.conversation_id,
                    "project_id": project_id,
                    "ply_start": ply_start,
                    "ply_end": ply_end,
                    "exchange_text": exchange_text,
                })

        return {
            "status": "ok",
            "file_path": file_path,
            "record": record,
            "exchange_texts": exchange_texts,
            "exchange_meta": exchange_meta,
        }

    def _try_parse_appended_conversation(self, path: Path, agent_id: str) -> Optional[Dict]:
        """Try an append-only reparse for changed Claude/Codex JSONL transcripts."""
        conversation_id = path.stem
        existing = self.storage.get_conversation(conversation_id)
        if existing is None:
            return None

        stored_size = existing.get("file_size") or 0
        try:
            stat = path.stat()
        except OSError:
            return None

        if stored_size <= 0 or stat.st_size <= stored_size:
            return None

        try:
            with path.open("rb") as handle:
                handle.seek(stored_size)
                tail_bytes = handle.read()
        except OSError:
            return None

        if not tail_bytes.strip():
            return None

        try:
            tail_text = tail_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return None

        load_result = load_jsonl_text(tail_text)
        if load_result.valid_count == 0 and load_result.invalid_count == 0:
            return None
        if load_result.invalid_count > 0:
            return {
                "status": "invalid",
                "file_path": str(path),
                "conversation_id": conversation_id,
                "file_size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "reason": (
                    f"Malformed appended JSONL in {path}; "
                    f"{load_result.describe_issues()}"
                ),
            }

        existing_messages = self.storage.get_conversation_messages(conversation_id)
        appended_messages = self._parse_appended_messages(
            agent_id, load_result.entries, starting_sequence=len(existing_messages),
        )
        if not appended_messages:
            return None

        existing_message_records = [
            MessageRecord(
                sequence=msg["sequence"],
                role=msg["role"],
                content=msg["content"],
                timestamp=msg["timestamp"],
                has_code=msg.get("has_code", False),
                code_blocks=[],
            )
            for msg in existing_messages
        ]
        merged_messages = existing_message_records + appended_messages
        merged_record = ConversationRecord(
            conversation_id=conversation_id,
            project_id=existing["project_id"],
            file_path=str(path),
            title=existing["title"],
            created_at=existing["created_at"],
            updated_at=merged_messages[-1].timestamp,
            message_count=len(merged_messages),
            messages=merged_messages,
            full_text="\n\n".join(msg.content for msg in merged_messages),
            embedding_id=-1,
            file_hash=existing["file_hash"],
            indexed_at=datetime.now(),
            file_size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
        )

        message_dicts = [
            {
                "sequence": msg.sequence,
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp,
            }
            for msg in merged_messages
        ]
        exchanges = self._segment_exchanges(message_dicts)
        existing_exchanges = self.storage.get_conversation_exchanges(conversation_id)
        delete_from_ply = existing_exchanges[-1]["ply_start"] if existing_exchanges else 0
        surviving_keys = {
            (conversation_id, exchange["ply_start"], exchange["ply_end"])
            for exchange in existing_exchanges
            if exchange["ply_start"] < delete_from_ply
        }

        exchange_texts = []
        exchange_meta = []
        for ply_start, ply_end in exchanges:
            if ply_start < delete_from_ply:
                continue
            if (conversation_id, ply_start, ply_end) in surviving_keys:
                continue

            exchange_msgs = [
                m for m in message_dicts
                if ply_start <= m["sequence"] <= ply_end
            ]
            exchange_text = "\n\n".join(
                f"{m['role']}: {m['content']}" for m in exchange_msgs
            )
            if not exchange_text.strip():
                continue

            exchange_texts.append(exchange_text)
            exchange_meta.append({
                "exchange_id": str(uuid.uuid4()),
                "conversation_id": conversation_id,
                "project_id": existing["project_id"],
                "ply_start": ply_start,
                "ply_end": ply_end,
                "exchange_text": exchange_text,
            })

        return {
            "status": "ok",
            "file_path": str(path),
            "record": merged_record,
            "exchange_texts": exchange_texts,
            "exchange_meta": exchange_meta,
            "delete_from_ply": delete_from_ply,
            "append_only": True,
        }

    def _parse_appended_messages(
        self,
        agent_id: str,
        entries: List[Dict],
        starting_sequence: int,
    ) -> List[MessageRecord]:
        """Parse appended JSONL entries for providers with append-only transcripts."""
        if agent_id == "claude":
            return self._parse_claude_appended_messages(entries, starting_sequence)
        if agent_id == "codex":
            return self._parse_codex_appended_messages(entries, starting_sequence)
        return []

    def _parse_claude_appended_messages(
        self,
        entries: List[Dict],
        starting_sequence: int,
    ) -> List[MessageRecord]:
        messages: List[MessageRecord] = []
        next_sequence = starting_sequence
        for entry in entries:
            msg_type = entry.get("type")
            if msg_type not in ("user", "assistant"):
                continue
            raw_content = entry.get("message", {}).get("content", "")
            if isinstance(raw_content, str):
                content = raw_content
            elif isinstance(raw_content, list):
                content = "\n\n".join(
                    block.get("text", "")
                    for block in raw_content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            else:
                content = ""
            timestamp_str = entry.get("timestamp")
            timestamp = datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.now()
            code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", content, re.DOTALL)
            messages.append(
                MessageRecord(
                    sequence=next_sequence,
                    role=msg_type,
                    content=content,
                    timestamp=timestamp,
                    has_code=bool(code_blocks),
                    code_blocks=code_blocks,
                )
            )
            next_sequence += 1
        return messages

    def _parse_codex_appended_messages(
        self,
        entries: List[Dict],
        starting_sequence: int,
    ) -> List[MessageRecord]:
        messages: List[MessageRecord] = []
        next_sequence = starting_sequence
        for entry in entries:
            if entry.get("type") != "response_item":
                continue
            payload = entry.get("payload", {})
            if payload.get("type") != "message":
                continue
            role = payload.get("role")
            if role not in ("user", "assistant"):
                continue
            parts = []
            for block in payload.get("content", []):
                if not isinstance(block, dict):
                    continue
                if block.get("type") in ("input_text", "output_text"):
                    text = block.get("text", "")
                    if text:
                        parts.append(text)
            content = "\n\n".join(parts).strip()
            if not content:
                continue
            timestamp_str = entry.get("timestamp")
            timestamp = (
                datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                if timestamp_str else datetime.now()
            )
            code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", content, re.DOTALL)
            messages.append(
                MessageRecord(
                    sequence=next_sequence,
                    role=role,
                    content=content,
                    timestamp=timestamp,
                    has_code=bool(code_blocks),
                    code_blocks=code_blocks,
                )
            )
            next_sequence += 1
        return messages

    def _detect_agent_format(self, file_path: Path) -> str:
        """Detect conversation file format from registered providers."""
        provider = detect_provider(file_path)
        return provider.agent_id if provider else "unknown"

    def _is_internal_prompt(self, json_path: Path) -> bool:
        """Check if conversation is an automated internal prompt.

        Internal prompts are created by `claude --print` subprocess calls from
        the distillation hook, batch distiller, eval grading pipeline, or
        Claude Code auto-compaction. They should not be indexed as user conversations.

        Prefixes are loaded from config: [indexing] excluded_prompt_prefixes
        (single source of truth in settings.toml).

        Detection strategy:
        1. Check the first non-empty user message against excluded prefixes.
           Skips empty user messages (compaction conversations sometimes have
           an empty first user message followed by the real prompt).
        2. Falls back to False if no user messages match.
        """
        prefixes = self.config.indexing.excluded_prompt_prefixes
        try:
            lines = []
            with open(json_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        lines.append(json.loads(line, strict=False))
                    except json.JSONDecodeError:
                        continue

            for entry in lines:
                if entry.get("type") == "user":
                    raw_content = entry.get("message", {}).get("content", "")
                    if isinstance(raw_content, str):
                        text = raw_content
                    elif isinstance(raw_content, list):
                        text = " ".join(
                            block.get("text", "")
                            for block in raw_content
                            if isinstance(block, dict) and block.get("type") == "text"
                        )
                    else:
                        text = ""
                    # Skip empty user messages — check the next one
                    if not text.strip():
                        continue
                    return any(text.startswith(p) for p in prefixes)

            return False
        except Exception:
            return False

    def _parse_source_file(
        self, file_path: Path, project_id: str, agent_format: str
    ) -> ConversationRecord:
        """Parse a source JSONL or JSON file into a ConversationRecord."""
        provider = detect_provider(file_path)
        if provider is None:
            raise ValueError(f"Unknown conversation format: {file_path}")
        return provider.parse_conversation(file_path, project_id)

    def _parse_claude_conversation(
        self, json_path: Path, project_id: str
    ) -> ConversationRecord:
        """Parse a Claude Code JSONL file."""
        st = json_path.stat()
        raw_bytes = json_path.read_bytes()
        mtime_ns = st.st_mtime_ns

        lines = []
        for line_num, raw_line in enumerate(raw_bytes.decode("utf-8").splitlines(), 1):
            if not raw_line.strip():
                continue
            try:
                lines.append(json.loads(raw_line, strict=False))
            except json.JSONDecodeError:
                logger.debug(
                    "Skipping corrupt line %d in %s", line_num, json_path
                )
        if not lines:
            raise ValueError(f"No valid JSON lines in {json_path}")
        conversation_id = json_path.stem

        # Extract title from first user message
        title = "Untitled"
        for entry in lines:
            content = entry.get("message", {}).get("content", "")
            if isinstance(content, str):
                text = content.strip()
            elif isinstance(content, list):
                text = " ".join(
                    block.get("text", "")
                    for block in content
                    if block.get("type") == "text"
                ).strip()
            else:
                text = ""
            if text:
                title = text[:100]
                break

        messages: List[MessageRecord] = []
        full_text_parts: List[str] = []

        for entry in lines:
            msg_type = entry.get("type")
            if msg_type not in ("user", "assistant"):
                continue

            raw_content = entry.get("message", {}).get("content", "")
            if isinstance(raw_content, str):
                content = raw_content
            elif isinstance(raw_content, list):
                content = "\n\n".join(
                    block.get("text", "")
                    for block in raw_content
                    if block.get("type") == "text"
                )
            else:
                content = ""

            code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", content, re.DOTALL)

            timestamp_str = entry.get("timestamp")
            timestamp = (
                datetime.fromisoformat(timestamp_str)
                if timestamp_str
                else datetime.now()
            )

            messages.append(
                MessageRecord(
                    sequence=len(messages),
                    role=msg_type,
                    content=content,
                    timestamp=timestamp,
                    has_code=len(code_blocks) > 0,
                    code_blocks=code_blocks,
                )
            )
            full_text_parts.append(content)

        created_at = messages[0].timestamp if messages else datetime.now()
        updated_at = messages[-1].timestamp if messages else datetime.now()

        return ConversationRecord(
            conversation_id=conversation_id,
            project_id=project_id,
            file_path=str(json_path),
            title=title,
            created_at=created_at,
            updated_at=updated_at,
            message_count=len(messages),
            messages=messages,
            full_text="\n\n".join(full_text_parts),
            embedding_id=-1,
            file_hash="",
            indexed_at=datetime.now(),
            file_size=len(raw_bytes),
            mtime_ns=mtime_ns,
        )

    def _parse_vibe_session(self, json_path: Path) -> ConversationRecord:
        """Parse a Mistral Vibe session JSON file."""
        st = json_path.stat()
        raw_bytes = json_path.read_bytes()
        mtime_ns = st.st_mtime_ns
        data = json.loads(raw_bytes.decode("utf-8"))
        metadata = data.get("metadata", {})
        session_id = metadata.get("session_id", json_path.stem)

        env = metadata.get("environment", {})
        working_dir = env.get("working_directory", "")
        project_id = Path(working_dir).name if working_dir else "vibe-session"

        start_time_str = metadata.get("start_time")
        end_time_str = metadata.get("end_time")
        created_at = (
            datetime.fromisoformat(start_time_str) if start_time_str else datetime.now()
        )
        updated_at = (
            datetime.fromisoformat(end_time_str) if end_time_str else created_at
        )

        messages: List[MessageRecord] = []
        full_text_parts: List[str] = []
        title = "Untitled Vibe Session"

        for msg in data.get("messages", []):
            role = msg.get("role")
            if role not in ("user", "assistant"):
                continue

            content = msg.get("content", "")
            if not content:
                continue

            if role == "user" and title == "Untitled Vibe Session":
                title = content[:100].replace("\n", " ").strip()

            code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", content, re.DOTALL)

            messages.append(
                MessageRecord(
                    sequence=len(messages),
                    role=role,
                    content=content,
                    timestamp=created_at,
                    has_code=len(code_blocks) > 0,
                    code_blocks=code_blocks,
                )
            )
            full_text_parts.append(content)

        return ConversationRecord(
            conversation_id=session_id,
            project_id=f"vibe-{project_id}",
            file_path=str(json_path),
            title=title,
            created_at=created_at,
            updated_at=updated_at,
            message_count=len(messages),
            messages=messages,
            full_text="\n\n".join(full_text_parts),
            embedding_id=-1,
            file_hash="",
            indexed_at=datetime.now(),
            file_size=len(raw_bytes),
            mtime_ns=mtime_ns,
        )

    def close(self) -> None:
        """Close the storage connection."""
        self.storage.close()
