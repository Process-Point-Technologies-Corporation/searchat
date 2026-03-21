"""Main distillation engine for the memory palace system."""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

from searchat.config import Config
from searchat.models.domain import (
    DistilledObject,
    DistillationStats,
    FileTouched,
    Room,
    RoomObject,
)
from searchat.palace.faiss_index import DistilledFaissIndex
from searchat.palace.llm import DistillationInput, DistillationLLM, RoomAssignment
from searchat.palace.storage import PalaceStorage

logger = logging.getLogger(__name__)

# Extensions recognized as file paths when matched by regex.
# Ordered roughly by frequency in coding conversations.
_FILE_EXTENSIONS = frozenset({
    "py", "js", "ts", "tsx", "jsx", "json", "jsonl", "toml", "yaml", "yml",
    "md", "html", "css", "scss", "sql", "sh", "bash", "zsh", "cfg", "ini",
    "txt", "xml", "csv", "parquet", "lock", "env", "dockerfile",
    "rs", "go", "java", "c", "h", "cpp", "hpp", "rb", "php", "swift",
    "kt", "scala", "ex", "exs", "erl", "hs", "ml", "r", "jl",
    "vue", "svelte", "astro", "prisma", "graphql", "proto", "tf",
    "conf", "log", "pid", "sock", "wasm", "map",
})

# Regex: path-like string ending in .known_extension
# Matches: src/main.py, tests/test_foo.py, ./config.toml, C:\Users\foo\bar.js
# Requires at least one path separator OR just filename.ext
_PATH_PATTERN = re.compile(
    r'(?:[\w./\\~-]+[/\\])?'    # optional directory prefix (word chars, dots, slashes, tildes, hyphens)
    r'[\w.-]+'                   # filename stem
    r'\.'                        # literal dot
    r'(' + '|'.join(_FILE_EXTENSIONS) + r')'  # known extension
    r'\b',                       # word boundary
    re.IGNORECASE,
)

# Backtick-wrapped content: `some/path.py` or ```code blocks```
_BACKTICK_INLINE = re.compile(r'`([^`\n]+)`')


def extract_file_paths(text: str) -> List[str]:
    """Extract deduplicated file paths from text using regex.

    Scans backtick-wrapped content first (higher signal), then bare text.
    Returns unique paths in order of first appearance, normalized to forward slashes.
    """
    seen: Set[str] = set()
    result: List[str] = []

    def _add(path: str) -> None:
        # Normalize backslashes to forward slashes
        normalized = path.replace("\\", "/")
        # Strip leading ./ prefix
        if normalized.startswith("./"):
            normalized = normalized[2:]
        if normalized not in seen:
            seen.add(normalized)
            result.append(normalized)

    # Phase 1: backtick-wrapped paths (higher confidence)
    for match in _BACKTICK_INLINE.finditer(text):
        content = match.group(1).strip()
        for path_match in _PATH_PATTERN.finditer(content):
            _add(path_match.group(0))

    # Phase 2: bare text paths
    for match in _PATH_PATTERN.finditer(text):
        _add(match.group(0))

    return result


def make_room_id(room_type: str, room_key: str, project_id: Optional[str] = None) -> str:
    """Deterministic room ID from type + key + project."""
    key = f"{room_type}:{room_key}:{project_id or ''}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


class Distiller:
    """Distillation engine for converting conversations into structured palace objects."""

    def __init__(
        self,
        search_dir: Path,
        config: Config,
        llm: Optional[DistillationLLM] = None,
        unified_storage: Optional["UnifiedStorage"] = None,
        embedder: Optional[SentenceTransformer] = None,
        palace_storage: Optional[PalaceStorage] = None,
        indexing_lock: Optional[threading.Lock] = None,
    ):
        self.search_dir = search_dir
        self.config = config
        self.data_dir = search_dir / "data"
        self.storage = palace_storage if palace_storage is not None else PalaceStorage(self.data_dir)
        self.faiss_index = DistilledFaissIndex(self.data_dir / "indices", config)
        self.llm = llm
        self.unified_storage = unified_storage
        self._indexing_lock = indexing_lock or threading.Lock()
        self._distill_lock = threading.Lock()
        if embedder is not None:
            self.embedder = embedder
        else:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(
                config.embedding.model, device=config.embedding.get_device()
            )

    # --- Batch mode (LLM subprocess) ---

    def distill_conversation(self, conversation_id: str) -> List[DistilledObject]:
        """Distill a single conversation using the LLM."""
        if self.llm is None:
            raise RuntimeError("No LLM configured. Use batch mode with CLIDistillationLLM or interactive mode.")

        conv = self._read_conversation(conversation_id)
        if conv is None:
            raise KeyError(f"Conversation not found: {conversation_id}")

        messages = conv["messages"]
        project_id = conv["project_id"]
        exchanges = self._segment_exchanges(messages)

        # No valid exchanges after filtering - mark as skipped
        if not exchanges:
            self.storage.mark_conversation_skipped(
                conversation_id, "no_valid_exchanges"
            )
            return []

        existing_keys = self.storage.get_existing_object_keys(conversation_id)
        inputs = []
        exchange_meta = []

        for ply_start, ply_end in exchanges:
            if (conversation_id, ply_start, ply_end) in existing_keys:
                continue
            exchange_msgs = [
                m for m in messages if ply_start <= m["sequence"] <= ply_end
            ]
            inputs.append(DistillationInput(
                conversation_id=conversation_id,
                project_id=project_id,
                messages=exchange_msgs,
                ply_start=ply_start,
                ply_end=ply_end,
            ))
            exchange_meta.append((ply_start, ply_end, exchange_msgs))

        # All exchanges already processed
        if not inputs:
            return []

        outputs = self.llm.distill(inputs)

        objects = []
        rooms = []
        junctions = []
        now = datetime.utcnow()

        for i, output in enumerate(outputs):
            ply_start, ply_end, exchange_msgs = exchange_meta[i]

            # Determine exchange timestamp from first message
            exchange_at = now
            if exchange_msgs:
                ts = exchange_msgs[0].get("timestamp")
                if isinstance(ts, datetime):
                    exchange_at = ts
                elif isinstance(ts, str):
                    try:
                        exchange_at = datetime.fromisoformat(ts)
                    except (ValueError, TypeError):
                        pass

            # Extract file paths from exchange text via regex (not LLM)
            combined_text = "\n".join(
                m.get("content", "") or "" for m in exchange_msgs
            )
            extracted_paths = extract_file_paths(combined_text)
            files_touched = [
                FileTouched(path=p, action="referenced")
                for p in extracted_paths
            ]

            distilled_text = f"{output.exchange_core}\n{output.specific_context}"
            object_id = str(uuid.uuid4())

            obj = DistilledObject(
                object_id=object_id,
                project_id=project_id,
                conversation_id=conversation_id,
                ply_start=ply_start,
                ply_end=ply_end,
                files_touched=files_touched,
                exchange_core=output.exchange_core,
                specific_context=output.specific_context,
                created_at=now,
                exchange_at=exchange_at,
                embedding_id=-1,  # Set during flush
                distilled_text=distilled_text,
            )
            objects.append(obj)

            # Build rooms and junctions from LLM output
            for ra in output.room_assignments:
                room_id = make_room_id(ra.room_type, ra.room_key, project_id)
                room = Room(
                    room_id=room_id,
                    room_type=ra.room_type,
                    room_key=ra.room_key,
                    room_label=ra.room_label,
                    project_id=project_id,
                    created_at=now,
                    updated_at=now,
                    object_count=1,
                )
                rooms.append(room)
                junctions.append(RoomObject(
                    room_id=room_id,
                    object_id=object_id,
                    relevance=ra.relevance,
                    placed_at=now,
                ))

        self.flush(objects, rooms, junctions)
        return objects

    def distill_all_pending(self, project_id: Optional[str] = None) -> DistillationStats:
        """Distill all conversations not yet fully distilled.

        Uses a threading lock to prevent concurrent runs from startup
        catch-up, watcher callback, and API endpoint.
        """
        if not self._distill_lock.acquire(blocking=False):
            logger.info("Distillation already in progress, skipping")
            return DistillationStats(
                conversations_processed=0,
                objects_created=0,
                rooms_created=0,
                rooms_updated=0,
                distillation_time_seconds=0.0,
            )

        try:
            return self._distill_all_pending_locked(project_id)
        finally:
            self._distill_lock.release()

    def _distill_all_pending_locked(self, project_id: Optional[str] = None) -> DistillationStats:
        """Inner distillation loop, must be called with _distill_lock held."""
        start = time.time()
        conversation_ids = self.list_pending_conversations(project_id)
        # Close the unified_storage cursor after reads to release the read
        # snapshot — a lingering read transaction causes write-write conflicts
        # when the indexer tries to COMMIT on a different cursor.
        if self.unified_storage is not None:
            self.unified_storage._close_thread_cursor()

        total_objects = 0
        total_rooms = 0
        total_rooms_updated = 0
        conversations_processed = 0

        for conv_id in conversation_ids:
            try:
                new_objects = self.distill_conversation(conv_id)
                total_objects += len(new_objects)
                conversations_processed += 1
            except (KeyError, ValueError, AttributeError) as e:
                logger.warning(f"Failed to distill conversation {conv_id}: {e}")
                self.storage.mark_conversation_skipped(conv_id, f"llm_error: {e}")
                continue
            except RuntimeError as e:
                logger.warning(f"Failed to distill conversation {conv_id} (will retry): {e}")
                continue

        elapsed = time.time() - start
        return DistillationStats(
            conversations_processed=conversations_processed,
            objects_created=total_objects,
            rooms_created=total_rooms,
            rooms_updated=total_rooms_updated,
            distillation_time_seconds=elapsed,
        )

    # --- Interactive mode ---

    def store_objects(
        self,
        objects: List[DistilledObject],
        rooms: List[Room],
        junctions: List[RoomObject],
    ) -> None:
        """Store pre-distilled results from the current Claude session."""
        self.flush(objects, rooms, junctions)

    # --- Shared ---

    def flush(
        self,
        objects: List[DistilledObject],
        rooms: List[Room],
        junctions: List[RoomObject],
    ) -> None:
        """Embed objects, assign vector IDs, write to DuckDB + FAISS."""
        if not objects:
            return

        # Embed distilled_texts
        texts = [obj.distilled_text for obj in objects]
        embeddings = self.embedder.encode(texts, batch_size=self.config.embedding.batch_size)
        embeddings = np.array(embeddings, dtype=np.float32)

        # Assign vector IDs via FAISS
        vector_ids = self.faiss_index.append_vectors(
            object_ids=[obj.object_id for obj in objects],
            project_ids=[obj.project_id for obj in objects],
            distilled_texts=texts,
            embeddings=embeddings,
            created_at_values=[obj.created_at for obj in objects],
        )

        # Update embedding_ids on objects
        for i, obj in enumerate(objects):
            obj.embedding_id = vector_ids[i]

        # Write to DuckDB
        self.storage.store_distillation_results(objects, rooms, junctions)

        # Update facet embeddings in unified storage if available.
        # Acquire indexing_lock to serialize against watcher/indexer transactions
        # on the same DuckDB connection (concurrent cursors cause write-write conflicts).
        if self.unified_storage is not None:
            self._indexing_lock.acquire()
            try:
                # Legacy basename-only facets
                facet_dicts = self._compute_facet_embeddings(objects, rooms)
                if facet_dicts:
                    self.unified_storage.store_facet_embeddings_batch(facet_dicts)

                # New hierarchical facets with weighted distinctiveness
                hierarchical_facet_dicts = self._compute_hierarchical_facet_embeddings(objects, rooms)
                if hierarchical_facet_dicts:
                    self.unified_storage.store_hierarchical_facets_batch(hierarchical_facet_dicts)
            finally:
                self._indexing_lock.release()

    def _compute_facet_embeddings(
        self,
        objects: List[DistilledObject],
        rooms: List[Room],
    ) -> List[Dict]:
        """Compute facet embeddings for new facet values from a batch.

        Extracts filenames from files_touched, room keys from rooms, and project
        fragments from project_ids. For each new facet not already in unified
        storage, embeds the text. For existing facets with new project_ids,
        updates project_ids and last_seen timestamp.

        Returns list of facet dicts ready for store_facet_embeddings_batch().
        """
        from searchat.core.unified_storage import make_facet_id

        # Collect all facet candidates from this batch
        # Track last_seen as most recent exchange_at for this facet
        facet_candidates: Dict[str, Dict] = {}  # key -> {facet_type, facet_text, project_ids, last_seen}

        # Filenames
        for obj in objects:
            for ft in obj.files_touched:
                basename = Path(ft.path).name
                if len(basename) > 3:
                    key = f"file:{basename.lower()}"
                    if key not in facet_candidates:
                        facet_candidates[key] = {
                            "facet_type": "file",
                            "facet_text": basename.lower(),
                            "project_ids": set(),
                            "last_seen": obj.exchange_at,
                        }
                    else:
                        # Update last_seen if this object is more recent
                        if obj.exchange_at > facet_candidates[key]["last_seen"]:
                            facet_candidates[key]["last_seen"] = obj.exchange_at
                    facet_candidates[key]["project_ids"].add(obj.project_id)

        # Room keys
        for room in rooms:
            if room.room_key and len(room.room_key) > 3:
                key = f"room:{room.room_key.lower()}"
                if key not in facet_candidates:
                    facet_candidates[key] = {
                        "facet_type": "room",
                        "facet_text": room.room_key.lower(),
                        "project_ids": set(),
                        "last_seen": room.updated_at,
                    }
                else:
                    if room.updated_at > facet_candidates[key]["last_seen"]:
                        facet_candidates[key]["last_seen"] = room.updated_at
                if room.project_id:
                    facet_candidates[key]["project_ids"].add(room.project_id)

        # Project fragments - use most recent object exchange_at per project
        stop_tokens = {
            "projects", "home", "tmp", "mnt", "users", "data",
            "subtask", "workspaces", "benchmark", "var", "opt",
        }
        seen_pids: Set[str] = set()
        for obj in objects:
            if obj.project_id not in seen_pids:
                seen_pids.add(obj.project_id)
                tokens = re.split(r"[-_]+", obj.project_id)
                for token in tokens:
                    t = token.lower()
                    if len(t) > 2 and t not in stop_tokens:
                        key = f"project:{t}"
                        if key not in facet_candidates:
                            facet_candidates[key] = {
                                "facet_type": "project",
                                "facet_text": t,
                                "project_ids": set(),
                                "last_seen": obj.exchange_at,
                            }
                        else:
                            if obj.exchange_at > facet_candidates[key]["last_seen"]:
                                facet_candidates[key]["last_seen"] = obj.exchange_at
                        facet_candidates[key]["project_ids"].add(obj.project_id)

        if not facet_candidates:
            return []

        # Check which facets already exist — single batch query instead of N+1
        facet_id_map = {
            make_facet_id(c["facet_type"], c["facet_text"]): c
            for c in facet_candidates.values()
        }
        existing_facets = self.unified_storage.get_facet_project_ids_batch(
            list(facet_id_map.keys())
        )

        new_facets = []
        for facet_id, candidate in facet_id_map.items():
            existing_pids = existing_facets.get(facet_id)

            if existing_pids is None:
                # New facet — needs embedding
                new_facets.append({
                    "facet_id": facet_id,
                    "facet_type": candidate["facet_type"],
                    "facet_text": candidate["facet_text"],
                    "project_ids": sorted(candidate["project_ids"]),
                    "last_seen": candidate["last_seen"],
                })
            else:
                # Existing — update project_ids and timestamp in one query
                existing_set = set(existing_pids)
                merged = existing_set | candidate["project_ids"]
                self.unified_storage.update_facet_meta(
                    facet_id, sorted(merged), candidate["last_seen"]
                )

        if not new_facets:
            return []

        # Embed new facet texts
        texts = [f["facet_text"] for f in new_facets]
        embeddings = self.embedder.encode(
            texts, batch_size=self.config.embedding.batch_size
        )
        embeddings = np.array(embeddings, dtype=np.float32)

        result = []
        for i, f in enumerate(new_facets):
            f["embedding"] = embeddings[i]
            result.append(f)

        logger.info("Computed %d new facet embeddings from batch", len(result))
        return result

    def _compute_hierarchical_facet_embeddings(
        self,
        objects: List[DistilledObject],
        rooms: List[Room],
    ) -> List[Dict]:
        """Compute hierarchical facet embeddings with weighted distinctiveness.

        Extracts 3 levels of file facets:
        - Full path: palace/storage.py (weight 3x)
        - Directory: palace/* (weight 2x)
        - Basename: storage.py (weight 1x)

        Plus room keys and project fragments as single-level facets.

        Returns list of facet dicts ready for store_hierarchical_facets_batch().
        """
        from searchat.palace.hierarchical_facets import HierarchicalFacetExtractor

        extractor = HierarchicalFacetExtractor()

        # Extract all hierarchical facets from batch
        facet_map = extractor.extract_from_objects(objects, rooms)

        if not facet_map:
            return []

        # Compute embeddings for all new facets
        facet_dicts = extractor.compute_facet_embeddings(
            facet_map,
            self.embedder,
            batch_size=self.config.embedding.batch_size,
        )

        return facet_dicts

    def _segment_exchanges(self, messages: List[dict]) -> List[Tuple[int, int]]:
        """Segment messages into exchanges, dropping empty ones at source.

        An exchange boundary occurs when a user message follows a substantive
        assistant response (non-empty content). Empty assistant messages (e.g.
        tool-use round-trips with stripped content) do not count as responses,
        so the exchange stays open until the assistant actually answers.

        This keeps a user's question and its eventual answer in the same exchange
        even when separated by empty tool-call messages.

        Empty exchanges (below min_exchange_chars) are dropped here so they
        are never produced regardless of call path.
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
                    # Close previous exchange — assistant actually responded
                    exchanges.append((current_start, current_end))
                    current_start = seq
                    current_end = seq
                    has_assistant_content = False
                elif current_start is None:
                    current_start = seq
                    current_end = seq
                else:
                    # No substantive assistant response yet — extend exchange
                    current_end = seq
            else:
                if current_start is None:
                    # Assistant message without a user message — start exchange
                    current_start = seq
                current_end = seq
                if content_len > 0:
                    has_assistant_content = True

        if current_start is not None:
            exchanges.append((current_start, current_end))

        # Drop empty exchanges before ply-length splitting
        min_chars = self.config.distillation.min_exchange_chars
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
            if total_chars < min_chars:
                logger.debug(
                    "Dropping empty exchange plies %d-%d: %d chars < %d minimum",
                    start, end, total_chars, min_chars,
                )
                continue
            if user_chars == 0:
                logger.debug(
                    "Dropping no-user-content exchange plies %d-%d: "
                    "%d total chars but 0 user chars",
                    start, end, total_chars,
                )
                continue
            non_empty.append((start, end))

        # Enforce max_ply_length
        max_ply = self.config.distillation.max_ply_length
        bounded = []
        for start, end in non_empty:
            if end - start + 1 > max_ply:
                # Split into chunks
                for chunk_start in range(start, end + 1, max_ply):
                    chunk_end = min(chunk_start + max_ply - 1, end)
                    bounded.append((chunk_start, chunk_end))
            else:
                bounded.append((start, end))

        return bounded

    def _read_conversation(self, conversation_id: str) -> Optional[dict]:
        """Read a conversation from unified DuckDB storage."""
        if self.unified_storage is None:
            raise RuntimeError(
                "unified_storage is required for _read_conversation. "
                "Pass unified_storage to Distiller constructor."
            )

        conv = self.unified_storage.get_conversation(conversation_id)
        if conv is None:
            self.unified_storage._close_thread_cursor()
            return None

        messages = self.unified_storage.get_conversation_messages(conversation_id)
        # Release read snapshot so it doesn't conflict with indexer writes
        self.unified_storage._close_thread_cursor()

        return {
            "conversation_id": conv["conversation_id"],
            "project_id": conv["project_id"],
            "messages": messages,
        }

    def list_pending_conversations(self, project_id: Optional[str] = None) -> List[str]:
        """List conversation IDs that have undistilled exchanges.

        Excludes:
        - Conversations with distilled objects
        - Conversations marked as skipped (empty/filtered)
        """
        if self.unified_storage is None:
            raise RuntimeError(
                "unified_storage is required for list_pending_conversations. "
                "Pass unified_storage to Distiller constructor."
            )

        all_ids = set(self.unified_storage.get_all_conversation_ids(project_id))
        distilled_ids = self.storage.get_distilled_conversation_ids()
        skipped_ids = self.storage.get_skipped_conversation_ids()

        return sorted(all_ids - distilled_ids - skipped_ids)

    def close(self) -> None:
        self.storage.close()
