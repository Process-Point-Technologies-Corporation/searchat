"""In-memory BM25 index for palace objects."""
import json
from typing import Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi

from searchat.palace.storage import PalaceStorage


class PalaceBM25Index:
    """In-memory BM25 index over palace objects for keyword search."""

    def __init__(self):
        self.corpus: List[List[str]] = []  # Tokenized documents
        self.object_ids: List[str] = []  # Parallel list of object_ids
        self.bm25: Optional[BM25Okapi] = None

    def build_from_storage(
        self,
        storage: PalaceStorage,
        include_files: bool = True,
        include_rooms: bool = True,
    ) -> int:
        """Load all objects and rooms, build searchable corpus.

        Args:
            storage: PalaceStorage instance to load from.
            include_files: Include files_touched in index (default True).
            include_rooms: Include room metadata in index (default True).

        Returns number of objects indexed.
        """
        self.corpus = []
        self.object_ids = []

        # Load all objects
        objects = storage.get_all_objects()
        if not objects:
            self.bm25 = None
            return 0

        # Build object_id -> rooms mapping
        object_rooms: Dict[str, List[Tuple[str, str]]] = {}  # object_id -> [(room_key, room_label)]
        rooms = storage.get_all_rooms()
        room_map = {r.room_id: (r.room_key, r.room_label) for r in rooms}

        # Query room_objects junction
        rows = storage.get_room_object_pairs()
        for obj_id, room_id in rows:
            if obj_id not in object_rooms:
                object_rooms[obj_id] = []
            if room_id in room_map:
                object_rooms[obj_id].append(room_map[room_id])

        # Build corpus
        for obj in objects:
            # Core fields (always included)
            text_parts = [
                obj.exchange_core,
                obj.specific_context,
            ]

            # Add conversation title
            if hasattr(obj, 'conv_title') and obj.conv_title:
                text_parts.append(obj.conv_title)

            # Optional: Add file paths
            if include_files:
                for ft in obj.files_touched:
                    text_parts.append(ft.path)

            # Optional: Add room metadata
            if include_rooms:
                for room_key, room_label in object_rooms.get(obj.object_id, []):
                    text_parts.append(room_key)
                    text_parts.append(room_label)

            # Concatenate and tokenize
            full_text = " ".join(text_parts)
            tokens = self._tokenize(full_text)

            self.corpus.append(tokens)
            self.object_ids.append(obj.object_id)

        # Build BM25 index
        if self.corpus:
            self.bm25 = BM25Okapi(self.corpus)

        return len(self.object_ids)

    def search(self, query: str, limit: int = 50) -> List[Tuple[str, float]]:
        """Search index, return (object_id, score) pairs sorted by score descending."""
        if self.bm25 is None or not self.corpus:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        # Pair scores with object_ids and filter out zero scores
        scored = [
            (self.object_ids[i], float(scores[i]))
            for i in range(len(scores))
            if scores[i] > 0
        ]

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored[:limit]

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase and split on whitespace/punctuation."""
        # Lowercase and replace common separators with spaces
        text = text.lower()
        for sep in ["_", "-", "/", "\\", ".", ":"]:
            text = text.replace(sep, " ")
        return text.split()

    @property
    def size(self) -> int:
        """Number of objects in the index."""
        return len(self.object_ids)
