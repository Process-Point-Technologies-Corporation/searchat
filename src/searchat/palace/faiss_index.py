"""FAISS index management for distilled object embeddings."""
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from searchat.config import Config
from searchat.models.schemas import DISTILLED_METADATA_SCHEMA


class DistilledFaissIndex:
    """FAISS IndexFlatL2 for distilled object embeddings."""

    def __init__(self, indices_dir: Path, config: Config):
        self.faiss_path = indices_dir / "distilled.faiss"
        self.metadata_path = indices_dir / "distilled.metadata.parquet"
        self.dimension = 384  # all-MiniLM-L6-v2
        self.index: Optional[faiss.IndexFlatL2] = None
        self._metadata_records: List[dict] = []
        self._vid_to_oid: dict[int, str] = {}

    def load_or_create(self) -> faiss.IndexFlatL2:
        """Load existing index from disk or create a new one."""
        if self.faiss_path.exists():
            self.index = faiss.read_index(str(self.faiss_path))
        else:
            self.index = faiss.IndexFlatL2(self.dimension)

        if self.metadata_path.exists():
            table = pq.read_table(self.metadata_path)
            self._metadata_records = table.to_pydict()
        else:
            self._metadata_records = {
                col: [] for col in DISTILLED_METADATA_SCHEMA.names
            }

        self._rebuild_vid_to_oid()
        return self.index

    def append_vectors(
        self,
        object_ids: List[str],
        project_ids: List[str],
        distilled_texts: List[str],
        embeddings: np.ndarray,
        created_at_values: List,
    ) -> List[int]:
        """Add vectors to the index and write metadata.

        Returns list of assigned vector_ids (embedding_ids).
        """
        if self.index is None:
            self.load_or_create()

        start_id = self.index.ntotal
        self.index.add(embeddings.astype(np.float32))

        vector_ids = list(range(start_id, start_id + len(object_ids)))

        for i, vid in enumerate(vector_ids):
            self._metadata_records["vector_id"].append(vid)
            self._metadata_records["object_id"].append(object_ids[i])
            self._metadata_records["project_id"].append(project_ids[i])
            self._metadata_records["chunk_index"].append(0)
            self._metadata_records["chunk_text"].append(distilled_texts[i])
            self._metadata_records["created_at"].append(created_at_values[i])

        self._rebuild_vid_to_oid()
        self._save()
        return vector_ids

    def search(self, query_embedding: np.ndarray, k: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Search the index. Returns (distances, indices)."""
        if self.index is None:
            self.load_or_create()
        if self.index.ntotal == 0:
            return np.array([[]]), np.array([[]])
        effective_k = min(k, self.index.ntotal)
        query = query_embedding.reshape(1, -1).astype(np.float32)
        distances, indices = self.index.search(query, effective_k)
        return distances, indices

    def get_object_ids_from_vectors(self, vector_ids: List[int]) -> List[str]:
        """Map vector IDs back to object IDs using cached lookup."""
        if not self._vid_to_oid:
            if not self._metadata_records or not self._metadata_records.get("vector_id"):
                if self.metadata_path.exists():
                    table = pq.read_table(self.metadata_path)
                    self._metadata_records = table.to_pydict()
                else:
                    return []
            self._rebuild_vid_to_oid()

        return [self._vid_to_oid[vid] for vid in vector_ids if vid in self._vid_to_oid]

    def _rebuild_vid_to_oid(self) -> None:
        """Rebuild the vector_id → object_id lookup cache."""
        vids = self._metadata_records.get("vector_id", [])
        oids = self._metadata_records.get("object_id", [])
        self._vid_to_oid = dict(zip(vids, oids))

    def _save(self) -> None:
        """Persist index and metadata to disk."""
        self.faiss_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.faiss_path))

        table = pa.table(self._metadata_records, schema=DISTILLED_METADATA_SCHEMA)
        pq.write_table(table, self.metadata_path)
