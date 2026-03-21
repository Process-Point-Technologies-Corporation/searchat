"""Tests for DistilledFaissIndex."""
from datetime import datetime

import numpy as np
import pytest

from searchat.config import Config
from searchat.palace.faiss_index import DistilledFaissIndex


@pytest.fixture
def faiss_idx(tmp_path):
    config = Config.load()
    indices_dir = tmp_path / "indices"
    indices_dir.mkdir()
    idx = DistilledFaissIndex(indices_dir, config)
    idx.load_or_create()
    return idx


class TestCreateIndex:
    def test_creates_new_index(self, faiss_idx):
        assert faiss_idx.index is not None
        assert faiss_idx.index.d == 384

    def test_index_starts_empty(self, faiss_idx):
        assert faiss_idx.index.ntotal == 0


class TestAppendVectors:
    def test_append_adds_vectors(self, faiss_idx):
        embeddings = np.random.rand(3, 384).astype(np.float32)
        now = datetime(2026, 1, 15, 10, 0, 0)

        vector_ids = faiss_idx.append_vectors(
            object_ids=["obj-1", "obj-2", "obj-3"],
            project_ids=["proj-1", "proj-1", "proj-1"],
            distilled_texts=["text1", "text2", "text3"],
            embeddings=embeddings,
            created_at_values=[now, now, now],
        )

        assert len(vector_ids) == 3
        assert vector_ids == [0, 1, 2]
        assert faiss_idx.index.ntotal == 3

    def test_append_increments_ids(self, faiss_idx):
        embeddings1 = np.random.rand(2, 384).astype(np.float32)
        embeddings2 = np.random.rand(2, 384).astype(np.float32)
        now = datetime(2026, 1, 15, 10, 0, 0)

        ids1 = faiss_idx.append_vectors(
            object_ids=["obj-1", "obj-2"],
            project_ids=["p", "p"],
            distilled_texts=["t1", "t2"],
            embeddings=embeddings1,
            created_at_values=[now, now],
        )
        ids2 = faiss_idx.append_vectors(
            object_ids=["obj-3", "obj-4"],
            project_ids=["p", "p"],
            distilled_texts=["t3", "t4"],
            embeddings=embeddings2,
            created_at_values=[now, now],
        )

        assert ids1 == [0, 1]
        assert ids2 == [2, 3]


class TestSearch:
    def test_search_returns_results(self, faiss_idx):
        embeddings = np.random.rand(5, 384).astype(np.float32)
        now = datetime(2026, 1, 15, 10, 0, 0)
        faiss_idx.append_vectors(
            object_ids=[f"obj-{i}" for i in range(5)],
            project_ids=["p"] * 5,
            distilled_texts=[f"text-{i}" for i in range(5)],
            embeddings=embeddings,
            created_at_values=[now] * 5,
        )

        query = np.random.rand(384).astype(np.float32)
        distances, indices = faiss_idx.search(query, k=3)

        assert distances.shape[1] == 3
        assert indices.shape[1] == 3

    def test_search_empty_index(self, faiss_idx):
        query = np.random.rand(384).astype(np.float32)
        distances, indices = faiss_idx.search(query, k=5)
        # Empty results
        assert distances.size == 0 or indices.size == 0


class TestMetadataRoundtrip:
    def test_object_id_lookup(self, faiss_idx):
        embeddings = np.random.rand(3, 384).astype(np.float32)
        now = datetime(2026, 1, 15, 10, 0, 0)
        faiss_idx.append_vectors(
            object_ids=["obj-a", "obj-b", "obj-c"],
            project_ids=["p", "p", "p"],
            distilled_texts=["ta", "tb", "tc"],
            embeddings=embeddings,
            created_at_values=[now, now, now],
        )

        oids = faiss_idx.get_object_ids_from_vectors([0, 2])
        assert oids == ["obj-a", "obj-c"]

    def test_missing_vector_id_skipped(self, faiss_idx):
        embeddings = np.random.rand(2, 384).astype(np.float32)
        now = datetime(2026, 1, 15, 10, 0, 0)
        faiss_idx.append_vectors(
            object_ids=["obj-a", "obj-b"],
            project_ids=["p", "p"],
            distilled_texts=["ta", "tb"],
            embeddings=embeddings,
            created_at_values=[now, now],
        )

        oids = faiss_idx.get_object_ids_from_vectors([0, 999])
        assert oids == ["obj-a"]
