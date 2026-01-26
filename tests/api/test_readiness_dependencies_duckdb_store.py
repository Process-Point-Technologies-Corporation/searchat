import pytest
from pathlib import Path
from unittest.mock import Mock, patch


@pytest.mark.unit
def test_readiness_state_transitions():
    from searchat.api.readiness import get_readiness, warming_payload, error_payload

    readiness = get_readiness()

    # Warmup start is single-shot
    readiness.mark_warmup_started()
    assert readiness.mark_warmup_started() is False

    readiness.set_component("faiss", "loading")
    readiness.set_component("faiss", "error", error="boom")

    warm = warming_payload(retry_after_ms=123)
    assert warm["status"] == "warming"
    assert warm["retry_after_ms"] == 123
    assert warm["components"]["faiss"] == "error"
    assert warm["errors"]["faiss"] == "boom"

    err = error_payload()
    assert err["status"] == "error"
    assert err["errors"]["faiss"] == "boom"


@pytest.mark.unit
def test_invalidate_search_index_marks_semantic_stale():
    import searchat.api.dependencies as deps

    engine = Mock()
    with patch.object(deps, "start_background_warmup") as start_warmup:
        deps._search_engine = engine
        deps.projects_cache = ["x"]
        deps.stats_cache = {"y": 1}

        # Pretend semantic components were ready
        readiness = deps.get_readiness()
        readiness.set_component("faiss", "ready")
        readiness.set_component("metadata", "ready")
        readiness.set_component("embedder", "ready")

        deps.invalidate_search_index()

        assert deps.projects_cache is None
        assert deps.stats_cache is None
        engine.refresh_index.assert_called_once()
        start_warmup.assert_called_once()

        snap = readiness.snapshot()
        assert snap.components["faiss"] == "idle"
        assert snap.components["metadata"] == "idle"
        assert snap.components["embedder"] == "idle"


@pytest.mark.unit
def test_warmup_semantic_components_sets_readiness_ready(tmp_path: Path):
    import searchat.api.dependencies as deps

    # Ensure services look initialized for the warmup guard.
    deps._config = Mock()
    deps._search_dir = tmp_path

    engine = Mock()
    engine.ensure_metadata_ready = Mock()
    engine.ensure_faiss_loaded = Mock()
    engine.ensure_embedder_loaded = Mock()

    with patch.object(deps, "_ensure_search_engine", return_value=engine):
        readiness = deps.get_readiness()
        readiness.set_component("faiss", "idle")
        readiness.set_component("metadata", "idle")
        readiness.set_component("embedder", "idle")

        deps._warmup_semantic_components()

        engine.ensure_metadata_ready.assert_called_once()
        engine.ensure_faiss_loaded.assert_called_once()
        engine.ensure_embedder_loaded.assert_called_once()
        snap = readiness.snapshot()
        assert snap.components["faiss"] == "ready"
        assert snap.components["metadata"] == "ready"
        assert snap.components["embedder"] == "ready"


@pytest.mark.unit
def test_duckdb_store_queries_use_parameterized_parquet_scan(tmp_path: Path):
    from searchat.api.duckdb_store import DuckDBStore

    search_dir = tmp_path / ".searchat"
    conversations_dir = search_dir / "data" / "conversations"
    conversations_dir.mkdir(parents=True)
    (conversations_dir / "project_test.parquet").write_bytes(b"")

    store = DuckDBStore(search_dir, memory_limit_mb=64)

    con = Mock()
    cursor = Mock()
    con.execute.return_value = cursor
    cursor.fetchall.return_value = [("p1",), ("p2",)]

    with patch('duckdb.connect', return_value=con):
        projects = store.list_projects()

    assert projects == ["p1", "p2"]
    sql = con.execute.call_args[0][0]
    assert "parquet_scan(?)" in sql


@pytest.mark.unit
def test_duckdb_store_get_statistics_and_meta(tmp_path: Path):
    from searchat.api.duckdb_store import DuckDBStore

    search_dir = tmp_path / ".searchat"
    conversations_dir = search_dir / "data" / "conversations"
    conversations_dir.mkdir(parents=True)
    (conversations_dir / "project_test.parquet").write_bytes(b"")

    store = DuckDBStore(search_dir)

    con = Mock()
    cursor = Mock()
    con.execute.return_value = cursor
    cursor.fetchone.return_value = (1, 10, 10.0, 1, "2025-01-01", "2025-01-02")

    with patch('duckdb.connect', return_value=con):
        stats = store.get_statistics()

    assert stats.total_conversations == 1
    assert stats.total_messages == 10
    assert stats.total_projects == 1

    cursor.fetchone.return_value = (
        "conv-1",
        "proj",
        "Title",
        "2025-01-01",
        "2025-01-02",
        2,
        "/tmp/conv.jsonl",
    )

    with patch('duckdb.connect', return_value=con):
        meta = store.get_conversation_meta("conv-1")

    assert meta is not None
    assert meta["conversation_id"] == "conv-1"
    assert meta["file_path"] == "/tmp/conv.jsonl"
