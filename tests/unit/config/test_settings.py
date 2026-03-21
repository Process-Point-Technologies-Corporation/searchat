from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import sys

from searchat.config.settings import (
    DistillationConfig,
    EmbeddingConfig,
    IndexingConfig,
    PerformanceConfig,
    _get_env_bool,
    _get_env_float,
    _get_env_int,
)


def test_invalid_bool_env_uses_default():
    with patch.dict("os.environ", {"SEARCHAT_AUTO_INDEX": "ture"}, clear=False):
        assert _get_env_bool("SEARCHAT_AUTO_INDEX", True) is True


def test_invalid_int_env_uses_default():
    with patch.dict("os.environ", {"SEARCHAT_MAX_RESULTS": "many"}, clear=False):
        assert _get_env_int("SEARCHAT_MAX_RESULTS", 25) == 25


def test_invalid_float_env_uses_default():
    with patch.dict("os.environ", {"SEARCHAT_INTERSECTION_BOOST": "high"}, clear=False):
        assert _get_env_float("SEARCHAT_INTERSECTION_BOOST", 0.2) == 0.2


def test_indexing_config_logs_and_falls_back_when_hardware_profile_fails(caplog):
    fake_hardware = SimpleNamespace(
        get_or_detect_hardware=lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("hardware profile broken")
        )
    )
    with patch.dict(sys.modules, {"searchat.utils.hardware": fake_hardware}):
        config = IndexingConfig.from_dict({}, config_dir=Path.cwd())

    assert config.max_workers > 0
    assert "Failed to load hardware profile for indexing config" in caplog.text


def test_embedding_config_logs_and_falls_back_when_hardware_profile_fails(caplog):
    fake_hardware = SimpleNamespace(
        get_or_detect_hardware=lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("hardware profile broken")
        )
    )
    with patch.dict(sys.modules, {"searchat.utils.hardware": fake_hardware}):
        config = EmbeddingConfig.from_dict({}, config_dir=Path.cwd())

    assert config.batch_size > 0
    assert "Failed to load hardware profile for embedding config" in caplog.text


def test_performance_config_defaults_to_keyword_startup_warmup():
    config = PerformanceConfig.from_dict({})
    assert config.startup_warmup_mode == "keyword"


def test_performance_config_accepts_startup_warmup_env_override():
    with patch.dict("os.environ", {"SEARCHAT_STARTUP_WARMUP_MODE": "semantic"}, clear=False):
        config = PerformanceConfig.from_dict({})
    assert config.startup_warmup_mode == "semantic"


def test_distillation_config_accepts_openai_provider():
    config = DistillationConfig.from_dict({"provider": "openai", "cli_model": "gpt-5"})
    assert config.provider == "openai"
    assert config.cli_model == "gpt-5"


def test_distillation_config_invalid_provider_falls_back_to_claude(caplog):
    config = DistillationConfig.from_dict({"provider": "bogus"})
    assert config.provider == "auto"
    assert "Invalid distillation provider" in caplog.text
