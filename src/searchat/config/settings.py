"""
Configuration management with environment variable and .env support.

Configuration precedence (highest to lowest):
1. Environment variables (SEARCHAT_*)
2. User config file (~/.searchat/config/settings.toml)
3. Default config file (./config/settings.default.toml)
4. Hardcoded constants (constants.py)
"""

import os
from dataclasses import dataclass
from pathlib import Path
import tomli
from dotenv import load_dotenv

from ..core.logging_config import LogConfig
from .constants import (
    DEFAULT_DATA_DIR,
    DEFAULT_CONFIG_SUBDIR,
    SETTINGS_FILE,
    DEFAULT_SETTINGS_FILE,
    ENV_FILE,
    # Defaults
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_INDEX_BATCH_SIZE,
    DEFAULT_MAX_WORKERS,
    DEFAULT_AUTO_INDEX,
    DEFAULT_INDEX_INTERVAL_MINUTES,
    DEFAULT_REINDEX_ON_MODIFICATION,
    DEFAULT_MODIFICATION_DEBOUNCE_MINUTES,
    DEFAULT_SEARCH_MODE,
    DEFAULT_MAX_RESULTS,
    DEFAULT_SNIPPET_LENGTH,
    DEFAULT_MEMORY_LIMIT_MB,
    DEFAULT_QUERY_CACHE_SIZE,
    DEFAULT_ENABLE_PROFILING,
    DEFAULT_THEME,
    DEFAULT_FONT_FAMILY,
    DEFAULT_FONT_SIZE,
    DEFAULT_HIGHLIGHT_COLOR,
    # Environment variable names
    ENV_DATA_DIR,
    ENV_WINDOWS_PROJECTS,
    ENV_WSL_PROJECTS,
    ENV_MEMORY_LIMIT,
    ENV_EMBEDDING_MODEL,
    ENV_EMBEDDING_BATCH,
    ENV_CACHE_SIZE,
    ENV_PROFILING,
    ERROR_NO_CONFIG,
)


# Load .env file at module import time
# Search order: ./.env, ~/.searchat/.env, ~/.searchat/config/.env
def _load_env_files():
    """Load .env files from standard locations."""
    env_locations = [
        Path.cwd() / ENV_FILE,  # Project root
        DEFAULT_DATA_DIR / ENV_FILE,  # Data directory
        DEFAULT_DATA_DIR / DEFAULT_CONFIG_SUBDIR / ENV_FILE,  # Config directory
    ]

    for env_path in env_locations:
        if env_path.exists():
            load_dotenv(env_path, override=False)  # Don't override already-set vars


_load_env_files()


def _get_env_str(key: str, default: str | None = None) -> str | None:
    """Get string value from environment variable. Empty strings are treated as missing."""
    value = os.getenv(key)
    if value is None or value == "":
        return default
    return value


def _get_env_int(key: str, default: int) -> int:
    """Get integer value from environment variable."""
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_env_bool(key: str, default: bool) -> bool:
    """Get boolean value from environment variable."""
    value = os.getenv(key)
    if value is None:
        return default
    return value.lower() in ("true", "1", "yes", "on")


@dataclass
class PathsConfig:
    claude_directory_windows: str
    claude_directory_wsl: str
    search_directory: str
    auto_detect_environment: bool

    @classmethod
    def from_dict(cls, data: dict) -> "PathsConfig":
        """Create PathsConfig from dict with environment variable overrides."""
        return cls(
            claude_directory_windows=_get_env_str(
                ENV_WINDOWS_PROJECTS,
                data.get("claude_directory_windows", "C:/Users/{username}/.claude")
            ),
            claude_directory_wsl=_get_env_str(
                ENV_WSL_PROJECTS,
                data.get("claude_directory_wsl", "")
            ),
            search_directory=_get_env_str(
                ENV_DATA_DIR,
                data.get("search_directory", str(DEFAULT_DATA_DIR))
            ),
            auto_detect_environment=_get_env_bool(
                "SEARCHAT_AUTO_DETECT",
                data.get("auto_detect_environment", True)
            ),
        )


@dataclass
class IndexingConfig:
    batch_size: int
    auto_index: bool
    index_interval_minutes: int
    max_workers: int
    reindex_on_modification: bool
    modification_debounce_minutes: int

    @classmethod
    def from_dict(cls, data: dict) -> "IndexingConfig":
        """Create IndexingConfig from dict with environment variable overrides."""
        return cls(
            batch_size=_get_env_int(
                "SEARCHAT_INDEX_BATCH_SIZE",
                data.get("batch_size", DEFAULT_INDEX_BATCH_SIZE)
            ),
            auto_index=_get_env_bool(
                "SEARCHAT_AUTO_INDEX",
                data.get("auto_index", DEFAULT_AUTO_INDEX)
            ),
            index_interval_minutes=_get_env_int(
                "SEARCHAT_INDEX_INTERVAL",
                data.get("index_interval_minutes", DEFAULT_INDEX_INTERVAL_MINUTES)
            ),
            max_workers=_get_env_int(
                "SEARCHAT_MAX_WORKERS",
                data.get("max_workers", DEFAULT_MAX_WORKERS)
            ),
            reindex_on_modification=_get_env_bool(
                "SEARCHAT_REINDEX_ON_MODIFICATION",
                data.get("reindex_on_modification", DEFAULT_REINDEX_ON_MODIFICATION)
            ),
            modification_debounce_minutes=_get_env_int(
                "SEARCHAT_MODIFICATION_DEBOUNCE_MINUTES",
                data.get("modification_debounce_minutes", DEFAULT_MODIFICATION_DEBOUNCE_MINUTES)
            ),
        )


@dataclass
class SearchConfig:
    default_mode: str
    max_results: int
    snippet_length: int

    @classmethod
    def from_dict(cls, data: dict) -> "SearchConfig":
        """Create SearchConfig from dict with environment variable overrides."""
        return cls(
            default_mode=_get_env_str(
                "SEARCHAT_DEFAULT_MODE",
                data.get("default_mode", DEFAULT_SEARCH_MODE)
            ),
            max_results=_get_env_int(
                "SEARCHAT_MAX_RESULTS",
                data.get("max_results", DEFAULT_MAX_RESULTS)
            ),
            snippet_length=_get_env_int(
                "SEARCHAT_SNIPPET_LENGTH",
                data.get("snippet_length", DEFAULT_SNIPPET_LENGTH)
            ),
        )


@dataclass
class EmbeddingConfig:
    model: str
    batch_size: int
    cache_embeddings: bool
    device: str = "auto"  # auto, cuda, cpu

    @classmethod
    def from_dict(cls, data: dict) -> "EmbeddingConfig":
        """Create EmbeddingConfig from dict with environment variable overrides."""
        return cls(
            model=_get_env_str(
                ENV_EMBEDDING_MODEL,
                data.get("model", DEFAULT_EMBEDDING_MODEL)
            ),
            batch_size=_get_env_int(
                ENV_EMBEDDING_BATCH,
                data.get("batch_size", DEFAULT_EMBEDDING_BATCH_SIZE)
            ),
            cache_embeddings=_get_env_bool(
                "SEARCHAT_CACHE_EMBEDDINGS",
                data.get("cache_embeddings", True)
            ),
            device=_get_env_str(
                "SEARCHAT_EMBEDDING_DEVICE",
                data.get("device", "auto")
            ),
        )

    def get_device(self) -> str:
        """
        Get the actual device to use (resolves 'auto' to cuda/mps/cpu).

        Priority order:
        1. CUDA (NVIDIA GPUs) - Windows, Linux
        2. MPS (Apple Silicon) - macOS M1/M2/M3
        3. CPU (fallback)
        """
        if self.device == "auto":
            try:
                import torch
                # Check CUDA first (NVIDIA GPUs on Windows/Linux)
                if torch.cuda.is_available():
                    return "cuda"
                # Check MPS (Apple Silicon on macOS)
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    return "mps"
                # Fallback to CPU
                return "cpu"
            except ImportError:
                return "cpu"
        return self.device


@dataclass
class UIConfig:
    theme: str
    font_family: str
    font_size: int
    highlight_color: str

    @classmethod
    def from_dict(cls, data: dict) -> "UIConfig":
        """Create UIConfig from dict with environment variable overrides."""
        return cls(
            theme=_get_env_str(
                "SEARCHAT_THEME",
                data.get("theme", DEFAULT_THEME)
            ),
            font_family=_get_env_str(
                "SEARCHAT_FONT_FAMILY",
                data.get("font_family", DEFAULT_FONT_FAMILY)
            ),
            font_size=_get_env_int(
                "SEARCHAT_FONT_SIZE",
                data.get("font_size", DEFAULT_FONT_SIZE)
            ),
            highlight_color=_get_env_str(
                "SEARCHAT_HIGHLIGHT_COLOR",
                data.get("highlight_color", DEFAULT_HIGHLIGHT_COLOR)
            ),
        )


@dataclass
class PerformanceConfig:
    memory_limit_mb: int
    query_cache_size: int
    enable_profiling: bool

    @classmethod
    def from_dict(cls, data: dict) -> "PerformanceConfig":
        """Create PerformanceConfig from dict with environment variable overrides."""
        return cls(
            memory_limit_mb=_get_env_int(
                ENV_MEMORY_LIMIT,
                data.get("memory_limit_mb", DEFAULT_MEMORY_LIMIT_MB)
            ),
            query_cache_size=_get_env_int(
                ENV_CACHE_SIZE,
                data.get("query_cache_size", DEFAULT_QUERY_CACHE_SIZE)
            ),
            enable_profiling=_get_env_bool(
                ENV_PROFILING,
                data.get("enable_profiling", DEFAULT_ENABLE_PROFILING)
            ),
        )


@dataclass
class Config:
    paths: PathsConfig
    indexing: IndexingConfig
    search: SearchConfig
    embedding: EmbeddingConfig
    ui: UIConfig
    performance: PerformanceConfig
    logging: LogConfig

    @classmethod
    def load(cls, config_path: Path | None = None) -> "Config":
        """
        Load configuration with proper precedence.

        Precedence (highest to lowest):
        1. Environment variables (SEARCHAT_*)
        2. User config (~/.searchat/config/settings.toml)
        3. Default config (./config/settings.default.toml)
        4. Hardcoded constants

        Args:
            config_path: Optional explicit config file path

        Returns:
            Loaded Config object

        Raises:
            FileNotFoundError: If no config file is found
        """
        # Determine config file locations
        if config_path is not None:
            # Explicit path provided
            config_files = [config_path]
        else:
            # Standard search order
            user_config = DEFAULT_DATA_DIR / DEFAULT_CONFIG_SUBDIR / SETTINGS_FILE
            default_config = Path(__file__).parent.parent / "config" / DEFAULT_SETTINGS_FILE
            config_files = [user_config, default_config]

        # Try to load from config files in order
        data = None
        loaded_from = None

        for config_file in config_files:
            if config_file.exists():
                with open(config_file, "rb") as f:
                    data = tomli.load(f)
                loaded_from = config_file
                break

        # If no config file found, use empty dict (will use constants.py defaults)
        if data is None:
            # Only raise error if an explicit config path was provided
            if config_path is not None:
                raise FileNotFoundError(
                    ERROR_NO_CONFIG.format(
                        path=config_path,
                        config_dir=DEFAULT_DATA_DIR / DEFAULT_CONFIG_SUBDIR,
                        default_file=DEFAULT_SETTINGS_FILE,
                        settings_file=SETTINGS_FILE,
                    )
                )
            # Otherwise, use empty dict and rely on constants.py
            data = {}

        # Build config objects with environment variable overrides
        return cls(
            paths=PathsConfig.from_dict(data.get("paths", {})),
            indexing=IndexingConfig.from_dict(data.get("indexing", {})),
            search=SearchConfig.from_dict(data.get("search", {})),
            embedding=EmbeddingConfig.from_dict(data.get("embedding", {})),
            ui=UIConfig.from_dict(data.get("ui", {})),
            performance=PerformanceConfig.from_dict(data.get("performance", {})),
            logging=LogConfig(**data.get("logging", {})),
        )
