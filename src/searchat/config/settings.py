"""
Configuration management with environment variable and .env support.

Configuration precedence (highest to lowest):
1. Environment variables (SEARCHAT_*)
2. User config file (~/.searchat/config/settings.toml)
3. Default config file (./config/settings.default.toml)
4. Hardcoded constants (constants.py)
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import tomli
from dotenv import load_dotenv

from .constants import (
    DEFAULT_DATA_DIR,
    DEFAULT_CONFIG_SUBDIR,
    DEFAULT_EXCLUDED_CONVERSATIONS_DIR,
    SETTINGS_FILE,
    DEFAULT_SETTINGS_FILE,
    SETTINGS_TEMPLATE_FILE,
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
    DEFAULT_EXCLUDED_PROMPT_PREFIXES,
    DEFAULT_SEARCH_MODE,
    DEFAULT_MAX_RESULTS,
    DEFAULT_SNIPPET_LENGTH,
    DEFAULT_INTERSECTION_BOOST,
    DEFAULT_PALACE_WEIGHT,
    DEFAULT_VERBATIM_WEIGHT,
    DEFAULT_MEMORY_LIMIT_MB,
    DEFAULT_QUERY_CACHE_SIZE,
    DEFAULT_ENABLE_PROFILING,
    DEFAULT_STARTUP_WARMUP_MODE,
    DEFAULT_THEME,
    DEFAULT_FONT_FAMILY,
    DEFAULT_FONT_SIZE,
    DEFAULT_HIGHLIGHT_COLOR,
    # Hybrid search tuning defaults
    DEFAULT_KEYWORD_WEIGHT,
    DEFAULT_SEMANTIC_WEIGHT,
    DEFAULT_RANK_DECAY,
    DEFAULT_TITLE_BOOST,
    DEFAULT_BM25_K1,
    DEFAULT_BM25_B,
    DEFAULT_BM25_CANDIDATES,
    DEFAULT_FAISS_K,
    # Environment variable names
    ENV_DATA_DIR,
    ENV_WINDOWS_PROJECTS,
    ENV_WSL_PROJECTS,
    ENV_AUTO_DETECT,
    ENV_EXCLUDED_CONVERSATIONS_DIR,
    ENV_INDEX_BATCH_SIZE,
    ENV_AUTO_INDEX,
    ENV_INDEX_INTERVAL,
    ENV_MAX_WORKERS,
    ENV_REINDEX_ON_MODIFICATION,
    ENV_MODIFICATION_DEBOUNCE_MINUTES,
    ENV_DEFAULT_MODE,
    ENV_MAX_RESULTS,
    ENV_SNIPPET_LENGTH,
    ENV_MEMORY_LIMIT,
    ENV_EMBEDDING_MODEL,
    ENV_EMBEDDING_BATCH,
    ENV_CACHE_EMBEDDINGS,
    ENV_EMBEDDING_DEVICE,
    ENV_CACHE_SIZE,
    ENV_PROFILING,
    ENV_STARTUP_WARMUP_MODE,
    ENV_THEME,
    ENV_FONT_FAMILY,
    ENV_FONT_SIZE,
    ENV_HIGHLIGHT_COLOR,
    ENV_INTERSECTION_BOOST,
    ENV_PALACE_WEIGHT,
    ENV_VERBATIM_WEIGHT,
    # Hybrid search tuning env vars
    ENV_KEYWORD_WEIGHT,
    ENV_SEMANTIC_WEIGHT,
    ENV_RANK_DECAY,
    ENV_TITLE_BOOST,
    ENV_BM25_K1,
    ENV_BM25_B,
    ENV_BM25_CANDIDATES,
    ENV_FAISS_K,
    ERROR_NO_CONFIG,
    # Distillation defaults
    DEFAULT_DISTILLATION_CLI_MODEL,
    DEFAULT_DISTILLATION_PROVIDER,
    DEFAULT_DISTILLATION_BATCH_SIZE,
    DEFAULT_DISTILLATION_MAX_PLY_LENGTH,
    DEFAULT_DISTILLATION_MIN_EXCHANGE_CHARS,
    DEFAULT_DISTILLATION_PROMPT,
    DEFAULT_PERTURN_PROMPT,
    ENV_DISTILLATION_PROVIDER,
    ENV_DISTILLATION_CLI_MODEL,
    ENV_DISTILLATION_BATCH_SIZE,
    ENV_DISTILLATION_MAX_PLY_LENGTH,
    ENV_DISTILLATION_MIN_EXCHANGE_CHARS,
    # Unified search engine
    DEFAULT_SEARCH_ENGINE,
    ENV_SEARCH_ENGINE,
    # Backfill defaults
    DEFAULT_BACKFILL_LLM_URL,
    DEFAULT_BACKFILL_TIMEOUT,
    DEFAULT_BACKFILL_BATCH_SIZE,
    DEFAULT_BACKFILL_TIER_SMALL_MAX_CHARS,
    DEFAULT_BACKFILL_TIER_SMALL_CONCURRENT,
    DEFAULT_BACKFILL_TIER_MEDIUM_MAX_CHARS,
    DEFAULT_BACKFILL_TIER_MEDIUM_CONCURRENT,
    DEFAULT_BACKFILL_TIER_LARGE_MAX_CHARS,
    DEFAULT_BACKFILL_TIER_LARGE_CONCURRENT,
    DEFAULT_BACKFILL_TIER_HUGE_CONCURRENT,
    ENV_BACKFILL_LLM_URL,
    ENV_BACKFILL_TIMEOUT,
    ENV_BACKFILL_BATCH_SIZE,
)

logger = logging.getLogger(__name__)


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


def _get_env_str(key: str, default: Optional[str] = None) -> Optional[str]:
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
        logger.warning("Invalid integer for %s: %r. Using default %r.", key, value, default)
        return default


def _get_env_bool(key: str, default: bool) -> bool:
    """Get boolean value from environment variable."""
    value = os.getenv(key)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in ("true", "1", "yes", "on"):
        return True
    if normalized in ("false", "0", "no", "off"):
        return False
    logger.warning("Invalid boolean for %s: %r. Using default %r.", key, value, default)
    return default


def _get_env_float(key: str, default: float) -> float:
    """Get float value from environment variable."""
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("Invalid float for %s: %r. Using default %r.", key, value, default)
        return default


@dataclass
class PathsConfig:
    claude_directory_windows: str
    claude_directory_wsl: str
    search_directory: str
    auto_detect_environment: bool
    excluded_conversations_dir: str

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
                ENV_AUTO_DETECT,
                data.get("auto_detect_environment", True)
            ),
            excluded_conversations_dir=_get_env_str(
                ENV_EXCLUDED_CONVERSATIONS_DIR,
                data.get("excluded_conversations_dir", DEFAULT_EXCLUDED_CONVERSATIONS_DIR)
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
    excluded_prompt_prefixes: tuple

    @classmethod
    def from_dict(cls, data: dict, config_dir: Optional[Path] = None) -> "IndexingConfig":
        """Create IndexingConfig from dict with hardware detection and env overrides."""
        # Get hardware profile for optimal worker count
        hw_workers = DEFAULT_MAX_WORKERS

        if config_dir is not None:
            try:
                from searchat.utils.hardware import get_or_detect_hardware
                hw_profile = get_or_detect_hardware(config_dir, force_detect=False)
                hw_workers = hw_profile.indexing_workers
            except Exception as e:
                logger.warning(
                    "Failed to load hardware profile for indexing config from %s: %s",
                    config_dir,
                    e,
                )

        workers_default = data.get("max_workers", hw_workers)

        return cls(
            batch_size=_get_env_int(
                ENV_INDEX_BATCH_SIZE,
                data.get("batch_size", DEFAULT_INDEX_BATCH_SIZE)
            ),
            auto_index=_get_env_bool(
                ENV_AUTO_INDEX,
                data.get("auto_index", DEFAULT_AUTO_INDEX)
            ),
            index_interval_minutes=_get_env_int(
                ENV_INDEX_INTERVAL,
                data.get("index_interval_minutes", DEFAULT_INDEX_INTERVAL_MINUTES)
            ),
            max_workers=_get_env_int(
                ENV_MAX_WORKERS,
                workers_default
            ),
            reindex_on_modification=_get_env_bool(
                ENV_REINDEX_ON_MODIFICATION,
                data.get("reindex_on_modification", DEFAULT_REINDEX_ON_MODIFICATION)
            ),
            modification_debounce_minutes=_get_env_int(
                ENV_MODIFICATION_DEBOUNCE_MINUTES,
                data.get("modification_debounce_minutes", DEFAULT_MODIFICATION_DEBOUNCE_MINUTES)
            ),
            excluded_prompt_prefixes=tuple(
                data.get("excluded_prompt_prefixes", DEFAULT_EXCLUDED_PROMPT_PREFIXES)
            ),
        )


@dataclass
class RankingConfig:
    """Configuration for unified search result ranking."""
    intersection_boost: float  # Multiplier for results appearing in both layers
    palace_weight: float  # Weight for palace layer scores
    verbatim_weight: float  # Weight for verbatim layer scores
    # Hybrid search tuning parameters
    keyword_weight: float  # Weight for BM25 keyword results in hybrid fusion
    semantic_weight: float  # Weight for FAISS semantic results in hybrid fusion
    rank_decay: float  # Decay constant for rank-based weighting
    title_boost: float  # Multiplier when query terms appear in title
    bm25_k1: float  # BM25 term frequency saturation parameter
    bm25_b: float  # BM25 document length normalization parameter
    bm25_candidates: int  # Number of BM25 candidates to retrieve
    faiss_k: int  # Number of FAISS nearest neighbors to retrieve

    @classmethod
    def from_dict(cls, data: dict) -> "RankingConfig":
        """Create RankingConfig from dict with environment variable overrides."""
        return cls(
            intersection_boost=_get_env_float(
                ENV_INTERSECTION_BOOST,
                data.get("intersection_boost", DEFAULT_INTERSECTION_BOOST)
            ),
            palace_weight=_get_env_float(
                ENV_PALACE_WEIGHT,
                data.get("palace_weight", DEFAULT_PALACE_WEIGHT)
            ),
            verbatim_weight=_get_env_float(
                ENV_VERBATIM_WEIGHT,
                data.get("verbatim_weight", DEFAULT_VERBATIM_WEIGHT)
            ),
            keyword_weight=_get_env_float(
                ENV_KEYWORD_WEIGHT,
                data.get("keyword_weight", DEFAULT_KEYWORD_WEIGHT)
            ),
            semantic_weight=_get_env_float(
                ENV_SEMANTIC_WEIGHT,
                data.get("semantic_weight", DEFAULT_SEMANTIC_WEIGHT)
            ),
            rank_decay=_get_env_float(
                ENV_RANK_DECAY,
                data.get("rank_decay", DEFAULT_RANK_DECAY)
            ),
            title_boost=_get_env_float(
                ENV_TITLE_BOOST,
                data.get("title_boost", DEFAULT_TITLE_BOOST)
            ),
            bm25_k1=_get_env_float(
                ENV_BM25_K1,
                data.get("bm25_k1", DEFAULT_BM25_K1)
            ),
            bm25_b=_get_env_float(
                ENV_BM25_B,
                data.get("bm25_b", DEFAULT_BM25_B)
            ),
            bm25_candidates=_get_env_int(
                ENV_BM25_CANDIDATES,
                data.get("bm25_candidates", DEFAULT_BM25_CANDIDATES)
            ),
            faiss_k=_get_env_int(
                ENV_FAISS_K,
                data.get("faiss_k", DEFAULT_FAISS_K)
            ),
        )

    @property
    def boost_multiplier(self) -> float:
        """Convert percentage boost to multiplier (0.2 -> 1.2)."""
        return 1 + self.intersection_boost

    @property
    def scaled_palace_weight(self) -> float:
        """Palace weight scaled so max intersection score = 1.0."""
        return self.palace_weight / self.boost_multiplier

    @property
    def scaled_verbatim_weight(self) -> float:
        """Verbatim weight scaled so max intersection score = 1.0."""
        return self.verbatim_weight / self.boost_multiplier


@dataclass
class SearchConfig:
    default_mode: str
    max_results: int
    snippet_length: int
    ranking: RankingConfig
    engine: str  # "legacy" | "unified" | "compare"

    @classmethod
    def from_dict(cls, data: dict) -> "SearchConfig":
        """Create SearchConfig from dict with environment variable overrides."""
        return cls(
            default_mode=_get_env_str(
                ENV_DEFAULT_MODE,
                data.get("default_mode", DEFAULT_SEARCH_MODE)
            ),
            max_results=_get_env_int(
                ENV_MAX_RESULTS,
                data.get("max_results", DEFAULT_MAX_RESULTS)
            ),
            snippet_length=_get_env_int(
                ENV_SNIPPET_LENGTH,
                data.get("snippet_length", DEFAULT_SNIPPET_LENGTH)
            ),
            ranking=RankingConfig.from_dict(data.get("ranking", {})),
            engine=_get_env_str(
                ENV_SEARCH_ENGINE,
                data.get("engine", DEFAULT_SEARCH_ENGINE)
            ),
        )


@dataclass
class EmbeddingConfig:
    model: str
    batch_size: int
    cache_embeddings: bool
    device: str = "auto"  # auto, cuda, cpu

    @classmethod
    def from_dict(cls, data: dict, config_dir: Optional[Path] = None) -> "EmbeddingConfig":
        """Create EmbeddingConfig from dict with hardware detection and env overrides.

        Priority for batch_size and device:
        1. Environment variable
        2. User config (settings.toml)
        3. Hardware profile (hardware.toml) - auto-detected
        4. Hardcoded defaults
        """
        # Get hardware profile if available
        hw_batch_size = DEFAULT_EMBEDDING_BATCH_SIZE
        hw_device = "auto"

        if config_dir is not None:
            try:
                from searchat.utils.hardware import get_or_detect_hardware
                hw_profile = get_or_detect_hardware(config_dir, force_detect=False)
                hw_batch_size = hw_profile.embedding_batch_size
                hw_device = hw_profile.embedding_device
            except Exception as e:
                logger.warning(
                    "Failed to load hardware profile for embedding config from %s: %s",
                    config_dir,
                    e,
                )

        # Use hardware profile as fallback if not in user config
        batch_default = data.get("batch_size", hw_batch_size)
        device_default = data.get("device", hw_device)

        return cls(
            model=_get_env_str(
                ENV_EMBEDDING_MODEL,
                data.get("model", DEFAULT_EMBEDDING_MODEL)
            ),
            batch_size=_get_env_int(
                ENV_EMBEDDING_BATCH,
                batch_default
            ),
            cache_embeddings=_get_env_bool(
                ENV_CACHE_EMBEDDINGS,
                data.get("cache_embeddings", True)
            ),
            device=_get_env_str(
                ENV_EMBEDDING_DEVICE,
                device_default
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
                ENV_THEME,
                data.get("theme", DEFAULT_THEME)
            ),
            font_family=_get_env_str(
                ENV_FONT_FAMILY,
                data.get("font_family", DEFAULT_FONT_FAMILY)
            ),
            font_size=_get_env_int(
                ENV_FONT_SIZE,
                data.get("font_size", DEFAULT_FONT_SIZE)
            ),
            highlight_color=_get_env_str(
                ENV_HIGHLIGHT_COLOR,
                data.get("highlight_color", DEFAULT_HIGHLIGHT_COLOR)
            ),
        )


@dataclass
class PerformanceConfig:
    memory_limit_mb: int
    query_cache_size: int
    enable_profiling: bool
    startup_warmup_mode: str

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
            startup_warmup_mode=_get_env_str(
                ENV_STARTUP_WARMUP_MODE,
                data.get("startup_warmup_mode", DEFAULT_STARTUP_WARMUP_MODE)
            ),
        )


@dataclass
class DistillationConfig:
    provider: str
    cli_model: str
    batch_size: int
    max_ply_length: int
    min_exchange_chars: int
    prompt: str
    perturn_prompt: str

    @classmethod
    def from_dict(cls, data: dict) -> "DistillationConfig":
        provider = _get_env_str(
            ENV_DISTILLATION_PROVIDER,
            data.get("provider", DEFAULT_DISTILLATION_PROVIDER)
        )
        normalized_provider = (provider or DEFAULT_DISTILLATION_PROVIDER).strip().lower()
        if normalized_provider not in {"claude", "openai", "auto"}:
            logger.warning(
                "Invalid distillation provider %r. Using default %r.",
                provider,
                DEFAULT_DISTILLATION_PROVIDER,
            )
            normalized_provider = DEFAULT_DISTILLATION_PROVIDER

        return cls(
            provider=normalized_provider,
            cli_model=_get_env_str(
                ENV_DISTILLATION_CLI_MODEL,
                data.get("cli_model", DEFAULT_DISTILLATION_CLI_MODEL)
            ),
            batch_size=_get_env_int(
                ENV_DISTILLATION_BATCH_SIZE,
                data.get("batch_size", DEFAULT_DISTILLATION_BATCH_SIZE)
            ),
            max_ply_length=_get_env_int(
                ENV_DISTILLATION_MAX_PLY_LENGTH,
                data.get("max_ply_length", DEFAULT_DISTILLATION_MAX_PLY_LENGTH)
            ),
            min_exchange_chars=_get_env_int(
                ENV_DISTILLATION_MIN_EXCHANGE_CHARS,
                data.get("min_exchange_chars", DEFAULT_DISTILLATION_MIN_EXCHANGE_CHARS)
            ),
            prompt=data.get("prompt", DEFAULT_DISTILLATION_PROMPT),
            perturn_prompt=data.get("perturn_prompt", DEFAULT_PERTURN_PROMPT),
        )


@dataclass
class BackfillTier:
    """A size tier for backfill processing."""
    name: str
    max_chars: int  # Max exchange size in chars for this tier (inf for last tier)
    max_concurrent: int  # Concurrent requests for this tier


@dataclass
class BackfillConfig:
    """Configuration for local llama-server backfill."""
    llm_url: str
    timeout: float
    batch_size: int
    tiers: list  # List of BackfillTier

    @classmethod
    def from_dict(cls, data: dict) -> "BackfillConfig":
        tiers_data = data.get("tiers", [])
        if tiers_data:
            tiers = [
                BackfillTier(
                    name=t.get("name", f"tier_{i}"),
                    max_chars=t.get("max_chars", float("inf")),
                    max_concurrent=t.get("max_concurrent", 1),
                )
                for i, t in enumerate(tiers_data)
            ]
        else:
            # Default tiers
            tiers = [
                BackfillTier("small", DEFAULT_BACKFILL_TIER_SMALL_MAX_CHARS, DEFAULT_BACKFILL_TIER_SMALL_CONCURRENT),
                BackfillTier("medium", DEFAULT_BACKFILL_TIER_MEDIUM_MAX_CHARS, DEFAULT_BACKFILL_TIER_MEDIUM_CONCURRENT),
                BackfillTier("large", DEFAULT_BACKFILL_TIER_LARGE_MAX_CHARS, DEFAULT_BACKFILL_TIER_LARGE_CONCURRENT),
                BackfillTier("huge", float("inf"), DEFAULT_BACKFILL_TIER_HUGE_CONCURRENT),
            ]
        return cls(
            llm_url=_get_env_str(
                ENV_BACKFILL_LLM_URL,
                data.get("llm_url", DEFAULT_BACKFILL_LLM_URL)
            ),
            timeout=_get_env_float(
                ENV_BACKFILL_TIMEOUT,
                data.get("timeout", DEFAULT_BACKFILL_TIMEOUT)
            ),
            batch_size=_get_env_int(
                ENV_BACKFILL_BATCH_SIZE,
                data.get("batch_size", DEFAULT_BACKFILL_BATCH_SIZE)
            ),
            tiers=tiers,
        )

    def get_tier_for_size(self, text_len: int) -> BackfillTier:
        """Get the tier for a given text length."""
        for tier in self.tiers:
            if text_len <= tier.max_chars:
                return tier
        return self.tiers[-1]  # Last tier handles everything else


@dataclass
class Config:
    paths: PathsConfig
    indexing: IndexingConfig
    search: SearchConfig
    embedding: EmbeddingConfig
    ui: UIConfig
    performance: PerformanceConfig
    distillation: DistillationConfig
    backfill: BackfillConfig

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "Config":
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
                        template_file=SETTINGS_TEMPLATE_FILE,
                        settings_file=SETTINGS_FILE,
                    )
                )
            # Otherwise, use empty dict and rely on constants.py
            data = {}

        # Build config objects with environment variable overrides
        config_dir = DEFAULT_DATA_DIR / DEFAULT_CONFIG_SUBDIR
        return cls(
            paths=PathsConfig.from_dict(data.get("paths", {})),
            indexing=IndexingConfig.from_dict(data.get("indexing", {}), config_dir=config_dir),
            search=SearchConfig.from_dict(data.get("search", {})),
            embedding=EmbeddingConfig.from_dict(data.get("embedding", {}), config_dir=config_dir),
            ui=UIConfig.from_dict(data.get("ui", {})),
            performance=PerformanceConfig.from_dict(data.get("performance", {})),
            distillation=DistillationConfig.from_dict(data.get("distillation", {})),
            backfill=BackfillConfig.from_dict(data.get("backfill", {})),
        )
