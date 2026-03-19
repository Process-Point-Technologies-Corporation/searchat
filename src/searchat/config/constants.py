"""
Constants and default values for Searchat.

Centralizes magic numbers and strings to improve maintainability.
"""

from pathlib import Path

# ============================================================================
# Application Metadata
# ============================================================================

APP_NAME = "searchat"
APP_VERSION = "0.1.0"
CONFIG_DIR_NAME = ".searchat"

# ============================================================================
# Path Defaults
# ============================================================================

# Default Claude conversation directory name
CLAUDE_DIR_NAME = ".claude"
CLAUDE_PROJECTS_SUBDIR = "projects"
CODEX_DIR_NAME = ".codex"
CODEX_SESSIONS_SUBDIR = "sessions"

# Default data directory patterns
DEFAULT_DATA_DIR = Path.home() / CONFIG_DIR_NAME
DEFAULT_EXCLUDED_CONVERSATIONS_DIR = ""  # No default; must be set in settings.toml
DEFAULT_CONFIG_SUBDIR = "config"
DEFAULT_DATA_SUBDIR = "data"
DEFAULT_LOGS_SUBDIR = "logs"

# Config file names
SETTINGS_FILE = "settings.toml"
DEFAULT_SETTINGS_FILE = "settings.default.toml"
SETTINGS_TEMPLATE_FILE = "settings.template.toml"
ENV_FILE = ".env"

# ============================================================================
# Web Server Defaults
# ============================================================================

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
PORT_SCAN_RANGE = (8000, 8010)  # Will try ports in this range

# ============================================================================
# Search & Indexing Defaults
# ============================================================================

# Embedding model
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_BATCH_SIZE = 32

# Text chunking
DEFAULT_CHUNK_SIZE = 1500
DEFAULT_CHUNK_OVERLAP = 200

# Indexing
DEFAULT_INDEX_BATCH_SIZE = 1000
DEFAULT_MAX_WORKERS = 4
DEFAULT_AUTO_INDEX = True
DEFAULT_INDEX_INTERVAL_MINUTES = 60
DEFAULT_REINDEX_ON_MODIFICATION = True
DEFAULT_MODIFICATION_DEBOUNCE_MINUTES = 5
DEFAULT_EXCLUDED_PROMPT_PREFIXES = (
    "Distill this conversation exchange into JSON",        # current perturn + batch distillation prompt
    "Distill the conversation exchange below into JSON",   # legacy perturn distillation prompt
    "You are a strict relevance assessor",                 # optional relevance-assessment prompt
    "Your task is to create a detailed summary of the conversation so far",   # Claude Code auto-compaction prompt
    "Your task is to create a detailed summary of the RECENT portion",        # Claude Code partial compaction variant
)

# Search
DEFAULT_SEARCH_MODE = "hybrid"
DEFAULT_MAX_RESULTS = 100
DEFAULT_SNIPPET_LENGTH = 200

# Search Ranking (unified two-layer search)
DEFAULT_INTERSECTION_BOOST = 0.2  # 20% boost for results appearing in both layers
DEFAULT_PALACE_WEIGHT = 0.5  # Weight for palace layer (before boost scaling)
DEFAULT_VERBATIM_WEIGHT = 0.5  # Weight for verbatim layer (before boost scaling)

# Hybrid Search Tuning Parameters
DEFAULT_KEYWORD_WEIGHT = 0.8  # Weight for BM25 keyword results
DEFAULT_SEMANTIC_WEIGHT = 0.2  # Weight for FAISS semantic results
DEFAULT_RANK_DECAY = 0.1  # Decay constant for rank-based weighting (1/(1 + decay*rank))
DEFAULT_TITLE_BOOST = 2.0  # Multiplier when query terms appear in title
DEFAULT_BM25_K1 = 2.5  # BM25 term frequency saturation
DEFAULT_BM25_B = 0.25  # BM25 document length normalization
DEFAULT_BM25_CANDIDATES = 500  # Number of BM25 candidates to retrieve before filtering
DEFAULT_FAISS_K = 100  # Number of FAISS nearest neighbors to retrieve

# ============================================================================
# Performance Defaults
# ============================================================================

DEFAULT_MEMORY_LIMIT_MB = 3000
DEFAULT_QUERY_CACHE_SIZE = 100
DEFAULT_ENABLE_PROFILING = False
DEFAULT_STARTUP_WARMUP_MODE = "keyword"

# ============================================================================
# UI Defaults
# ============================================================================

DEFAULT_THEME = "auto"
DEFAULT_FONT_FAMILY = "Segoe UI"
DEFAULT_FONT_SIZE = 11
DEFAULT_HIGHLIGHT_COLOR = "#FFEB3B"

# ============================================================================
# Platform Detection
# ============================================================================

# Common Claude directory locations by platform
CLAUDE_DIR_CANDIDATES = [
    Path.home() / CLAUDE_DIR_NAME / CLAUDE_PROJECTS_SUBDIR,  # Standard location
    Path.home() / CLAUDE_DIR_NAME,  # Fallback
]

# Common Codex directory locations by platform
CODEX_DIR_CANDIDATES = [
    Path.home() / CODEX_DIR_NAME / CODEX_SESSIONS_SUBDIR,
]

# WSL mount point patterns
WSL_MOUNT_PREFIX = "/mnt/"
WSL_UNC_PREFIX = "\\\\wsl$\\"

# ============================================================================
# Environment Variable Names
# ============================================================================

ENV_DATA_DIR = "SEARCHAT_DATA_DIR"
ENV_WINDOWS_PROJECTS = "SEARCHAT_WINDOWS_PROJECTS_DIR"
ENV_WSL_PROJECTS = "SEARCHAT_WSL_PROJECTS_DIR"
ENV_ADDITIONAL_DIRS = "SEARCHAT_ADDITIONAL_DIRS"

ENV_PORT = "SEARCHAT_PORT"
ENV_HOST = "SEARCHAT_HOST"

ENV_MEMORY_LIMIT = "SEARCHAT_MEMORY_LIMIT_MB"
ENV_EMBEDDING_MODEL = "SEARCHAT_EMBEDDING_MODEL"
ENV_EMBEDDING_BATCH = "SEARCHAT_EMBEDDING_BATCH_SIZE"
ENV_CACHE_SIZE = "SEARCHAT_QUERY_CACHE_SIZE"
ENV_PROFILING = "SEARCHAT_ENABLE_PROFILING"
ENV_STARTUP_WARMUP_MODE = "SEARCHAT_STARTUP_WARMUP_MODE"

ENV_ISOLATION_MODE = "SEARCHAT_ISOLATION_MODE"
ENV_VARIANT_SUFFIX = "SEARCHAT_VARIANT_SUFFIX"

# Ranking
ENV_INTERSECTION_BOOST = "SEARCHAT_INTERSECTION_BOOST"
ENV_PALACE_WEIGHT = "SEARCHAT_PALACE_WEIGHT"
ENV_VERBATIM_WEIGHT = "SEARCHAT_VERBATIM_WEIGHT"

# Hybrid Search Tuning
ENV_KEYWORD_WEIGHT = "SEARCHAT_KEYWORD_WEIGHT"
ENV_SEMANTIC_WEIGHT = "SEARCHAT_SEMANTIC_WEIGHT"
ENV_RANK_DECAY = "SEARCHAT_RANK_DECAY"
ENV_TITLE_BOOST = "SEARCHAT_TITLE_BOOST"
ENV_BM25_K1 = "SEARCHAT_BM25_K1"
ENV_BM25_B = "SEARCHAT_BM25_B"
ENV_BM25_CANDIDATES = "SEARCHAT_BM25_CANDIDATES"
ENV_FAISS_K = "SEARCHAT_FAISS_K"

# ============================================================================
# Unified Search Engine Defaults
# ============================================================================

# Search engine mode: "unified" (DuckDB-based search with VSS/FTS)
DEFAULT_SEARCH_ENGINE = "unified"

ENV_SEARCH_ENGINE = "SEARCHAT_SEARCH_ENGINE"

# ============================================================================
# Distillation Defaults
# ============================================================================

DEFAULT_DISTILLATION_PROVIDER = "auto"
DEFAULT_DISTILLATION_CLI_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_DISTILLATION_CLI_MODEL_OPENAI = "gpt-5.3-codex"
DEFAULT_DISTILLATION_BATCH_SIZE = 10
DEFAULT_DISTILLATION_MAX_PLY_LENGTH = 20
DEFAULT_DISTILLATION_MIN_EXCHANGE_CHARS = 100

# Batch distillation prompt (comprehensive, for batch processing with room assignments)
DEFAULT_DISTILLATION_PROMPT = """Distill this conversation exchange into JSON:

- "exchange_core": 1-2 sentences. What was accomplished or decided? Use the specific terms from the exchange. Do not invent details not present in the text. If the exchange is mostly empty, say so briefly.
- "specific_context": One concrete detail from the text: a number, error message, parameter name, or file path. Copy it exactly from the text. Do not use the project path.
- "room_assignments": 1-3 rooms. Each room is a topic this exchange belongs to. {{"room_type": "<file|concept|workflow>", "room_key": "<identifier>", "room_label": "<short label>", "relevance": <0.0-1.0>}}. A room should be specific enough to group related exchanges (e.g. "retry_timeout" not "errors").

Do NOT include "files_touched".

Project: {project_id}

Exchange (messages {ply_start}-{ply_end}):
{messages_text}

Respond with ONLY valid JSON."""

ENV_DISTILLATION_PROVIDER = "SEARCHAT_DISTILLATION_PROVIDER"
ENV_DISTILLATION_CLI_MODEL = "SEARCHAT_DISTILLATION_CLI_MODEL"
ENV_DISTILLATION_BATCH_SIZE = "SEARCHAT_DISTILLATION_BATCH_SIZE"
ENV_DISTILLATION_MAX_PLY_LENGTH = "SEARCHAT_DISTILLATION_MAX_PLY_LENGTH"
ENV_DISTILLATION_MIN_EXCHANGE_CHARS = "SEARCHAT_DISTILLATION_MIN_EXCHANGE_CHARS"

# ============================================================================
# Backfill Defaults (local llama-server distillation)
# ============================================================================

DEFAULT_BACKFILL_LLM_URL = "http://localhost:8080"
DEFAULT_BACKFILL_TIMEOUT = 180.0
DEFAULT_BACKFILL_BATCH_SIZE = 16

# Size tiers: (max_chars, max_concurrent, name)
# Assumes ~3 chars per token, and llama-server allocates ctx_size / parallel per slot
# For 65536 ctx / 8 parallel = 8K tokens/slot ≈ 24K chars max
DEFAULT_BACKFILL_TIER_SMALL_MAX_CHARS = 12000  # fits in 4K tokens
DEFAULT_BACKFILL_TIER_SMALL_CONCURRENT = 8
DEFAULT_BACKFILL_TIER_MEDIUM_MAX_CHARS = 48000  # fits in 16K tokens
DEFAULT_BACKFILL_TIER_MEDIUM_CONCURRENT = 4
DEFAULT_BACKFILL_TIER_LARGE_MAX_CHARS = 120000  # fits in 40K tokens
DEFAULT_BACKFILL_TIER_LARGE_CONCURRENT = 2
DEFAULT_BACKFILL_TIER_HUGE_CONCURRENT = 1  # everything else, 1 at a time

ENV_BACKFILL_LLM_URL = "SEARCHAT_BACKFILL_LLM_URL"
ENV_BACKFILL_TIMEOUT = "SEARCHAT_BACKFILL_TIMEOUT"
ENV_BACKFILL_BATCH_SIZE = "SEARCHAT_BACKFILL_BATCH_SIZE"

# ============================================================================
# Error Messages
# ============================================================================

ERROR_NO_CONFIG = """
Configuration file not found: {path}

Run the setup wizard to create configuration:
    python -m searchat.setup

Or manually copy the example config:
    mkdir -p {config_dir}
    cp config/{template_file} {config_dir}/{settings_file}
    cp .env.example {config_dir}/.env
"""

ERROR_NO_CLAUDE_DIR = """
Claude conversation directory not found.

Searched locations:
{locations}

Please ensure Claude CLI is installed and has created conversations, or
specify the directory manually in {config_file}
"""

ERROR_INVALID_PORT = """
Invalid port number: {port}

Port must be between 1 and 65535.
"""

ERROR_PORT_IN_USE = """
All ports in range {start}-{end} are in use.

Try:
1. Stop other services using these ports
2. Specify a different port: SEARCHAT_PORT=9000 searchat-web
3. Check for zombie processes: netstat -ano | findstr :{port}
"""


