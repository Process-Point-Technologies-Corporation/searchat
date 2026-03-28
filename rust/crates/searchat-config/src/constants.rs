/// Application metadata
pub const APP_NAME: &str = "searchat";
pub const APP_VERSION: &str = "0.1.0";
pub const CONFIG_DIR_NAME: &str = ".searchat";

/// Path constants
pub const CLAUDE_DIR_NAME: &str = ".claude";
pub const CLAUDE_PROJECTS_SUBDIR: &str = "projects";
pub const CODEX_DIR_NAME: &str = ".codex";
pub const CODEX_SESSIONS_SUBDIR: &str = "sessions";
pub const DEFAULT_CONFIG_SUBDIR: &str = "config";
pub const DEFAULT_DATA_SUBDIR: &str = "data";
pub const DEFAULT_LOGS_SUBDIR: &str = "logs";

/// Config file names
pub const SETTINGS_FILE: &str = "settings.toml";
pub const DEFAULT_SETTINGS_FILE: &str = "settings.default.toml";
pub const SETTINGS_TEMPLATE_FILE: &str = "settings.template.toml";
pub const ENV_FILE: &str = ".env";

/// WSL path prefixes
pub const WSL_MOUNT_PREFIX: &str = "/mnt/";
pub const WSL_UNC_PREFIX: &str = "\\\\wsl$\\";

/// Web server defaults
pub const DEFAULT_HOST: &str = "0.0.0.0";
pub const DEFAULT_PORT: u16 = 8000;
pub const PORT_SCAN_START: u16 = 8000;
pub const PORT_SCAN_END: u16 = 8010;

/// Embedding defaults
pub const DEFAULT_EMBEDDING_MODEL: &str = "all-MiniLM-L6-v2";
pub const DEFAULT_EMBEDDING_BATCH_SIZE: usize = 32;

/// Text chunking
pub const DEFAULT_CHUNK_SIZE: usize = 1500;
pub const DEFAULT_CHUNK_OVERLAP: usize = 200;

/// Indexing defaults
pub const DEFAULT_INDEX_BATCH_SIZE: usize = 1000;
pub const DEFAULT_MAX_WORKERS: usize = 4;
pub const DEFAULT_AUTO_INDEX: bool = true;
pub const DEFAULT_INDEX_INTERVAL_MINUTES: u64 = 60;
pub const DEFAULT_REINDEX_ON_MODIFICATION: bool = true;
pub const DEFAULT_MODIFICATION_DEBOUNCE_MINUTES: u64 = 5;

/// Excluded prompt prefixes (5-element array)
/// These identify automated conversations that should not be indexed.
pub const DEFAULT_EXCLUDED_PROMPT_PREFIXES: [&str; 5] = [
    "Distill this conversation exchange into JSON",
    "Distill the conversation exchange below into JSON",
    "You are a strict relevance assessor",
    "Your task is to create a detailed summary of the conversation so far",
    "Your task is to create a detailed summary of the RECENT portion",
];

/// Search defaults
pub const DEFAULT_SEARCH_MODE: &str = "hybrid";
pub const DEFAULT_MAX_RESULTS: usize = 100;
pub const DEFAULT_SNIPPET_LENGTH: usize = 200;

/// Search ranking defaults
pub const DEFAULT_INTERSECTION_BOOST: f64 = 0.2;
pub const DEFAULT_PALACE_WEIGHT: f64 = 0.5;
pub const DEFAULT_VERBATIM_WEIGHT: f64 = 0.5;

/// Hybrid search tuning defaults
pub const DEFAULT_KEYWORD_WEIGHT: f64 = 0.6;
pub const DEFAULT_SEMANTIC_WEIGHT: f64 = 0.4;
pub const DEFAULT_RANK_DECAY: f64 = 0.1;
pub const DEFAULT_TITLE_BOOST: f64 = 2.0;
pub const DEFAULT_BM25_K1: f64 = 1.5;
pub const DEFAULT_BM25_B: f64 = 0.75;
pub const DEFAULT_BM25_CANDIDATES: usize = 500;
pub const DEFAULT_FAISS_K: usize = 100;

/// Performance defaults
pub const DEFAULT_MEMORY_LIMIT_MB: usize = 3000;
pub const DEFAULT_QUERY_CACHE_SIZE: usize = 100;
pub const DEFAULT_ENABLE_PROFILING: bool = false;
pub const DEFAULT_STARTUP_WARMUP_MODE: &str = "keyword";

/// UI defaults
pub const DEFAULT_THEME: &str = "auto";
pub const DEFAULT_FONT_FAMILY: &str = "Segoe UI";
pub const DEFAULT_FONT_SIZE: usize = 11;
pub const DEFAULT_HIGHLIGHT_COLOR: &str = "#FFEB3B";

/// Search engine default
pub const DEFAULT_SEARCH_ENGINE: &str = "unified";

/// Distillation defaults
pub const DEFAULT_DISTILLATION_PROVIDER: &str = "auto";
pub const DEFAULT_DISTILLATION_CLI_MODEL: &str = "claude-haiku-4-5-20251001";
pub const DEFAULT_DISTILLATION_CLI_MODEL_OPENAI: &str = "gpt-5.3-codex";
pub const DEFAULT_DISTILLATION_BATCH_SIZE: usize = 10;
pub const DEFAULT_DISTILLATION_MAX_PLY_LENGTH: usize = 20;
pub const DEFAULT_DISTILLATION_MIN_EXCHANGE_CHARS: usize = 50;

/// Batch distillation prompt (comprehensive, for batch processing with room assignments)
pub const DEFAULT_DISTILLATION_PROMPT: &str = r#"Distill this conversation exchange into JSON:

- "exchange_core": 1-2 sentences. What was accomplished or decided? Use the specific terms from the exchange. Do not invent details not present in the text. If the exchange is mostly empty, say so briefly.
- "specific_context": One concrete detail from the text: a number, error message, parameter name, or file path. Copy it exactly from the text. Do not use the project path.
- "room_assignments": 1-3 rooms. Each room is a topic this exchange belongs to. {"room_type": "<file|concept|workflow>", "room_key": "<identifier>", "room_label": "<short label>", "relevance": <0.0-1.0>}. A room should be specific enough to group related exchanges (e.g. "retry_timeout" not "errors").

Do NOT include "files_touched".

Project: {project_id}

Exchange (messages {ply_start}-{ply_end}):
{messages_text}

Respond with ONLY valid JSON."#;

/// Per-turn distillation prompt (simple, for real-time hook)
pub const DEFAULT_PERTURN_PROMPT: &str = r#"Distill this conversation exchange into JSON:

- "exchange_core": 1-2 sentences. What was accomplished or decided? Use specific terms from the text.
- "specific_context": One concrete detail: number, error message, parameter, or file path. Copy exactly.
- "tags": 2-4 keywords for retrieval (lowercase, underscore-separated).

User: {user_text}
Assistant: {assistant_text}

Respond with ONLY valid JSON."#;

/// Backfill defaults
pub const DEFAULT_BACKFILL_LLM_URL: &str = "http://localhost:8080";
pub const DEFAULT_BACKFILL_TIMEOUT: f64 = 180.0;
pub const DEFAULT_BACKFILL_BATCH_SIZE: usize = 16;

/// Backfill tier defaults
pub const DEFAULT_BACKFILL_TIER_SMALL_MAX_CHARS: usize = 12000;
pub const DEFAULT_BACKFILL_TIER_SMALL_CONCURRENT: usize = 8;
pub const DEFAULT_BACKFILL_TIER_MEDIUM_MAX_CHARS: usize = 48000;
pub const DEFAULT_BACKFILL_TIER_MEDIUM_CONCURRENT: usize = 4;
pub const DEFAULT_BACKFILL_TIER_LARGE_MAX_CHARS: usize = 120000;
pub const DEFAULT_BACKFILL_TIER_LARGE_CONCURRENT: usize = 2;
pub const DEFAULT_BACKFILL_TIER_HUGE_CONCURRENT: usize = 1;

// ============================================================================
// Environment Variable Name Constants
// ============================================================================

pub const ENV_DATA_DIR: &str = "SEARCHAT_DATA_DIR";
pub const ENV_WINDOWS_PROJECTS: &str = "SEARCHAT_WINDOWS_PROJECTS_DIR";
pub const ENV_WSL_PROJECTS: &str = "SEARCHAT_WSL_PROJECTS_DIR";
pub const ENV_ADDITIONAL_DIRS: &str = "SEARCHAT_ADDITIONAL_DIRS";

pub const ENV_PORT: &str = "SEARCHAT_PORT";
pub const ENV_HOST: &str = "SEARCHAT_HOST";

pub const ENV_AUTO_DETECT: &str = "SEARCHAT_AUTO_DETECT";
pub const ENV_EXCLUDED_CONVERSATIONS_DIR: &str = "SEARCHAT_EXCLUDED_CONVERSATIONS_DIR";

pub const ENV_INDEX_BATCH_SIZE: &str = "SEARCHAT_INDEX_BATCH_SIZE";
pub const ENV_AUTO_INDEX: &str = "SEARCHAT_AUTO_INDEX";
pub const ENV_INDEX_INTERVAL: &str = "SEARCHAT_INDEX_INTERVAL";
pub const ENV_MAX_WORKERS: &str = "SEARCHAT_MAX_WORKERS";
pub const ENV_REINDEX_ON_MODIFICATION: &str = "SEARCHAT_REINDEX_ON_MODIFICATION";
pub const ENV_MODIFICATION_DEBOUNCE_MINUTES: &str = "SEARCHAT_MODIFICATION_DEBOUNCE_MINUTES";

pub const ENV_DEFAULT_MODE: &str = "SEARCHAT_DEFAULT_MODE";
pub const ENV_MAX_RESULTS: &str = "SEARCHAT_MAX_RESULTS";
pub const ENV_SNIPPET_LENGTH: &str = "SEARCHAT_SNIPPET_LENGTH";

pub const ENV_MEMORY_LIMIT: &str = "SEARCHAT_MEMORY_LIMIT_MB";
pub const ENV_EMBEDDING_MODEL: &str = "SEARCHAT_EMBEDDING_MODEL";
pub const ENV_EMBEDDING_BATCH: &str = "SEARCHAT_EMBEDDING_BATCH_SIZE";
pub const ENV_CACHE_EMBEDDINGS: &str = "SEARCHAT_CACHE_EMBEDDINGS";
pub const ENV_EMBEDDING_DEVICE: &str = "SEARCHAT_EMBEDDING_DEVICE";
pub const ENV_CACHE_SIZE: &str = "SEARCHAT_QUERY_CACHE_SIZE";
pub const ENV_PROFILING: &str = "SEARCHAT_ENABLE_PROFILING";
pub const ENV_STARTUP_WARMUP_MODE: &str = "SEARCHAT_STARTUP_WARMUP_MODE";

pub const ENV_THEME: &str = "SEARCHAT_THEME";
pub const ENV_FONT_FAMILY: &str = "SEARCHAT_FONT_FAMILY";
pub const ENV_FONT_SIZE: &str = "SEARCHAT_FONT_SIZE";
pub const ENV_HIGHLIGHT_COLOR: &str = "SEARCHAT_HIGHLIGHT_COLOR";

pub const ENV_ISOLATION_MODE: &str = "SEARCHAT_ISOLATION_MODE";
pub const ENV_VARIANT_SUFFIX: &str = "SEARCHAT_VARIANT_SUFFIX";

/// Ranking env vars
pub const ENV_INTERSECTION_BOOST: &str = "SEARCHAT_INTERSECTION_BOOST";
pub const ENV_PALACE_WEIGHT: &str = "SEARCHAT_PALACE_WEIGHT";
pub const ENV_VERBATIM_WEIGHT: &str = "SEARCHAT_VERBATIM_WEIGHT";

/// Hybrid search tuning env vars
pub const ENV_KEYWORD_WEIGHT: &str = "SEARCHAT_KEYWORD_WEIGHT";
pub const ENV_SEMANTIC_WEIGHT: &str = "SEARCHAT_SEMANTIC_WEIGHT";
pub const ENV_RANK_DECAY: &str = "SEARCHAT_RANK_DECAY";
pub const ENV_TITLE_BOOST: &str = "SEARCHAT_TITLE_BOOST";
pub const ENV_BM25_K1: &str = "SEARCHAT_BM25_K1";
pub const ENV_BM25_B: &str = "SEARCHAT_BM25_B";
pub const ENV_BM25_CANDIDATES: &str = "SEARCHAT_BM25_CANDIDATES";
pub const ENV_FAISS_K: &str = "SEARCHAT_FAISS_K";

pub const ENV_SEARCH_ENGINE: &str = "SEARCHAT_SEARCH_ENGINE";

/// Distillation env vars
pub const ENV_DISTILLATION_PROVIDER: &str = "SEARCHAT_DISTILLATION_PROVIDER";
pub const ENV_DISTILLATION_CLI_MODEL: &str = "SEARCHAT_DISTILLATION_CLI_MODEL";
pub const ENV_DISTILLATION_BATCH_SIZE: &str = "SEARCHAT_DISTILLATION_BATCH_SIZE";
pub const ENV_DISTILLATION_MAX_PLY_LENGTH: &str = "SEARCHAT_DISTILLATION_MAX_PLY_LENGTH";
pub const ENV_DISTILLATION_MIN_EXCHANGE_CHARS: &str = "SEARCHAT_DISTILLATION_MIN_EXCHANGE_CHARS";

/// Backfill env vars
pub const ENV_BACKFILL_LLM_URL: &str = "SEARCHAT_BACKFILL_LLM_URL";
pub const ENV_BACKFILL_TIMEOUT: &str = "SEARCHAT_BACKFILL_TIMEOUT";
pub const ENV_BACKFILL_BATCH_SIZE: &str = "SEARCHAT_BACKFILL_BATCH_SIZE";
