use serde::{Deserialize, Serialize};
use std::env;

use crate::constants::*;

// ============================================================================
// Env var helpers
// ============================================================================

/// Returns Some(val) if the env var is set and non-empty, otherwise None.
fn env_str(key: &str) -> Option<String> {
    match env::var(key) {
        Ok(v) if !v.is_empty() => Some(v),
        _ => None,
    }
}

/// Returns the env var parsed as `T`, falling back to `default` on missing or parse failure.
fn env_int<T>(key: &str, default: T) -> T
where
    T: std::str::FromStr + Copy,
{
    match env::var(key) {
        Ok(v) => v.trim().parse::<T>().unwrap_or(default),
        Err(_) => default,
    }
}

fn env_bool(key: &str, default: bool) -> bool {
    match env::var(key) {
        Ok(v) => match v.trim().to_lowercase().as_str() {
            "true" | "1" | "yes" | "on" => true,
            "false" | "0" | "no" | "off" => false,
            _ => {
                log::warn!("Invalid boolean for {}: {:?}. Using default {:?}.", key, v, default);
                default
            }
        },
        Err(_) => default,
    }
}

fn env_float(key: &str, default: f64) -> f64 {
    match env::var(key) {
        Ok(v) => v.trim().parse::<f64>().unwrap_or(default),
        Err(_) => default,
    }
}

// ============================================================================
// PathsConfig
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PathsConfig {
    pub claude_directory_windows: String,
    pub claude_directory_wsl: String,
    pub search_directory: String,
    pub auto_detect_environment: bool,
    pub excluded_conversations_dir: String,
}

impl Default for PathsConfig {
    fn default() -> Self {
        let home = dirs::home_dir()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_default();
        Self {
            claude_directory_windows: format!("C:/Users/{{username}}/{}", CLAUDE_DIR_NAME),
            claude_directory_wsl: String::new(),
            search_directory: format!("{}/{}", home, CONFIG_DIR_NAME),
            auto_detect_environment: true,
            excluded_conversations_dir: String::new(),
        }
    }
}

impl PathsConfig {
    fn apply_env_overrides(mut self) -> Self {
        if let Some(v) = env_str(ENV_WINDOWS_PROJECTS) {
            self.claude_directory_windows = v;
        }
        if let Some(v) = env_str(ENV_WSL_PROJECTS) {
            self.claude_directory_wsl = v;
        }
        if let Some(v) = env_str(ENV_DATA_DIR) {
            self.search_directory = v;
        }
        self.auto_detect_environment = env_bool(ENV_AUTO_DETECT, self.auto_detect_environment);
        if let Some(v) = env_str(ENV_EXCLUDED_CONVERSATIONS_DIR) {
            self.excluded_conversations_dir = v;
        }
        self
    }
}

// ============================================================================
// IndexingConfig
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct IndexingConfig {
    pub batch_size: usize,
    pub auto_index: bool,
    pub index_interval_minutes: u64,
    pub max_workers: usize,
    pub reindex_on_modification: bool,
    pub modification_debounce_minutes: u64,
    pub excluded_prompt_prefixes: Vec<String>,
}

impl Default for IndexingConfig {
    fn default() -> Self {
        Self {
            batch_size: DEFAULT_INDEX_BATCH_SIZE,
            auto_index: DEFAULT_AUTO_INDEX,
            index_interval_minutes: DEFAULT_INDEX_INTERVAL_MINUTES,
            max_workers: DEFAULT_MAX_WORKERS,
            reindex_on_modification: DEFAULT_REINDEX_ON_MODIFICATION,
            modification_debounce_minutes: DEFAULT_MODIFICATION_DEBOUNCE_MINUTES,
            excluded_prompt_prefixes: DEFAULT_EXCLUDED_PROMPT_PREFIXES
                .iter()
                .map(|s| s.to_string())
                .collect(),
        }
    }
}

impl IndexingConfig {
    fn apply_env_overrides(mut self) -> Self {
        self.batch_size = env_int(ENV_INDEX_BATCH_SIZE, self.batch_size);
        self.auto_index = env_bool(ENV_AUTO_INDEX, self.auto_index);
        self.index_interval_minutes = env_int(ENV_INDEX_INTERVAL, self.index_interval_minutes);
        self.max_workers = env_int(ENV_MAX_WORKERS, self.max_workers);
        self.reindex_on_modification =
            env_bool(ENV_REINDEX_ON_MODIFICATION, self.reindex_on_modification);
        self.modification_debounce_minutes = env_int(
            ENV_MODIFICATION_DEBOUNCE_MINUTES,
            self.modification_debounce_minutes,
        );
        self
    }
}

// ============================================================================
// RankingConfig
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RankingConfig {
    pub intersection_boost: f64,
    pub palace_weight: f64,
    pub verbatim_weight: f64,
    pub keyword_weight: f64,
    pub semantic_weight: f64,
    pub rank_decay: f64,
    pub title_boost: f64,
    pub bm25_k1: f64,
    pub bm25_b: f64,
    pub bm25_candidates: usize,
    pub faiss_k: usize,
}

impl Default for RankingConfig {
    fn default() -> Self {
        Self {
            intersection_boost: DEFAULT_INTERSECTION_BOOST,
            palace_weight: DEFAULT_PALACE_WEIGHT,
            verbatim_weight: DEFAULT_VERBATIM_WEIGHT,
            keyword_weight: DEFAULT_KEYWORD_WEIGHT,
            semantic_weight: DEFAULT_SEMANTIC_WEIGHT,
            rank_decay: DEFAULT_RANK_DECAY,
            title_boost: DEFAULT_TITLE_BOOST,
            bm25_k1: DEFAULT_BM25_K1,
            bm25_b: DEFAULT_BM25_B,
            bm25_candidates: DEFAULT_BM25_CANDIDATES,
            faiss_k: DEFAULT_FAISS_K,
        }
    }
}

impl RankingConfig {
    /// Convert percentage boost to multiplier (0.2 -> 1.2).
    pub fn boost_multiplier(&self) -> f64 {
        1.0 + self.intersection_boost
    }

    /// Palace weight scaled so max intersection score = 1.0.
    pub fn scaled_palace_weight(&self) -> f64 {
        self.palace_weight / self.boost_multiplier()
    }

    /// Verbatim weight scaled so max intersection score = 1.0.
    pub fn scaled_verbatim_weight(&self) -> f64 {
        self.verbatim_weight / self.boost_multiplier()
    }

    fn apply_env_overrides(mut self) -> Self {
        self.intersection_boost = env_float(ENV_INTERSECTION_BOOST, self.intersection_boost);
        self.palace_weight = env_float(ENV_PALACE_WEIGHT, self.palace_weight);
        self.verbatim_weight = env_float(ENV_VERBATIM_WEIGHT, self.verbatim_weight);
        self.keyword_weight = env_float(ENV_KEYWORD_WEIGHT, self.keyword_weight);
        self.semantic_weight = env_float(ENV_SEMANTIC_WEIGHT, self.semantic_weight);
        self.rank_decay = env_float(ENV_RANK_DECAY, self.rank_decay);
        self.title_boost = env_float(ENV_TITLE_BOOST, self.title_boost);
        self.bm25_k1 = env_float(ENV_BM25_K1, self.bm25_k1);
        self.bm25_b = env_float(ENV_BM25_B, self.bm25_b);
        self.bm25_candidates = env_int(ENV_BM25_CANDIDATES, self.bm25_candidates);
        self.faiss_k = env_int(ENV_FAISS_K, self.faiss_k);
        self
    }
}

// ============================================================================
// SearchConfig
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SearchConfig {
    pub default_mode: String,
    pub max_results: usize,
    pub snippet_length: usize,
    pub ranking: RankingConfig,
    pub engine: String,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            default_mode: DEFAULT_SEARCH_MODE.to_string(),
            max_results: DEFAULT_MAX_RESULTS,
            snippet_length: DEFAULT_SNIPPET_LENGTH,
            ranking: RankingConfig::default(),
            engine: DEFAULT_SEARCH_ENGINE.to_string(),
        }
    }
}

impl SearchConfig {
    fn apply_env_overrides(mut self) -> Self {
        if let Some(v) = env_str(ENV_DEFAULT_MODE) {
            self.default_mode = v;
        }
        self.max_results = env_int(ENV_MAX_RESULTS, self.max_results);
        self.snippet_length = env_int(ENV_SNIPPET_LENGTH, self.snippet_length);
        self.ranking = self.ranking.apply_env_overrides();
        if let Some(v) = env_str(ENV_SEARCH_ENGINE) {
            self.engine = v;
        }
        self
    }
}

// ============================================================================
// EmbeddingConfig
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EmbeddingConfig {
    pub model: String,
    pub batch_size: usize,
    pub cache_embeddings: bool,
    /// "auto", "cuda", "mps", or "cpu"
    pub device: String,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model: DEFAULT_EMBEDDING_MODEL.to_string(),
            batch_size: DEFAULT_EMBEDDING_BATCH_SIZE,
            cache_embeddings: true,
            device: "auto".to_string(),
        }
    }
}

impl EmbeddingConfig {
    fn apply_env_overrides(mut self) -> Self {
        if let Some(v) = env_str(ENV_EMBEDDING_MODEL) {
            self.model = v;
        }
        self.batch_size = env_int(ENV_EMBEDDING_BATCH, self.batch_size);
        self.cache_embeddings = env_bool(ENV_CACHE_EMBEDDINGS, self.cache_embeddings);
        if let Some(v) = env_str(ENV_EMBEDDING_DEVICE) {
            self.device = v;
        }
        self
    }
}

// ============================================================================
// UIConfig
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct UIConfig {
    pub theme: String,
    pub font_family: String,
    pub font_size: usize,
    pub highlight_color: String,
}

impl Default for UIConfig {
    fn default() -> Self {
        Self {
            theme: DEFAULT_THEME.to_string(),
            font_family: DEFAULT_FONT_FAMILY.to_string(),
            font_size: DEFAULT_FONT_SIZE,
            highlight_color: DEFAULT_HIGHLIGHT_COLOR.to_string(),
        }
    }
}

impl UIConfig {
    fn apply_env_overrides(mut self) -> Self {
        if let Some(v) = env_str(ENV_THEME) {
            self.theme = v;
        }
        if let Some(v) = env_str(ENV_FONT_FAMILY) {
            self.font_family = v;
        }
        self.font_size = env_int(ENV_FONT_SIZE, self.font_size);
        if let Some(v) = env_str(ENV_HIGHLIGHT_COLOR) {
            self.highlight_color = v;
        }
        self
    }
}

// ============================================================================
// PerformanceConfig
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PerformanceConfig {
    pub memory_limit_mb: usize,
    pub query_cache_size: usize,
    pub enable_profiling: bool,
    pub startup_warmup_mode: String,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            memory_limit_mb: DEFAULT_MEMORY_LIMIT_MB,
            query_cache_size: DEFAULT_QUERY_CACHE_SIZE,
            enable_profiling: DEFAULT_ENABLE_PROFILING,
            startup_warmup_mode: DEFAULT_STARTUP_WARMUP_MODE.to_string(),
        }
    }
}

impl PerformanceConfig {
    fn apply_env_overrides(mut self) -> Self {
        self.memory_limit_mb = env_int(ENV_MEMORY_LIMIT, self.memory_limit_mb);
        self.query_cache_size = env_int(ENV_CACHE_SIZE, self.query_cache_size);
        self.enable_profiling = env_bool(ENV_PROFILING, self.enable_profiling);
        if let Some(v) = env_str(ENV_STARTUP_WARMUP_MODE) {
            self.startup_warmup_mode = v;
        }
        self
    }
}

// ============================================================================
// DistillationConfig
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DistillationConfig {
    pub provider: String,
    pub cli_model: String,
    pub batch_size: usize,
    pub max_ply_length: usize,
    pub min_exchange_chars: usize,
    pub prompt: String,
    pub perturn_prompt: String,
}

impl Default for DistillationConfig {
    fn default() -> Self {
        Self {
            provider: DEFAULT_DISTILLATION_PROVIDER.to_string(),
            cli_model: DEFAULT_DISTILLATION_CLI_MODEL.to_string(),
            batch_size: DEFAULT_DISTILLATION_BATCH_SIZE,
            max_ply_length: DEFAULT_DISTILLATION_MAX_PLY_LENGTH,
            min_exchange_chars: DEFAULT_DISTILLATION_MIN_EXCHANGE_CHARS,
            prompt: DEFAULT_DISTILLATION_PROMPT.to_string(),
            perturn_prompt: DEFAULT_PERTURN_PROMPT.to_string(),
        }
    }
}

impl DistillationConfig {
    fn apply_env_overrides(mut self) -> Self {
        if let Some(v) = env_str(ENV_DISTILLATION_PROVIDER) {
            let normalized = v.trim().to_lowercase();
            if ["claude", "openai", "auto"].contains(&normalized.as_str()) {
                self.provider = normalized;
            } else {
                log::warn!(
                    "Invalid distillation provider {:?}. Using default {:?}.",
                    v,
                    DEFAULT_DISTILLATION_PROVIDER
                );
            }
        }
        if let Some(v) = env_str(ENV_DISTILLATION_CLI_MODEL) {
            self.cli_model = v;
        }
        self.batch_size = env_int(ENV_DISTILLATION_BATCH_SIZE, self.batch_size);
        self.max_ply_length = env_int(ENV_DISTILLATION_MAX_PLY_LENGTH, self.max_ply_length);
        self.min_exchange_chars =
            env_int(ENV_DISTILLATION_MIN_EXCHANGE_CHARS, self.min_exchange_chars);
        self
    }
}

// ============================================================================
// BackfillTier / BackfillConfig
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackfillTier {
    pub name: String,
    /// Max exchange size in chars for this tier. None means unbounded (last tier).
    pub max_chars: Option<usize>,
    pub max_concurrent: usize,
}

impl BackfillTier {
    fn default_tiers() -> Vec<Self> {
        vec![
            BackfillTier {
                name: "small".to_string(),
                max_chars: Some(DEFAULT_BACKFILL_TIER_SMALL_MAX_CHARS),
                max_concurrent: DEFAULT_BACKFILL_TIER_SMALL_CONCURRENT,
            },
            BackfillTier {
                name: "medium".to_string(),
                max_chars: Some(DEFAULT_BACKFILL_TIER_MEDIUM_MAX_CHARS),
                max_concurrent: DEFAULT_BACKFILL_TIER_MEDIUM_CONCURRENT,
            },
            BackfillTier {
                name: "large".to_string(),
                max_chars: Some(DEFAULT_BACKFILL_TIER_LARGE_MAX_CHARS),
                max_concurrent: DEFAULT_BACKFILL_TIER_LARGE_CONCURRENT,
            },
            BackfillTier {
                name: "huge".to_string(),
                max_chars: None,
                max_concurrent: DEFAULT_BACKFILL_TIER_HUGE_CONCURRENT,
            },
        ]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct BackfillConfig {
    pub llm_url: String,
    pub timeout: f64,
    pub batch_size: usize,
    pub tiers: Vec<BackfillTier>,
}

impl Default for BackfillConfig {
    fn default() -> Self {
        Self {
            llm_url: DEFAULT_BACKFILL_LLM_URL.to_string(),
            timeout: DEFAULT_BACKFILL_TIMEOUT,
            batch_size: DEFAULT_BACKFILL_BATCH_SIZE,
            tiers: BackfillTier::default_tiers(),
        }
    }
}

impl BackfillConfig {
    /// Returns the tier appropriate for a given text length.
    pub fn tier_for_size(&self, text_len: usize) -> &BackfillTier {
        for tier in &self.tiers {
            match tier.max_chars {
                Some(max) if text_len <= max => return tier,
                None => return tier,
                _ => {}
            }
        }
        self.tiers.last().expect("tiers must not be empty")
    }

    fn apply_env_overrides(mut self) -> Self {
        if let Some(v) = env_str(ENV_BACKFILL_LLM_URL) {
            self.llm_url = v;
        }
        self.timeout = env_float(ENV_BACKFILL_TIMEOUT, self.timeout);
        self.batch_size = env_int(ENV_BACKFILL_BATCH_SIZE, self.batch_size);
        self
    }
}

// ============================================================================
// Top-level Config
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub paths: PathsConfig,
    pub indexing: IndexingConfig,
    pub search: SearchConfig,
    pub embedding: EmbeddingConfig,
    pub ui: UIConfig,
    pub performance: PerformanceConfig,
    pub distillation: DistillationConfig,
    pub backfill: BackfillConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            paths: PathsConfig::default(),
            indexing: IndexingConfig::default(),
            search: SearchConfig::default(),
            embedding: EmbeddingConfig::default(),
            ui: UIConfig::default(),
            performance: PerformanceConfig::default(),
            distillation: DistillationConfig::default(),
            backfill: BackfillConfig::default(),
        }
    }
}

impl Config {
    /// Load configuration with the following precedence (highest → lowest):
    /// 1. Environment variables (SEARCHAT_*)
    /// 2. User config (`~/.searchat/config/settings.toml`)
    /// 3. Hardcoded defaults from constants.rs
    pub fn load() -> Result<Self, crate::error::ConfigError> {
        Self::load_from(None)
    }

    /// Load from an explicit path, or fall back to the standard user config location.
    pub fn load_from(
        config_path: Option<std::path::PathBuf>,
    ) -> Result<Self, crate::error::ConfigError> {
        // Load .env files before reading env vars
        crate::settings::load_env_files();

        let toml_text = if let Some(ref path) = config_path {
            // Explicit path — error if missing
            std::fs::read_to_string(path).map_err(|e| {
                crate::error::ConfigError::Io(format!("{}: {}", path.display(), e))
            })?
        } else {
            // Standard user config location
            let user_config = crate::path_resolver::default_config_dir().join(SETTINGS_FILE);
            if user_config.exists() {
                std::fs::read_to_string(&user_config).map_err(|e| {
                    crate::error::ConfigError::Io(format!("{}: {}", user_config.display(), e))
                })?
            } else {
                // No config file — use pure defaults
                String::new()
            }
        };

        let config: Config = if toml_text.is_empty() {
            Config::default()
        } else {
            toml::from_str(&toml_text)
                .map_err(|e| crate::error::ConfigError::Parse(e.to_string()))?
        };

        // Apply env var overrides on top of file values
        Ok(config.apply_env_overrides())
    }

    fn apply_env_overrides(self) -> Self {
        Self {
            paths: self.paths.apply_env_overrides(),
            indexing: self.indexing.apply_env_overrides(),
            search: self.search.apply_env_overrides(),
            embedding: self.embedding.apply_env_overrides(),
            ui: self.ui.apply_env_overrides(),
            performance: self.performance.apply_env_overrides(),
            distillation: self.distillation.apply_env_overrides(),
            backfill: self.backfill.apply_env_overrides(),
        }
    }
}

/// Load .env files from standard locations (called once before env var reads).
pub(crate) fn load_env_files() {
    let locations: Vec<std::path::PathBuf> = {
        let home = dirs::home_dir().unwrap_or_default();
        let data_dir = home.join(CONFIG_DIR_NAME);
        vec![
            std::path::PathBuf::from(ENV_FILE),               // ./env
            data_dir.join(ENV_FILE),                          // ~/.searchat/.env
            data_dir.join(DEFAULT_CONFIG_SUBDIR).join(ENV_FILE), // ~/.searchat/config/.env
        ]
    };

    for path in &locations {
        if path.exists() {
            // override=false: don't stomp vars already set in the process environment
            let _ = dotenvy::from_path_override(path);
        }
    }
}
