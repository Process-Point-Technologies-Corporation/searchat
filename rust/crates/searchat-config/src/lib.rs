pub mod constants;
pub mod error;
pub mod path_resolver;
pub mod settings;

pub use constants::*;
pub use error::ConfigError;
pub use path_resolver::{
    Platform,
    default_claude_projects_dir,
    default_codex_sessions_dir,
    default_config_dir,
    default_data_dir,
    default_logs_dir,
    default_storage_dir,
    detect_platform,
    ensure_directory,
    expand_path_template,
    is_wsl,
    resolve_claude_dirs,
    resolve_codex_dirs,
    resolve_vibe_dirs,
    safe_exists,
    safe_resolve,
    translate_path,
};
pub use settings::{
    BackfillConfig,
    BackfillTier,
    Config,
    DistillationConfig,
    EmbeddingConfig,
    IndexingConfig,
    PathsConfig,
    PerformanceConfig,
    RankingConfig,
    SearchConfig,
    UIConfig,
};
