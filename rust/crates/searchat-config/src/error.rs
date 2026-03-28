use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("IO error: {0}")]
    Io(String),

    #[error("TOML parse error: {0}")]
    Parse(String),

    #[error("Missing required config: {0}")]
    Missing(String),
}
