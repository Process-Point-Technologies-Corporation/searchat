use thiserror::Error;

#[derive(Debug, Error)]
pub enum StorageError {
    #[error("DuckDB error: {0}")]
    DuckDb(#[from] duckdb::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Room not found: {0}")]
    RoomNotFound(String),

    #[error("Palace object not found: {0}")]
    ObjectNotFound(String),

    #[error("Extension not available: {0}")]
    ExtensionUnavailable(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type StorageResult<T> = Result<T, StorageError>;
