use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum EmbedError {
    #[error("ONNX Runtime error: {0}")]
    Ort(#[from] ort::Error),
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),
    #[error("Model file not found: {0}")]
    ModelNotFound(PathBuf),
    #[error("Invalid output shape")]
    InvalidShape,
}
