use thiserror::Error;

#[derive(Debug, Error)]
pub enum SearchError {
    #[error("embedding failed: {0}")]
    Embedding(String),

    #[error("storage error: {0}")]
    Storage(String),

    #[error("query parse error: {0}")]
    QueryParse(String),

    #[error("search failed: {0}")]
    Search(String),
}
