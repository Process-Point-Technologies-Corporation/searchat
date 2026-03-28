use axum::{Json, extract::State};
use serde_json::{Value, json};

use crate::error::ApiError;
use crate::state::AppState;

/// GET /api/statistics — index statistics from DuckDB.
pub async fn get_statistics(
    State(state): State<AppState>,
) -> Result<Json<Value>, ApiError> {
    let storage = state.storage.clone();

    let stats = tokio::task::spawn_blocking(move || {
        storage.get_stats()
    })
    .await
    .map_err(|e| ApiError::Internal(e.to_string()))?
    .map_err(|e| ApiError::Internal(e.to_string()))?;

    Ok(Json(json!({
        "total_conversations": stats.conversations,
        "total_messages": stats.messages,
        "total_exchanges": stats.exchanges,
        "verbatim_embeddings": stats.verbatim_embeddings,
        "total_palace_objects": stats.palace_objects,
        "total_rooms": stats.rooms,
        "facet_embeddings": stats.facet_embeddings,
        "hierarchical_facets": stats.hierarchical_facets,
        "vss_available": stats.vss_available,
        "fts_available": stats.fts_available,
    })))
}
