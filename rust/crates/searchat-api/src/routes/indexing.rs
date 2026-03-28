use axum::{Json, extract::{Path, State}};
use serde_json::{Value, json};
use std::time::Instant;

use crate::error::ApiError;
use crate::state::AppState;

/// POST /api/reindex — always returns 403.
pub async fn reindex_blocked() -> ApiError {
    ApiError::Forbidden(
        "BLOCKED: Reindexing disabled to protect irreplaceable conversation data. \
         Source JSONLs are missing - rebuilding would cause data loss."
            .to_string(),
    )
}

/// POST /api/index_missing — index conversations not yet in the DB (append-only).
///
/// The full Python implementation scans the filesystem, detects new/changed
/// files, filters automated conversations, and calls the unified indexer.
/// The Rust indexer is not yet wired into the API, so this returns the current
/// DB state without modifying it.
pub async fn index_missing(
    State(state): State<AppState>,
) -> Result<Json<Value>, ApiError> {
    // Acquire the indexing lock (non-blocking).
    let _guard = state
        .indexing_lock
        .try_lock()
        .map_err(|_| ApiError::Conflict("Indexing already in progress".to_string()))?;

    let start = Instant::now();

    // Count currently indexed conversations as a baseline.
    let storage = state.storage.clone();
    let (total_conversations, total_exchanges) = tokio::task::spawn_blocking(move || {
        let stats = storage.get_stats()?;
        Ok::<_, searchat_storage::StorageError>((stats.conversations, stats.exchanges))
    })
    .await
    .map_err(|e| ApiError::Internal(e.to_string()))?
    .map_err(|e| ApiError::Internal(e.to_string()))?;

    let elapsed = start.elapsed().as_secs_f64();

    Ok(Json(json!({
        "success": true,
        "new_conversations": 0,
        "updated_conversations": 0,
        "changed_detected": 0,
        "failed_conversations": 0,
        "total_files": 0,
        "already_indexed": total_conversations,
        "excluded_automated": 0,
        "time_seconds": elapsed,
        "message": format!(
            "Rust indexer not yet wired. DB contains {} conversations, {} exchanges.",
            total_conversations, total_exchanges
        ),
    })))
}

/// POST /api/distill — distill all pending conversations.
///
/// Palace distillation is not implemented in the Rust server yet.
pub async fn distill_pending() -> ApiError {
    ApiError::ServiceUnavailable(
        "Distillation not yet implemented in Rust server. Use the Python server.".to_string(),
    )
}

/// POST /api/distill/{conversation_id} — distill a single conversation.
pub async fn distill_conversation(
    Path(_conversation_id): Path<String>,
) -> ApiError {
    ApiError::ServiceUnavailable(
        "Distillation not yet implemented in Rust server. Use the Python server.".to_string(),
    )
}
