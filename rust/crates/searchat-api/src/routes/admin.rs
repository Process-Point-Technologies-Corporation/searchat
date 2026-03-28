use axum::{Json, extract::State};
use serde_json::{Value, json};

use crate::state::AppState;

/// GET /api/watcher/status — live file watcher status.
pub async fn watcher_status(
    State(state): State<AppState>,
) -> Json<Value> {
    let running = *state.watcher_running.read();
    let indexed = *state.indexed_since_start.read();
    let last_update = state.last_watcher_update.read().clone();

    Json(json!({
        "running": running,
        "watched_directories": [],
        "indexed_since_start": indexed,
        "last_update": last_update,
    }))
}

/// GET /api/indexing/status — current indexing operation status.
pub async fn indexing_status(
    State(state): State<AppState>,
) -> Json<Value> {
    // Indexing lock is held when an operation is in progress.
    let in_progress = state.indexing_lock.try_lock().is_err();

    Json(json!({
        "in_progress": in_progress,
        "operation": null,
        "started_at": null,
        "files_total": 0,
        "files_processed": 0,
    }))
}

/// POST /api/shutdown — gracefully shut down the server.
pub async fn shutdown_server(
    State(_state): State<AppState>,
) -> Json<Value> {
    // Spawn a task to exit after returning the response.
    tokio::spawn(async {
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        std::process::exit(0);
    });

    Json(json!({
        "success": true,
        "message": "Server shutting down",
    }))
}
