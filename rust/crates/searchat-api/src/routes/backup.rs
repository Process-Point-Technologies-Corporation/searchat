use axum::{
    Json,
    extract::{Path, Query, State},
};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::error::ApiError;
use crate::state::AppState;

#[derive(Debug, Deserialize)]
pub struct CreateBackupParams {
    pub backup_name: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct RestoreBackupParams {
    pub backup_name: String,
}

/// POST /api/backup/create — backup is not implemented in the Rust server.
pub async fn create_backup(
    State(_state): State<AppState>,
    Query(_params): Query<CreateBackupParams>,
) -> ApiError {
    ApiError::ServiceUnavailable(
        "Backup not yet implemented in Rust server. Use the Python server.".to_string(),
    )
}

/// GET /api/backup/list — list available backups.
pub async fn list_backups(
    State(state): State<AppState>,
) -> Result<Json<Value>, ApiError> {
    let backup_dir = {
        let search_dir = &state.config.paths.search_directory;
        std::path::PathBuf::from(search_dir).join("backups")
    };

    if !backup_dir.exists() {
        return Ok(Json(json!({
            "backups": [],
            "total": 0,
            "backup_directory": backup_dir.to_string_lossy(),
        })));
    }

    let entries = std::fs::read_dir(&backup_dir)
        .map_err(|e| ApiError::Internal(e.to_string()))?;

    let mut backups: Vec<Value> = Vec::new();
    for entry in entries.flatten() {
        let metadata = entry.metadata().ok();
        let name = entry.file_name().to_string_lossy().to_string();
        let size = metadata.as_ref().map(|m| m.len()).unwrap_or(0);
        let modified = metadata
            .and_then(|m| m.modified().ok())
            .and_then(|t| {
                t.duration_since(std::time::UNIX_EPOCH).ok()
            })
            .map(|d| {
                chrono::DateTime::<chrono::Utc>::from_timestamp(d.as_secs() as i64, 0)
                    .map(|dt| dt.to_rfc3339())
                    .unwrap_or_default()
            })
            .unwrap_or_default();

        backups.push(json!({
            "backup_name": name,
            "backup_path": entry.path().to_string_lossy(),
            "created_at": modified,
            "total_size_bytes": size,
            "file_count": 0,
        }));
    }

    backups.sort_by(|a, b| {
        let at = a["created_at"].as_str().unwrap_or("");
        let bt = b["created_at"].as_str().unwrap_or("");
        bt.cmp(at)
    });

    let total = backups.len();
    Ok(Json(json!({
        "backups": backups,
        "total": total,
        "backup_directory": backup_dir.to_string_lossy(),
    })))
}

/// POST /api/backup/restore — not implemented.
pub async fn restore_backup(
    State(_state): State<AppState>,
    Json(_body): Json<RestoreBackupParams>,
) -> ApiError {
    ApiError::ServiceUnavailable(
        "Backup restore not yet implemented in Rust server. Use the Python server.".to_string(),
    )
}

/// DELETE /api/backup/delete/{name} — delete a backup by name.
pub async fn delete_backup(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let backup_dir = std::path::PathBuf::from(&state.config.paths.search_directory)
        .join("backups")
        .join(&name);

    if !backup_dir.exists() {
        return Err(ApiError::NotFound(format!("Backup not found: {name}")));
    }

    if backup_dir.is_dir() {
        std::fs::remove_dir_all(&backup_dir)
            .map_err(|e| ApiError::Internal(e.to_string()))?;
    } else {
        std::fs::remove_file(&backup_dir)
            .map_err(|e| ApiError::Internal(e.to_string()))?;
    }

    Ok(Json(json!({
        "success": true,
        "deleted": name,
        "message": format!("Backup deleted: {name}"),
    })))
}
