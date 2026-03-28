use axum::{
    Json,
    extract::{Path, Query, State},
};
use chrono::{Duration, NaiveDate, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::error::ApiError;
use crate::state::AppState;

// ---------------------------------------------------------------------------
// Query params
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct ConversationsAllParams {
    #[serde(default = "default_sort_length")]
    pub sort_by: String,
    pub project: Option<String>,
    pub date: Option<String>,
    pub date_from: Option<String>,
    pub date_to: Option<String>,
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default)]
    pub offset: usize,
}

fn default_sort_length() -> String { "length".to_string() }
fn default_limit() -> usize { 50 }

#[derive(Debug, Deserialize)]
pub struct ResumeRequest {
    pub conversation_id: String,
}

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct ConversationListItem {
    pub conversation_id: String,
    pub project_id: String,
    pub title: String,
    pub created_at: String,
    pub updated_at: String,
    pub message_count: i64,
    pub file_path: String,
    pub snippet: String,
    pub score: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message_start_index: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message_end_index: Option<i64>,
    pub source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ConversationMessage {
    pub role: String,
    pub content: String,
    pub timestamp: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ply_index: Option<i64>,
}

#[derive(Debug, Serialize)]
pub struct ConversationResponse {
    pub conversation_id: String,
    pub title: String,
    pub project_id: String,
    pub file_path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool: Option<String>,
    pub message_count: usize,
    pub messages: Vec<ConversationMessage>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn detect_source(file_path: &str) -> &'static str {
    let fp = file_path.to_lowercase();
    if fp.contains("/home/") || fp.contains("wsl") { "WSL" } else { "WIN" }
}

fn detect_tool(file_path: &str) -> Option<String> {
    let n = file_path.replace('\\', "/").to_lowercase();
    if n.contains("/.codex/") {
        Some("codex".to_string())
    } else if n.contains("/.vibe/") {
        Some("vibe".to_string())
    } else {
        Some("claude".to_string())
    }
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// GET /api/conversations/all — paginated conversation listing.
pub async fn get_all_conversations(
    State(state): State<AppState>,
    Query(params): Query<ConversationsAllParams>,
) -> Result<Json<Value>, ApiError> {
    let limit = params.limit.min(200).max(1);
    let offset = params.offset;

    let storage = state.storage.clone();
    let sort_by = params.sort_by.clone();
    let project = params.project.clone();
    let date = params.date.clone();
    let date_from = params.date_from.clone();
    let date_to = params.date_to.clone();

    let (rows, total_count) = tokio::task::spawn_blocking(move || {
        query_conversations(&storage, &sort_by, project.as_deref(), date.as_deref(),
            date_from.as_deref(), date_to.as_deref(), limit, offset)
    })
    .await
    .map_err(|e| ApiError::Internal(e.to_string()))??;

    let has_more = offset + limit < total_count as usize;

    let response_results: Vec<ConversationListItem> = rows
        .into_iter()
        .map(|(conv_id, proj_id, title, created_at, updated_at, msg_count, fp)| {
            ConversationListItem {
                conversation_id: conv_id,
                project_id: proj_id,
                title,
                created_at,
                updated_at,
                message_count: msg_count,
                file_path: fp.clone(),
                snippet: String::new(),
                score: 0.0,
                message_start_index: None,
                message_end_index: None,
                source: detect_source(&fp).to_string(),
                tool: detect_tool(&fp),
            }
        })
        .collect();

    Ok(Json(json!({
        "results": response_results,
        "total": total_count,
        "limit": limit,
        "offset": offset,
        "has_more": has_more,
        "search_time_ms": 0,
    })))
}

type ConvRow = (String, String, String, String, String, i64, String);

fn query_conversations(
    storage: &searchat_storage::UnifiedStorage,
    sort_by: &str,
    project: Option<&str>,
    date: Option<&str>,
    date_from: Option<&str>,
    date_to: Option<&str>,
    limit: usize,
    offset: usize,
) -> Result<(Vec<ConvRow>, i64), ApiError> {
    // Get all conversation IDs with optional project filter.
    let all_ids = storage
        .get_all_conversation_ids(project)
        .map_err(|e| ApiError::Internal(e.to_string()))?;

    // Fetch conversation details.
    let conv_map = storage
        .get_conversations_batch(&all_ids)
        .map_err(|e| ApiError::Internal(e.to_string()))?;

    // Apply date filters.
    let date_from_dt = date_from.and_then(|s| s.parse::<NaiveDate>().ok())
        .map(|d| d.and_hms_opt(0, 0, 0).unwrap().and_utc());
    let date_to_dt = date_to.and_then(|s| s.parse::<NaiveDate>().ok())
        .map(|d| (d.and_hms_opt(0, 0, 0).unwrap().and_utc()) + Duration::days(1));

    let now = Utc::now();
    let (effective_from, effective_to): (Option<chrono::DateTime<Utc>>, Option<chrono::DateTime<Utc>>) =
        match date {
            Some("custom") => (date_from_dt, date_to_dt),
            Some("today") => (
                Some(now.date_naive().and_hms_opt(0, 0, 0).unwrap().and_utc()),
                Some(now),
            ),
            Some("week") => (Some(now - Duration::days(7)), Some(now)),
            Some("month") => (Some(now - Duration::days(30)), Some(now)),
            _ => (None, None),
        };

    let mut filtered: Vec<&searchat_storage::ConversationRow> = conv_map
        .values()
        .filter(|c| c.message_count > 0)
        .filter(|c| {
            if let Some(from) = effective_from {
                if c.updated_at < from {
                    return false;
                }
            }
            if let Some(to) = effective_to {
                if c.updated_at >= to {
                    return false;
                }
            }
            true
        })
        .collect();

    // Sort.
    match sort_by {
        "date_newest" => filtered.sort_by(|a, b| b.updated_at.cmp(&a.updated_at)),
        "date_oldest" => filtered.sort_by(|a, b| a.updated_at.cmp(&b.updated_at)),
        "title" => filtered.sort_by(|a, b| a.title.cmp(&b.title)),
        _ => filtered.sort_by(|a, b| b.message_count.cmp(&a.message_count)), // length
    }

    let total_count = filtered.len() as i64;

    let page: Vec<ConvRow> = filtered
        .into_iter()
        .skip(offset)
        .take(limit)
        .map(|c| {
            (
                c.conversation_id.clone(),
                c.project_id.clone(),
                c.title.clone(),
                c.created_at.to_rfc3339(),
                c.updated_at.to_rfc3339(),
                c.message_count as i64,
                c.file_path.clone(),
            )
        })
        .collect();

    Ok((page, total_count))
}

/// GET /api/conversation/{id} — full conversation with messages.
pub async fn get_conversation(
    State(state): State<AppState>,
    Path(conversation_id): Path<String>,
) -> Result<Json<ConversationResponse>, ApiError> {
    let storage = state.storage.clone();
    let conv_id = conversation_id.clone();

    let (conv, messages) = tokio::task::spawn_blocking(move || {
        let conv = storage
            .get_conversation(&conv_id)
            .map_err(|e| ApiError::Internal(e.to_string()))?
            .ok_or_else(|| ApiError::NotFound(format!("Conversation not found: {conv_id}")))?;

        let msg_rows = storage
            .get_conversation_messages(&conv_id)
            .map_err(|e| ApiError::Internal(e.to_string()))?;

        Ok::<_, ApiError>((conv, msg_rows))
    })
    .await
    .map_err(|e| ApiError::Internal(e.to_string()))??;

    let file_path = conv.file_path.clone();
    if !std::path::Path::new(&file_path).exists() {
        return Err(ApiError::NotFound(format!(
            "Conversation file not found. The file may have been moved or deleted: {file_path}"
        )));
    }

    let tool = detect_tool(&file_path);
    let message_count = messages.len();

    let msgs: Vec<ConversationMessage> = messages
        .into_iter()
        .map(|m| ConversationMessage {
            role: m.role,
            content: m.content,
            timestamp: m.timestamp
                .map(|t| t.to_rfc3339())
                .unwrap_or_else(|| Utc::now().to_rfc3339()),
            ply_index: Some(m.sequence as i64),
        })
        .collect();

    Ok(Json(ConversationResponse {
        conversation_id: conversation_id,
        title: conv.title,
        project_id: conv.project_id,
        file_path: conv.file_path,
        tool,
        message_count,
        messages: msgs,
    }))
}

/// POST /api/resume — open a terminal to resume a conversation session.
pub async fn resume_session(
    State(_state): State<AppState>,
    Json(body): Json<ResumeRequest>,
) -> Result<Json<Value>, ApiError> {
    // Platform-specific terminal launch is not yet implemented in Rust.
    // Return a structured response indicating what would be launched.
    Ok(Json(json!({
        "success": false,
        "conversation_id": body.conversation_id,
        "message": "Resume not yet implemented in Rust server. Use the Python server for this feature.",
    })))
}
