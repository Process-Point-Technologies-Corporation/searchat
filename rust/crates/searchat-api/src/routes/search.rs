use axum::{
    Json,
    extract::{Query, State},
};
use chrono::{Duration, NaiveDate, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::time::Instant;

use searchat_models::{AlgorithmType, SearchFilters, SearchResult};
use searchat_storage::UnifiedStorage;

use crate::error::ApiError;
use crate::state::AppState;

// ---------------------------------------------------------------------------
// Query parameter structs
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct SearchParams {
    pub q: String,
    #[serde(default = "default_cross_layer_mode")]
    pub mode: String,
    pub project: Option<String>,
    pub date: Option<String>,
    pub date_from: Option<String>,
    pub date_to: Option<String>,
    #[serde(default = "default_sort_relevance")]
    pub sort_by: String,
    #[serde(default = "default_limit_100")]
    pub limit: usize,
}

#[derive(Debug, Deserialize)]
pub struct UnifiedSearchParams {
    pub q: String,
    #[serde(default = "default_distill_mode")]
    pub mode: String,
    pub project: Option<String>,
    pub date: Option<String>,
    pub date_from: Option<String>,
    pub date_to: Option<String>,
    #[serde(default = "default_limit_50")]
    pub limit: usize,
}

fn default_cross_layer_mode() -> String { "cross-layer".to_string() }
fn default_distill_mode() -> String { "distill".to_string() }
fn default_sort_relevance() -> String { "relevance".to_string() }
fn default_limit_100() -> usize { 100 }
fn default_limit_50() -> usize { 50 }

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct SearchResultResponse {
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bm25_score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub semantic_score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exchange_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exchange_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub match_source: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct UnifiedSearchResultResponse {
    pub conversation_id: String,
    pub project_id: String,
    pub title: String,
    pub created_at: String,
    pub updated_at: String,
    pub message_count: i64,
    pub file_path: String,
    pub combined_score: f64,
    pub source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool: Option<String>,
    // Palace layer
    #[serde(skip_serializing_if = "Option::is_none")]
    pub palace_score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub palace_summary: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub palace_context: Option<String>,
    pub rooms: Vec<Value>,
    pub files_touched: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ply_start: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ply_end: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub object_id: Option<String>,
    // Verbatim layer
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verbatim_score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verbatim_snippet: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message_start_index: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message_end_index: Option<i64>,
    // Sub-scores
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verbatim_bm25_score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub palace_semantic_score: Option<f64>,
    // Flags
    pub has_palace: bool,
    pub has_verbatim: bool,
    pub is_intersection: bool,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn detect_source(file_path: &str) -> &'static str {
    let fp = file_path.to_lowercase();
    if fp.contains("/home/") || fp.contains("wsl") {
        "WSL"
    } else {
        "WIN"
    }
}

fn detect_tool(file_path: &str) -> Option<String> {
    let normalized = file_path.replace('\\', "/").to_lowercase();
    if normalized.contains("/.codex/") {
        Some("codex".to_string())
    } else if normalized.contains("/.vibe/") {
        Some("vibe".to_string())
    } else {
        Some("claude".to_string())
    }
}

fn parse_date_filters(
    filters: &mut SearchFilters,
    date: Option<&str>,
    date_from: Option<&str>,
    date_to: Option<&str>,
) {
    match date {
        Some("custom") => {
            if let Some(from) = date_from {
                if let Ok(d) = from.parse::<NaiveDate>() {
                    filters.date_from = Some(d.and_hms_opt(0, 0, 0).unwrap().and_utc());
                }
            }
            if let Some(to) = date_to {
                if let Ok(d) = to.parse::<NaiveDate>() {
                    filters.date_to =
                        Some((d.and_hms_opt(0, 0, 0).unwrap().and_utc()) + Duration::days(1));
                }
            }
        }
        Some("today") => {
            let now = Utc::now();
            filters.date_from = Some(
                now.date_naive()
                    .and_hms_opt(0, 0, 0)
                    .unwrap()
                    .and_utc(),
            );
            filters.date_to = Some(now);
        }
        Some("week") => {
            filters.date_from = Some(Utc::now() - Duration::days(7));
            filters.date_to = Some(Utc::now());
        }
        Some("month") => {
            filters.date_from = Some(Utc::now() - Duration::days(30));
            filters.date_to = Some(Utc::now());
        }
        _ => {}
    }
}

/// Map cross-layer mode strings to AlgorithmType.
fn map_cross_layer_mode(mode: &str) -> Option<AlgorithmType> {
    match mode {
        "cross-layer" => Some(AlgorithmType::CrossLayer),
        "verbatim" => Some(AlgorithmType::Keyword),
        "distill" => Some(AlgorithmType::Distill),
        _ => None,
    }
}

/// Map generic mode strings to AlgorithmType.
fn map_mode(mode: &str) -> AlgorithmType {
    match mode {
        "distill" | "hybrid" => AlgorithmType::Hybrid,
        "semantic" => AlgorithmType::Semantic,
        "keyword" => AlgorithmType::Keyword,
        "adaptive" => AlgorithmType::Adaptive,
        _ => AlgorithmType::Hybrid,
    }
}

fn search_result_to_response(r: SearchResult) -> SearchResultResponse {
    let source = detect_source(&r.file_path).to_string();
    let tool = detect_tool(&r.file_path);
    SearchResultResponse {
        conversation_id: r.conversation_id,
        project_id: r.project_id,
        title: r.title,
        created_at: r.created_at.to_rfc3339(),
        updated_at: r.updated_at.to_rfc3339(),
        message_count: r.message_count,
        file_path: r.file_path,
        snippet: r.snippet,
        score: r.score,
        message_start_index: r.message_start_index,
        message_end_index: r.message_end_index,
        source,
        tool,
        bm25_score: r.bm25_score,
        semantic_score: r.semantic_score,
        exchange_id: r.exchange_id,
        exchange_text: r.exchange_text,
        match_source: r.match_source,
    }
}

fn search_result_to_unified_response(r: SearchResult) -> UnifiedSearchResultResponse {
    let source = detect_source(&r.file_path).to_string();
    let tool = detect_tool(&r.file_path);
    let has_palace = r.palace_summary.is_some();
    let has_verbatim = r.bm25_score.is_some();
    UnifiedSearchResultResponse {
        conversation_id: r.conversation_id,
        project_id: r.project_id,
        title: r.title,
        created_at: r.created_at.to_rfc3339(),
        updated_at: r.updated_at.to_rfc3339(),
        message_count: r.message_count,
        file_path: r.file_path,
        combined_score: r.score,
        source,
        tool,
        palace_score: r.semantic_score,
        palace_summary: r.palace_summary,
        palace_context: r.palace_context,
        rooms: vec![],
        files_touched: vec![],
        ply_start: r.message_start_index,
        ply_end: r.message_end_index,
        object_id: r.object_id,
        verbatim_score: r.bm25_score,
        verbatim_snippet: if has_verbatim { Some(r.snippet) } else { None },
        message_start_index: r.message_start_index,
        message_end_index: r.message_end_index,
        verbatim_bm25_score: r.bm25_score,
        palace_semantic_score: r.semantic_score,
        has_palace,
        has_verbatim,
        is_intersection: has_palace && has_verbatim,
    }
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// GET /api/search — cross-layer search with cross-layer|verbatim|distill modes.
pub async fn search(
    State(state): State<AppState>,
    Query(params): Query<SearchParams>,
) -> Result<Json<Value>, ApiError> {
    let start = Instant::now();

    let limit = params.limit.min(100).max(1);

    let algorithm = map_cross_layer_mode(&params.mode).ok_or_else(|| {
        ApiError::BadRequest(format!(
            "Invalid mode '{}'. Use: cross-layer, verbatim, distill",
            params.mode
        ))
    })?;

    let mut filters = SearchFilters::default();
    if let Some(ref p) = params.project {
        filters.project_ids = Some(vec![p.clone()]);
    }
    parse_date_filters(
        &mut filters,
        params.date.as_deref(),
        params.date_from.as_deref(),
        params.date_to.as_deref(),
    );

    let storage = state.storage.clone();
    let q = params.q.clone();
    let sort_by = params.sort_by.clone();

    // Run search on blocking thread (DuckDB is synchronous).
    let results = tokio::task::spawn_blocking(move || {
        run_storage_search(&storage, &q, algorithm, &filters, limit)
    })
    .await
    .map_err(|e| ApiError::Internal(e.to_string()))??;

    let mut result_list = results;

    match sort_by.as_str() {
        "date_newest" => result_list.sort_by(|a, b| b.updated_at.cmp(&a.updated_at)),
        "date_oldest" => result_list.sort_by(|a, b| a.updated_at.cmp(&b.updated_at)),
        "messages" => result_list.sort_by(|a, b| b.message_count.cmp(&a.message_count)),
        _ => {} // relevance — already sorted by score
    }

    let total = result_list.len() as i64;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    let mode_used = format!("{:?}", algorithm).to_lowercase();

    let response_results: Vec<UnifiedSearchResultResponse> =
        result_list.into_iter().map(search_result_to_unified_response).collect();

    Ok(Json(json!({
        "results": response_results,
        "total": total,
        "search_time_ms": elapsed_ms,
        "mode_used": mode_used,
    })))
}

/// GET /api/projects — list all distinct project IDs.
pub async fn get_projects(
    State(state): State<AppState>,
) -> Result<Json<Vec<String>>, ApiError> {
    // Return cached value if available.
    {
        let cache = state.projects_cache.read();
        if let Some(ref projects) = *cache {
            return Ok(Json(projects.clone()));
        }
    }

    let storage = state.storage.clone();
    let projects = tokio::task::spawn_blocking(move || {
        // Fetch all conversation IDs (no project filter), then batch-load their
        // metadata to extract the distinct project_id values.
        let ids = storage.get_all_conversation_ids(None)?;
        let conv_map = storage.get_conversations_batch(&ids)?;
        let mut projects: Vec<String> = conv_map
            .values()
            .map(|c| c.project_id.clone())
            .filter(|p| !p.is_empty())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        projects.sort();
        Ok::<Vec<String>, searchat_storage::StorageError>(projects)
    })
    .await
    .map_err(|e| ApiError::Internal(e.to_string()))?
    .map_err(ApiError::from)?;

    // Cache result.
    {
        let mut cache = state.projects_cache.write();
        *cache = Some(projects.clone());
    }

    Ok(Json(projects))
}

/// GET /api/search/unified — exchange-level search with distill|semantic|keyword|hybrid modes.
pub async fn search_unified(
    State(state): State<AppState>,
    Query(params): Query<UnifiedSearchParams>,
) -> Result<Json<Value>, ApiError> {
    let start = Instant::now();
    let limit = params.limit.min(100).max(1);
    let algorithm = map_mode(&params.mode);

    let mut filters = SearchFilters::default();
    if let Some(ref p) = params.project {
        filters.project_ids = Some(vec![p.clone()]);
    }
    parse_date_filters(
        &mut filters,
        params.date.as_deref(),
        params.date_from.as_deref(),
        params.date_to.as_deref(),
    );

    let storage = state.storage.clone();
    let q = params.q.clone();

    let results = tokio::task::spawn_blocking(move || {
        run_storage_search(&storage, &q, algorithm, &filters, limit)
    })
    .await
    .map_err(|e| ApiError::Internal(e.to_string()))??;

    let total = results.len() as i64;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    let mode_used = format!("{:?}", algorithm).to_lowercase();

    let response_results: Vec<SearchResultResponse> =
        results.into_iter().map(search_result_to_response).collect();

    Ok(Json(json!({
        "results": response_results,
        "total": total,
        "search_time_ms": elapsed_ms,
        "mode_used": mode_used,
        "engine": "unified",
    })))
}

// ---------------------------------------------------------------------------
// Storage search dispatcher
// ---------------------------------------------------------------------------

/// Convert a `VerbatimSearchResult` from storage into a `SearchResult` domain model.
fn verbatim_storage_to_search_result(r: searchat_storage::VerbatimSearchResult) -> SearchResult {
    let snippet: String = r.exchange_text.chars().take(300).collect();
    SearchResult {
        conversation_id: r.conversation_id,
        project_id: r.project_id.unwrap_or_default(),
        title: r.title,
        created_at: r.created_at,
        updated_at: r.updated_at,
        message_count: r.message_count as i64,
        file_path: r.file_path,
        score: r.score,
        snippet: snippet.clone(),
        message_start_index: Some(r.ply_start as i64),
        message_end_index: Some(r.ply_end as i64),
        bm25_score: Some(r.score),
        semantic_score: None,
        exchange_id: Some(r.exchange_id),
        exchange_text: Some(r.exchange_text),
        match_source: Some("unified".to_string()),
        palace_summary: None,
        palace_context: None,
        files_touched_raw: None,
        object_id: None,
        search_metadata: None,
    }
}

/// Dispatch a search to the storage layer using the given algorithm.
///
/// Runs on a blocking thread (called via `spawn_blocking`).
/// Semantic/hybrid modes fall back to BM25 until an embedder is wired in.
fn run_storage_search(
    storage: &UnifiedStorage,
    query: &str,
    algorithm: AlgorithmType,
    filters: &SearchFilters,
    limit: usize,
) -> Result<Vec<SearchResult>, ApiError> {
    let project_ids: Option<Vec<String>> = filters.project_ids.clone();
    let project_ids_ref: Option<&[String]> = project_ids.as_deref();

    // All modes use BM25 for now; semantic/cross-layer will be added when the
    // embedder is injected into AppState.
    let fetch_limit = match algorithm {
        AlgorithmType::Keyword | AlgorithmType::CrossLayer => limit,
        _ => limit * 2,
    };

    let rows = storage
        .search_verbatim_bm25(query, fetch_limit, project_ids_ref)
        .map_err(|e| ApiError::Internal(e.to_string()))?;

    // Apply date filtering.
    let filtered: Vec<_> = rows
        .into_iter()
        .filter(|r| {
            if let Some(from) = filters.date_from {
                if r.updated_at < from {
                    return false;
                }
            }
            if let Some(to) = filters.date_to {
                if r.updated_at > to {
                    return false;
                }
            }
            true
        })
        .take(limit)
        .collect();

    Ok(filtered.into_iter().map(verbatim_storage_to_search_result).collect())
}
