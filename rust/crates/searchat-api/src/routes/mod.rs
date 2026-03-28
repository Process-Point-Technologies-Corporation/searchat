pub mod admin;
pub mod backup;
pub mod conversations;
pub mod indexing;
pub mod search;
pub mod stats;

use axum::{
    Router,
    routing::{delete, get, post},
};
use tower_http::cors::{Any, CorsLayer};

use crate::state::AppState;

pub fn build_router(state: AppState) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        // HTML pages
        .route("/", get(serve_index))
        .route("/conversation/{id}", get(serve_conversation_page))
        // Search endpoints
        .route("/api/search", get(search::search))
        .route("/api/projects", get(search::get_projects))
        .route("/api/search/unified", get(search::search_unified))
        // Conversation endpoints
        .route("/api/conversations/all", get(conversations::get_all_conversations))
        .route("/api/conversation/{id}", get(conversations::get_conversation))
        .route("/api/resume", post(conversations::resume_session))
        // Statistics
        .route("/api/statistics", get(stats::get_statistics))
        // Indexing
        .route("/api/index_missing", post(indexing::index_missing))
        .route("/api/reindex", post(indexing::reindex_blocked))
        .route("/api/distill", post(indexing::distill_pending))
        .route("/api/distill/{conversation_id}", post(indexing::distill_conversation))
        // Backup CRUD
        .route("/api/backup/create", post(backup::create_backup))
        .route("/api/backup/list", get(backup::list_backups))
        .route("/api/backup/restore", post(backup::restore_backup))
        .route("/api/backup/delete/{name}", delete(backup::delete_backup))
        // Admin
        .route("/api/watcher/status", get(admin::watcher_status))
        .route("/api/indexing/status", get(admin::indexing_status))
        .route("/api/shutdown", post(admin::shutdown_server))
        // Static files are served by the binary after reading from the web directory.
        // For now we return 404 for /static/* — the Python server mounts them from disk.
        .layer(cors)
        .with_state(state)
}

async fn serve_index(
    axum::extract::State(state): axum::extract::State<AppState>,
) -> axum::response::Html<String> {
    axum::response::Html((*state.index_html).clone())
}

async fn serve_conversation_page(
    axum::extract::State(state): axum::extract::State<AppState>,
) -> axum::response::Html<String> {
    axum::response::Html((*state.conversation_html).clone())
}
