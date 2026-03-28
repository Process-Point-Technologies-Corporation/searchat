//! Axum HTTP API — Rust port of the FastAPI searchat web server.
//!
//! ## Module Layout
//!
//! - `lib.rs`         — `AppState` construction, `start_server` entry point
//! - `state.rs`       — `AppState` shared state struct
//! - `error.rs`       — `ApiError` implementing `IntoResponse`
//! - `routes/mod.rs`  — router assembly
//! - `routes/search.rs`        — GET /api/search, /api/projects, /api/search/unified
//! - `routes/conversations.rs` — GET /api/conversations/all, /api/conversation/{id}, POST /api/resume
//! - `routes/stats.rs`         — GET /api/statistics
//! - `routes/indexing.rs`      — POST /api/index_missing, /api/reindex (403)
//! - `routes/backup.rs`        — backup CRUD
//! - `routes/admin.rs`         — GET /api/watcher/status, POST /api/shutdown

pub mod error;
pub mod routes;
pub mod state;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::RwLock;
use tokio::sync::Mutex as TokioMutex;

use searchat_config::settings::Config;
use searchat_storage::UnifiedStorage;

use crate::error::ApiError;
use crate::routes::build_router;
use crate::state::AppState;

// ---------------------------------------------------------------------------
// Default HTML fallback (used when web assets are not found on disk)
// ---------------------------------------------------------------------------

const FALLBACK_INDEX_HTML: &str = r#"<!DOCTYPE html>
<html><head><title>Searchat</title></head>
<body><h1>Searchat</h1>
<p>Web assets not found. Run the Python server for the full UI.</p>
</body></html>"#;

const FALLBACK_CONVERSATION_HTML: &str = r#"<!DOCTYPE html>
<html><head><title>Searchat — Conversation</title></head>
<body><h1>Conversation</h1>
<p>Web assets not found.</p>
</body></html>"#;

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Start the Axum HTTP server.
///
/// Reads HTML templates from `web_dir` (the Python `src/searchat/web/` directory
/// or equivalent). Falls back to embedded stubs when the path doesn't exist.
///
/// Port scanning: tries `start_port` through `start_port + PORT_SCAN_RANGE`,
/// binding the first available port. Set `SEARCHAT_PORT` to override.
pub async fn start_server(config: Config, web_dir: Option<PathBuf>) -> Result<(), ApiError> {
    // Determine data directory.
    let data_dir = PathBuf::from(&config.paths.search_directory).join("data");

    // Open (or create) the DuckDB database.
    let storage = Arc::new(
        UnifiedStorage::open(&data_dir)
            .map_err(|e| ApiError::Internal(format!("Failed to open storage: {e}")))?,
    );

    // Load HTML templates.
    let index_html = load_html(
        web_dir.as_deref().map(|d| d.join("index.html")),
        FALLBACK_INDEX_HTML,
    );
    let conversation_html = load_html(
        web_dir.as_deref().map(|d| d.join("conversation.html")),
        FALLBACK_CONVERSATION_HTML,
    );

    let state = AppState {
        config: Arc::new(config.clone()),
        storage,
        projects_cache: Arc::new(RwLock::new(None)),
        indexing_lock: Arc::new(TokioMutex::new(())),
        watcher_running: Arc::new(RwLock::new(false)),
        indexed_since_start: Arc::new(RwLock::new(0)),
        last_watcher_update: Arc::new(RwLock::new(None)),
        index_html: Arc::new(index_html),
        conversation_html: Arc::new(conversation_html),
    };

    let app = build_router(state);

    // Resolve port.
    let port = resolve_port(&config);
    let addr: SocketAddr = format!("0.0.0.0:{port}").parse()
        .map_err(|e| ApiError::Internal(format!("Invalid address: {e}")))?;

    tracing::info!("Searchat API server listening on http://{addr}");

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to bind {addr}: {e}")))?;

    axum::serve(listener, app)
        .await
        .map_err(|e| ApiError::Internal(format!("Server error: {e}")))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn load_html(path: Option<PathBuf>, fallback: &str) -> String {
    if let Some(p) = path {
        if let Ok(content) = std::fs::read_to_string(&p) {
            return content;
        }
        tracing::warn!("HTML template not found at {:?}, using fallback", p);
    }
    fallback.to_string()
}

const DEFAULT_PORT: u16 = 8000;
const PORT_SCAN_RANGE: u16 = 10;
const ENV_PORT: &str = "SEARCHAT_PORT";

fn resolve_port(_config: &Config) -> u16 {
    // Check environment variable first.
    if let Ok(val) = std::env::var(ENV_PORT) {
        if let Ok(p) = val.trim().parse::<u16>() {
            return p;
        }
    }

    // Scan for an available port.
    let start = DEFAULT_PORT;
    let end = start + PORT_SCAN_RANGE;
    for port in start..=end {
        if port_available(port) {
            return port;
        }
    }
    // Fallback: return start port and let bind() fail with a clear error.
    start
}

fn port_available(port: u16) -> bool {
    std::net::TcpListener::bind(format!("0.0.0.0:{port}")).is_ok()
}
