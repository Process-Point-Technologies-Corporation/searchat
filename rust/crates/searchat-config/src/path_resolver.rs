use std::path::{Path, PathBuf};

use crate::constants::*;

// ============================================================================
// Platform detection
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    Windows,
    Wsl,
    Linux,
    Macos,
    Unknown,
}

/// Detect the current runtime platform.
pub fn detect_platform() -> Platform {
    if cfg!(target_os = "windows") {
        Platform::Windows
    } else if cfg!(target_os = "macos") {
        Platform::Macos
    } else if cfg!(target_os = "linux") {
        if is_wsl() {
            Platform::Wsl
        } else {
            Platform::Linux
        }
    } else {
        Platform::Unknown
    }
}

/// Check whether the process is running under WSL by inspecting /proc/version.
pub fn is_wsl() -> bool {
    match std::fs::read_to_string("/proc/version") {
        Ok(content) => {
            let lower = content.to_lowercase();
            lower.contains("microsoft") || lower.contains("wsl")
        }
        Err(_) => false,
    }
}

// ============================================================================
// Standard directory helpers
// ============================================================================

/// `~/.searchat/`
pub fn default_data_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(CONFIG_DIR_NAME)
}

/// `~/.searchat/config/`
pub fn default_config_dir() -> PathBuf {
    default_data_dir().join(DEFAULT_CONFIG_SUBDIR)
}

/// `~/.searchat/data/`
pub fn default_storage_dir() -> PathBuf {
    default_data_dir().join(DEFAULT_DATA_SUBDIR)
}

/// `~/.searchat/logs/`
pub fn default_logs_dir() -> PathBuf {
    default_data_dir().join(DEFAULT_LOGS_SUBDIR)
}

/// `~/.claude/projects/`
pub fn default_claude_projects_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(CLAUDE_DIR_NAME)
        .join(CLAUDE_PROJECTS_SUBDIR)
}

/// `~/.codex/sessions/`
pub fn default_codex_sessions_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(CODEX_DIR_NAME)
        .join(CODEX_SESSIONS_SUBDIR)
}

// ============================================================================
// Path translation (Windows <-> WSL)
// ============================================================================

/// Translate a path string between Windows and WSL conventions, based on the
/// current platform.
///
/// - On WSL/Linux: `C:\Users\...` or `C:/Users/...` → `/mnt/c/Users/...`
/// - On Windows:   `/mnt/c/Users/...` → `C:\Users\...`
/// - Otherwise:    return as-is
pub fn translate_path(path: &str) -> PathBuf {
    let platform = detect_platform();

    // Windows drive path seen from WSL
    if matches!(platform, Platform::Wsl | Platform::Linux) {
        if path.len() >= 3 {
            let bytes = path.as_bytes();
            let second = bytes.get(1).copied();
            let third = bytes.get(2).copied();
            if second == Some(b':') && (third == Some(b'\\') || third == Some(b'/')) {
                let drive = path.chars().next().unwrap().to_lowercase().next().unwrap();
                let rest = path[3..].replace('\\', "/");
                return PathBuf::from(format!("/mnt/{}/{}", drive, rest));
            }
        }
    }

    // WSL mount point seen from Windows
    if platform == Platform::Windows && path.starts_with("/mnt/") {
        let stripped = &path[5..]; // remove "/mnt/"
        let mut parts = stripped.splitn(2, '/');
        let drive = parts.next().unwrap_or("c");
        let rest = parts.next().unwrap_or("");
        return PathBuf::from(format!("{}:\\{}", drive.to_uppercase(), rest.replace('/', "\\")));
    }

    // WSL UNC path (\\wsl$\...) — already Windows-formatted
    // No translation needed; pass through.
    PathBuf::from(path)
}

// ============================================================================
// Safe filesystem helpers
// ============================================================================

/// Returns `false` instead of panicking for inaccessible paths (e.g. UNC shares).
pub fn safe_exists(path: &Path) -> bool {
    path.try_exists().unwrap_or(false)
}

/// Resolve a path to absolute without panicking on inaccessible network shares.
pub fn safe_resolve(path: &Path) -> PathBuf {
    if safe_exists(path) {
        path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
    } else {
        path.to_path_buf()
    }
}

/// Ensure a directory exists, creating it (and parents) if needed.
pub fn ensure_directory(path: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(path)
}

// ============================================================================
// Agent directory resolution
// ============================================================================

/// Resolve all accessible Claude conversation directories in priority order:
/// 1. `SEARCHAT_ADDITIONAL_DIRS` (colon/semicolon-separated on the current OS)
/// 2. Windows project dir from config (translated if needed)
/// 3. WSL project dir from config (Windows only)
/// 4. Standard fallback candidates
pub fn resolve_claude_dirs(config: &crate::settings::Config) -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = Vec::new();
    let platform = detect_platform();

    // 1. Additional dirs from environment
    if let Ok(additional) = std::env::var(ENV_ADDITIONAL_DIRS) {
        let sep = if platform == Platform::Windows { ';' } else { ':' };
        for dir in additional.split(sep) {
            let expanded = expand_path_template(dir);
            let translated = translate_path(&expanded);
            if safe_exists(&translated) {
                paths.push(translated);
            }
        }
    }

    // 2. Configured Windows directory
    let windows_expanded = expand_path_template(&config.paths.claude_directory_windows);
    let windows_path = translate_path(&windows_expanded);
    if safe_exists(&windows_path) {
        paths.push(windows_path);
    }

    // 3. WSL directory (only relevant from Windows)
    if platform == Platform::Windows {
        let wsl = config.paths.claude_directory_wsl.trim();
        if !wsl.is_empty() {
            // UNC-style WSL paths don't respond to Path::exists reliably on Windows,
            // so always include them as candidates.
            paths.push(PathBuf::from(wsl));
        }
    }

    // 4. Standard fallback
    if paths.is_empty() {
        let fallback = default_claude_projects_dir();
        if safe_exists(&fallback) {
            paths.push(fallback);
        }
    }

    deduplicate_paths(paths)
}

/// Resolve Codex session directories.
pub fn resolve_codex_dirs() -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = Vec::new();
    let platform = detect_platform();

    let standard = default_codex_sessions_dir();
    if safe_exists(&standard) {
        paths.push(standard);
    }

    if platform == Platform::Windows {
        let username = std::env::var("USERNAME")
            .or_else(|_| std::env::var("USER"))
            .unwrap_or_else(|_| "user".to_string());
        let wsl_candidates = vec![
            PathBuf::from(format!(
                "\\\\wsl.localhost\\Ubuntu\\home\\{}\\{}\\{}",
                username, CODEX_DIR_NAME, CODEX_SESSIONS_SUBDIR
            )),
            PathBuf::from(format!(
                "\\\\wsl$\\Ubuntu\\home\\{}\\{}\\{}",
                username, CODEX_DIR_NAME, CODEX_SESSIONS_SUBDIR
            )),
        ];
        for candidate in wsl_candidates {
            paths.push(candidate);
        }
    }

    deduplicate_paths(paths)
}

/// Resolve Vibe session directories (`~/.vibe/logs/session/`).
pub fn resolve_vibe_dirs() -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = Vec::new();

    let standard = dirs::home_dir()
        .unwrap_or_default()
        .join(".vibe")
        .join("logs")
        .join("session");
    if safe_exists(&standard) {
        paths.push(standard);
    }

    if let Ok(vibe_home) = std::env::var("VIBE_HOME") {
        let custom = PathBuf::from(&vibe_home).join("logs").join("session");
        if safe_exists(&custom) && !paths.contains(&custom) {
            paths.push(custom);
        }
    }

    paths
}

// ============================================================================
// Helpers
// ============================================================================

/// Expand `{username}`, `{home}`, `~`, and environment variables in a path string.
pub fn expand_path_template(path: &str) -> String {
    let username = std::env::var("USERNAME")
        .or_else(|_| std::env::var("USER"))
        .unwrap_or_else(|_| "unknown".to_string());
    let home = dirs::home_dir()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();

    let mut result = path.to_string();
    result = result.replace("{username}", &username);
    result = result.replace("{home}", &home);
    result = result.replace('~', &home);
    result
}

/// Deduplicate a list of paths while preserving order, using canonical form for comparison.
fn deduplicate_paths(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut seen = std::collections::HashSet::new();
    let mut unique = Vec::new();
    for path in paths {
        let key = safe_resolve(&path).to_string_lossy().into_owned();
        if seen.insert(key) {
            unique.push(path);
        }
    }
    unique
}
