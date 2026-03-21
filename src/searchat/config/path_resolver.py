"""
Cross-platform path resolution for AI coding agent conversation directories.

Supports:
- Claude Code: ~/.claude/projects/**/*.jsonl
- Codex: ~/.codex/sessions/**/*.jsonl
- Mistral Vibe: ~/.vibe/logs/session/*.json

Handles path translation between:
- Windows native paths (C:/Users/...)
- WSL paths (/mnt/c/Users/... or //wsl$/Ubuntu/...)
- Unix/Linux/macOS paths (/home/user/...)
"""

import os
import sys
import platform
from pathlib import Path
from typing import List, Optional

from .constants import (
    ENV_DATA_DIR,
    ENV_WINDOWS_PROJECTS,
    ENV_WSL_PROJECTS,
    ENV_ADDITIONAL_DIRS,
    CLAUDE_DIR_CANDIDATES,
    CODEX_DIR_CANDIDATES,
)


class PathResolver:
    """Resolves and translates paths across different platforms."""

    @staticmethod
    def safe_exists(path: Path) -> bool:
        """Return False instead of raising for inaccessible paths."""
        try:
            return path.exists()
        except OSError:
            return False

    @staticmethod
    def safe_resolve(path: Path) -> Path:
        """Resolve a path when possible without failing on broken UNC shares."""
        try:
            return path.resolve() if PathResolver.safe_exists(path) else path
        except OSError:
            return path

    @staticmethod
    def expand_path_template(path: str) -> str:
        """
        Expand path templates with environment variables.

        Supports:
        - {username} -> current username
        - {home} -> user home directory
        - Environment variables: $VAR or ${VAR}

        Args:
            path: Path template string

        Returns:
            Expanded path string
        """
        # Expand custom placeholders
        path = path.replace("{username}", os.getenv("USERNAME") or os.getenv("USER") or "unknown")
        path = path.replace("{home}", str(Path.home()))

        # Expand ~ to home directory
        path = os.path.expanduser(path)

        # Expand environment variables
        path = os.path.expandvars(path)

        return path

    @staticmethod
    def get_shared_search_dir(config=None) -> Path:
        """
        Get the shared search directory path.

        Checks in order:
        1. Environment variable (SEARCHAT_DATA_DIR)
        2. Config file setting
        3. Default location

        Args:
            config: Optional Config object

        Returns:
            Path to search directory
        """
        # Check environment variable first
        env_dir = os.getenv(ENV_DATA_DIR)
        if env_dir:
            return Path(PathResolver.expand_path_template(env_dir))

        # Load config if not provided
        if config is None:
            from searchat.config import Config
            config = Config.load()

        # Expand template in config path
        search_dir = PathResolver.expand_path_template(config.paths.search_directory)
        return Path(search_dir)

    @staticmethod
    def detect_platform() -> str:
        """
        Detect the current platform.

        Returns:
            One of: "windows", "wsl", "linux", "macos"
        """
        if sys.platform == "win32":
            return "windows"
        elif sys.platform == "darwin":
            return "macos"
        elif sys.platform.startswith("linux"):
            # Check if we're in WSL
            if PathResolver._is_wsl():
                return "wsl"
            return "linux"
        return "unknown"

    @staticmethod
    def _is_wsl() -> bool:
        """Check if running under WSL."""
        try:
            with open("/proc/version", "r") as f:
                return "microsoft" in f.read().lower()
        except (FileNotFoundError, PermissionError):
            return False

    @staticmethod
    def translate_claude_path(path: str) -> Path:
        """
        Translate path between different platform conventions.

        Handles:
        - Windows <-> WSL path translation
        - Network UNC paths (//wsl$/...)
        - Unix mount points (/mnt/c/...)

        Args:
            path: Path string to translate

        Returns:
            Translated Path object
        """
        current_platform = PathResolver.detect_platform()

        # Windows native -> WSL mount point
        if current_platform in ("wsl", "linux") and (path[1:3] == ":\\" or path[1:3] == ":/"):
            # C:\Users\... -> /mnt/c/Users/...
            drive = path[0].lower()
            rest = path[3:].replace("\\", "/")
            return Path(f"/mnt/{drive}/{rest}")

        # WSL mount point -> Windows native
        elif current_platform == "windows" and path.startswith("/mnt/"):
            # /mnt/c/Users/... -> C:\Users\...
            parts = path[5:].split("/", 1)
            if len(parts) == 2:
                drive, rest = parts
                return Path(f"{drive.upper()}:\\{rest.replace('/', chr(92))}")
            else:
                drive = parts[0]
                return Path(f"{drive.upper()}:\\")

        # WSL UNC path (\\wsl$\Ubuntu\home\...) - already in Windows format
        elif current_platform == "windows" and path.startswith("\\\\wsl"):
            return Path(path)

        # No translation needed
        return Path(path)

    @staticmethod
    def resolve_claude_dirs(config=None) -> List[Path]:
        """
        Resolve all accessible Claude conversation directories.

        Checks:
        1. Additional directories from environment variable
        2. Configured Windows directory
        3. Configured WSL directory
        4. Standard fallback locations

        Args:
            config: Optional Config object

        Returns:
            List of accessible Path objects

        Raises:
            RuntimeError: If no directories are found
        """
        if config is None:
            from searchat.config import Config
            config = Config.load()

        paths = []
        current_platform = PathResolver.detect_platform()

        # 1. Check additional directories from environment
        additional_dirs = os.getenv(ENV_ADDITIONAL_DIRS)
        if additional_dirs:
            for dir_path in additional_dirs.split(os.pathsep):
                expanded = PathResolver.expand_path_template(dir_path)
                path = PathResolver.translate_claude_path(expanded)
                if PathResolver.safe_exists(path):
                    paths.append(path)

        # 2. Check Windows directory
        windows_dir = PathResolver.expand_path_template(config.paths.claude_directory_windows)
        windows_path = PathResolver.translate_claude_path(windows_dir)
        if PathResolver.safe_exists(windows_path):
            paths.append(windows_path)

        # 3. Check WSL directory (only from Windows)
        if current_platform == "windows":
            wsl_dir = PathResolver.expand_path_template(config.paths.claude_directory_wsl)
            if wsl_dir.strip():
                wsl_path = Path(wsl_dir)
                # Note: Path.exists() doesn't work reliably with WSL network paths,
                # but glob/iterdir work fine, so always add the path on Windows
                paths.append(wsl_path)

        # 4. Check standard fallback locations
        if not paths:
            for candidate in CLAUDE_DIR_CANDIDATES:
                if PathResolver.safe_exists(candidate):
                    paths.append(candidate)

        # Remove duplicates while preserving order
        seen = set()
        unique_paths = []
        for path in paths:
            # Resolve to absolute path for comparison
            resolved = PathResolver.safe_resolve(path)
            path_str = str(resolved)
            if path_str not in seen:
                seen.add(path_str)
                unique_paths.append(path)

        if not unique_paths:
            searched = [
                PathResolver.expand_path_template(config.paths.claude_directory_windows),
                PathResolver.expand_path_template(config.paths.claude_directory_wsl),
            ] + [str(c) for c in CLAUDE_DIR_CANDIDATES]

            raise RuntimeError(
                f"No Claude conversation directories found.\n"
                f"Platform: {current_platform}\n"
                f"Searched locations:\n" +
                "\n".join(f"  - {p}" for p in searched)
            )

        return unique_paths

    @staticmethod
    def ensure_directory(path: Path) -> Path:
        """
        Ensure a directory exists, creating it if necessary.

        Args:
            path: Directory path to ensure

        Returns:
            The path object

        Raises:
            PermissionError: If directory cannot be created
        """
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def resolve_vibe_dirs() -> List[Path]:
        """
        Resolve Mistral Vibe session directories.

        Vibe stores sessions at ~/.vibe/logs/session/

        Returns:
            List of accessible Vibe session directories
        """
        paths = []

        # Standard Vibe location
        vibe_session_dir = Path.home() / ".vibe" / "logs" / "session"
        if PathResolver.safe_exists(vibe_session_dir):
            paths.append(vibe_session_dir)

        # Check VIBE_HOME environment variable
        vibe_home = os.getenv("VIBE_HOME")
        if vibe_home:
            custom_session_dir = Path(vibe_home) / "logs" / "session"
            if PathResolver.safe_exists(custom_session_dir) and custom_session_dir not in paths:
                paths.append(custom_session_dir)

        return paths

    @staticmethod
    def resolve_codex_dirs() -> List[Path]:
        """
        Resolve Codex session directories.

        Codex stores sessions at ~/.codex/sessions/YYYY/MM/DD/*.jsonl

        Returns:
            List of accessible Codex session directories
        """
        paths = []
        current_platform = PathResolver.detect_platform()

        for candidate in CODEX_DIR_CANDIDATES:
            if PathResolver.safe_exists(candidate):
                paths.append(candidate)

        if current_platform == "windows":
            username = os.getenv("USERNAME") or os.getenv("USER") or "user"
            wsl_candidates = [
                Path(f"\\\\wsl.localhost\\Ubuntu\\home\\{username}\\.codex\\sessions"),
                Path(f"\\\\wsl$\\Ubuntu\\home\\{username}\\.codex\\sessions"),
            ]
            for candidate in wsl_candidates:
                if candidate not in paths:
                    paths.append(candidate)

        seen = set()
        unique_paths = []
        for path in paths:
            resolved = PathResolver.safe_resolve(path)
            path_str = str(resolved)
            if path_str not in seen:
                seen.add(path_str)
                unique_paths.append(path)

        return unique_paths

    @staticmethod
    def resolve_all_agent_dirs(config=None) -> dict:
        """
        Resolve all supported AI coding agent directories.

        Returns:
            Dict mapping agent name to list of directories:
            {
                'claude': [Path(...), ...],
                'codex': [Path(...), ...],
                'vibe': [Path(...), ...]
            }
        """
        return {
            'claude': PathResolver.resolve_claude_dirs(config),
            'codex': PathResolver.resolve_codex_dirs(),
            'vibe': PathResolver.resolve_vibe_dirs(),
        }
