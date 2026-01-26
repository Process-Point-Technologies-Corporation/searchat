"""Cross-platform utilities for terminal and path management."""

import platform
import subprocess
import logging
from typing import Optional
import shlex
from pathlib import Path

logger = logging.getLogger(__name__)


class PlatformManager:
    """Manages platform-specific terminal launching and path handling."""

    def __init__(self):
        # Import here to avoid circular dependency
        from searchat.config import PathResolver

        # Use PathResolver for accurate platform detection (distinguishes WSL from Linux)
        self.platform = PathResolver.detect_platform()  # 'windows', 'wsl', 'linux', 'macos'
        self.system = platform.system()  # Keep for compatibility: 'Windows', 'Darwin', 'Linux'

        self.is_windows = self.platform == 'windows'
        self.is_wsl = self.platform == 'wsl'
        self.is_macos = self.platform == 'macos'
        self.is_linux = self.platform == 'linux'

    def _translate_cwd_if_needed(self, cwd: Optional[str]) -> Optional[str]:
        """
        Translate cwd path to be suitable for current platform.

        Args:
            cwd: Working directory path (can be native or foreign format)

        Returns:
            Translated path string or None
        """
        if not cwd:
            return None

        from searchat.config import PathResolver

        # Check if translation is needed
        if self.is_windows:
            # On Windows, keep Windows paths as-is
            # WSL paths (starting with /) will be handled by _open_windows_terminal
            return cwd

        elif self.is_wsl or self.is_linux:
            # On WSL/Linux, translate Windows paths to Unix format
            if len(cwd) >= 3 and (cwd[1:3] == ':\\' or cwd[1:3] == ':/'):
                # Windows path like C:\... or C:/...
                translated = PathResolver.translate_claude_path(cwd)
                logger.info(f"Translated Windows path {cwd} → {translated}")
                return str(translated)
            return cwd

        elif self.is_macos:
            # macOS uses Unix paths, no translation needed
            return cwd

        return cwd

    def open_terminal_with_command(
        self,
        command: str,
        cwd: Optional[str] = None
    ) -> subprocess.Popen:
        """
        Open a new terminal window and execute a command.

        Uses subprocess argument lists (not shell=True) to avoid escaping issues.
        Automatically translates paths between Windows/WSL/Unix formats as needed.

        Args:
            command: The command to execute (e.g., "claude --resume abc123")
            cwd: Working directory path (can be Unix or Windows format)

        Returns:
            subprocess.Popen instance
        """
        # Translate cwd to appropriate format for current platform
        translated_cwd = self._translate_cwd_if_needed(cwd)

        if self.is_windows:
            return self._open_windows_terminal(command, translated_cwd)
        elif self.is_wsl:
            return self._open_wsl_terminal(command, translated_cwd)
        elif self.is_macos:
            return self._open_macos_terminal(command, translated_cwd)
        elif self.is_linux:
            return self._open_linux_terminal(command, translated_cwd)
        else:
            raise NotImplementedError(f"Unsupported platform: {self.platform}")

    def _open_windows_terminal(
        self,
        command: str,
        cwd: Optional[str]
    ) -> subprocess.Popen:
        """Open terminal on Windows using argument lists."""
        if cwd and cwd.startswith('/'):
            # WSL path - use wsl with --cd
            # cmd.exe /c start cmd.exe /k wsl.exe --cd /path bash -i -c "command"
            cmd_args = [
                'cmd.exe', '/c', 'start',
                'cmd.exe', '/k',
                'wsl.exe', '--cd', cwd,
                'bash', '-i', '-c', command
            ]
        elif cwd:
            # Windows path - cd /d then run command
            # cmd.exe /c start cmd.exe /k "cd /d path && command"
            combined = f'cd /d {cwd} && {command}'
            cmd_args = ['cmd.exe', '/c', 'start', 'cmd.exe', '/k', combined]
        else:
            # No cwd
            cmd_args = ['cmd.exe', '/c', 'start', 'cmd.exe', '/k', command]

        logger.info(f"Windows args: {cmd_args}")
        return subprocess.Popen(cmd_args)

    def _open_wsl_terminal(
        self,
        command: str,
        cwd: Optional[str]
    ) -> subprocess.Popen:
        """
        Open terminal from WSL.

        Can launch either WSL bash terminal or Windows terminal depending on context.
        Uses cmd.exe /c start pattern to launch Windows GUI terminals from WSL.
        """
        # For now, launch bash terminal in WSL
        # Future: could detect Windows paths and launch Windows terminal instead
        if cwd:
            bash_cmd = f'cd {shlex.quote(cwd)} && {command}'
        else:
            bash_cmd = command

        # Launch bash in a way that keeps terminal open
        cmd_args = ['bash', '-c', f'{bash_cmd}; exec bash']

        logger.info(f"WSL args: {cmd_args}")
        return subprocess.Popen(cmd_args, start_new_session=True)

    def _open_macos_terminal(
        self,
        command: str,
        cwd: Optional[str]
    ) -> subprocess.Popen:
        """Open terminal on macOS using osascript."""
        if cwd:
            script = f'tell application "Terminal" to do script "cd {shlex.quote(cwd)} && {command}"'
        else:
            script = f'tell application "Terminal" to do script "{command}"'

        cmd_args = ['osascript', '-e', script]
        logger.info(f"macOS args: {cmd_args}")
        return subprocess.Popen(cmd_args)

    def _open_linux_terminal(
        self,
        command: str,
        cwd: Optional[str]
    ) -> subprocess.Popen:
        """Open terminal on Linux using gnome-terminal or alternatives."""
        # Build bash command
        if cwd:
            bash_cmd = f'cd {shlex.quote(cwd)} && {command}'
        else:
            bash_cmd = command

        # Try gnome-terminal first
        cmd_args = ['gnome-terminal', '--', 'bash', '-c', f'{bash_cmd}; exec bash']

        logger.info(f"Linux args: {cmd_args}")
        return subprocess.Popen(cmd_args)

    def normalize_path(self, path: str) -> str:
        """Normalize path for the current platform."""
        if self.is_windows:
            return path
        else:
            return path.replace('\\', '/')

    def detect_wsl_path(self, path: str) -> bool:
        """Detect if a path is a WSL path (only relevant on Windows)."""
        if not self.is_windows:
            return False

        return (
            path.startswith('/home/') or
            path.startswith('/mnt/') or
            '\\\\wsl' in path.lower()
        )
