from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from searchat.config.path_resolver import PathResolver


def test_safe_exists_returns_false_for_inaccessible_path():
    broken_path = Path(r"\\wsl$\Ubuntu\home\Syd\.codex\sessions")

    with patch.object(Path, "exists", side_effect=OSError("network share unavailable")):
        assert PathResolver.safe_exists(broken_path) is False


def test_resolve_codex_dirs_skips_inaccessible_exists_checks():
    local_codex = Path(r"C:\Users\Syd\.codex\sessions")
    inaccessible_wsl = Path(r"\\wsl$\Ubuntu\home\Syd\.codex\sessions")

    def fake_exists(self):
        path_str = str(self)
        if path_str == str(local_codex):
            return True
        if path_str == str(inaccessible_wsl):
            raise OSError("network share unavailable")
        return False

    with (
        patch.object(PathResolver, "detect_platform", return_value="windows"),
        patch("searchat.config.path_resolver.CODEX_DIR_CANDIDATES", [local_codex]),
        patch.dict("os.environ", {"USERNAME": "Syd"}, clear=False),
        patch.object(Path, "exists", fake_exists),
    ):
        resolved = PathResolver.resolve_codex_dirs()

    assert local_codex in resolved
    assert inaccessible_wsl in resolved


def test_resolve_claude_dirs_skips_empty_wsl_path_on_windows():
    config = SimpleNamespace(
        paths=SimpleNamespace(
            claude_directory_windows=r"C:\Users\Syd\.claude",
            claude_directory_wsl="",
        )
    )
    windows_path = Path(r"C:\Users\Syd\.claude")

    def fake_exists(self):
        return str(self) == str(windows_path)

    with (
        patch.object(PathResolver, "detect_platform", return_value="windows"),
        patch.object(Path, "exists", fake_exists),
        patch("searchat.config.path_resolver.CLAUDE_DIR_CANDIDATES", []),
    ):
        resolved = PathResolver.resolve_claude_dirs(config)

    assert resolved == [windows_path]
