"""Registry for agent-specific conversation providers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from .base import AgentProvider
from .claude import ClaudeProvider
from .codex import CodexProvider
from .vibe import VibeProvider


_PROVIDERS = (
    CodexProvider(),
    ClaudeProvider(),
    VibeProvider(),
)


def iter_providers() -> Iterable[AgentProvider]:
    """Return all registered providers in match priority order."""
    return _PROVIDERS


def get_provider(agent_id: str) -> AgentProvider:
    """Return a provider by id."""
    for provider in _PROVIDERS:
        if provider.agent_id == agent_id:
            return provider
    raise KeyError(f"Unknown agent provider: {agent_id}")


def detect_provider(file_path: Path) -> Optional[AgentProvider]:
    """Return the provider responsible for a transcript file."""
    for provider in _PROVIDERS:
        if provider.matches_file(file_path):
            return provider
    return None


def detect_provider_id(file_path: str) -> Optional[str]:
    """Convenience wrapper for string file paths."""
    provider = detect_provider(Path(file_path))
    return provider.agent_id if provider else None
