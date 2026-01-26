"""Readiness and warmup state tracking for the API.

This module is intentionally lightweight and safe to import at startup.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Literal


ComponentState = Literal["idle", "loading", "ready", "error"]
WatcherState = Literal["disabled", "starting", "running", "error"]


@dataclass(frozen=True)
class ReadinessSnapshot:
    """Immutable snapshot of current readiness."""

    warmup_started_at: str | None
    components: dict[str, ComponentState]
    watcher: WatcherState
    errors: dict[str, str]


class Readiness:
    """Thread-safe readiness state manager."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._warmup_started_at: str | None = None
        self._components: dict[str, ComponentState] = {
            "services": "idle",
            "search_engine": "idle",
            "indexer": "idle",
        }
        self._watcher: WatcherState = "disabled"
        self._errors: dict[str, str] = {}

    def mark_warmup_started(self) -> bool:
        """Mark warmup started; returns True only the first time."""
        with self._lock:
            if self._warmup_started_at is not None:
                return False
            self._warmup_started_at = datetime.now(timezone.utc).isoformat()
            return True

    def set_component(self, name: str, state: ComponentState, *, error: str | None = None) -> None:
        with self._lock:
            self._components[name] = state
            if error is None:
                self._errors.pop(name, None)
            else:
                self._errors[name] = error

    def set_watcher(self, state: WatcherState, *, error: str | None = None) -> None:
        with self._lock:
            self._watcher = state
            if error is None:
                self._errors.pop("watcher", None)
            else:
                self._errors["watcher"] = error

    def snapshot(self) -> ReadinessSnapshot:
        with self._lock:
            return ReadinessSnapshot(
                warmup_started_at=self._warmup_started_at,
                components=dict(self._components),
                watcher=self._watcher,
                errors=dict(self._errors),
            )


_READINESS = Readiness()


def get_readiness() -> Readiness:
    return _READINESS


def warming_payload(*, retry_after_ms: int = 500) -> dict[str, Any]:
    snap = _READINESS.snapshot()
    return {
        "status": "warming",
        "retry_after_ms": retry_after_ms,
        "warmup_started_at": snap.warmup_started_at,
        "components": snap.components,
        "watcher": snap.watcher,
        "errors": snap.errors,
    }


def error_payload() -> dict[str, Any]:
    snap = _READINESS.snapshot()
    return {
        "status": "error",
        "warmup_started_at": snap.warmup_started_at,
        "components": snap.components,
        "watcher": snap.watcher,
        "errors": snap.errors,
    }
