"""Status endpoints - readiness and warmup state."""

from fastapi import APIRouter

from searchat.api.readiness import get_readiness


router = APIRouter()


@router.get("/status")
async def get_status():
    """Return current readiness state for UI polling and diagnostics."""
    snap = get_readiness().snapshot()
    return {
        "warmup_started_at": snap.warmup_started_at,
        "components": snap.components,
        "watcher": snap.watcher,
        "errors": snap.errors,
    }
