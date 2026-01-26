"""FastAPI route handlers organized by resource."""
from searchat.api.routers.search import router as search_router
from searchat.api.routers.conversations import router as conversations_router
from searchat.api.routers.stats import router as stats_router
from searchat.api.routers.indexing import router as indexing_router
from searchat.api.routers.backup import router as backup_router
from searchat.api.routers.admin import router as admin_router
from searchat.api.routers.status import router as status_router

__all__ = [
    "search_router",
    "conversations_router",
    "stats_router",
    "indexing_router",
    "backup_router",
    "admin_router",
    "status_router",
]
