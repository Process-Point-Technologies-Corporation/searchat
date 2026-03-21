"""Pydantic models for API requests and responses."""
from searchat.api.models.requests import (
    SearchRequest,
    ResumeRequest,
    BackupCreateRequest,
    BackupRestoreRequest,
)
from searchat.api.models.responses import (
    SearchResultResponse,
    UnifiedSearchResultResponse,
    RoomResponse,
    FileTouchedResponse,
    ConversationMessage,
    ConversationResponse,
    BackupMetadataResponse,
    BackupCreateResponse,
    BackupListResponse,
    BackupRestoreResponse,
)

__all__ = [
    # Requests
    "SearchRequest",
    "ResumeRequest",
    "BackupCreateRequest",
    "BackupRestoreRequest",
    # Responses
    "SearchResultResponse",
    "UnifiedSearchResultResponse",
    "RoomResponse",
    "FileTouchedResponse",
    "ConversationMessage",
    "ConversationResponse",
    "BackupMetadataResponse",
    "BackupCreateResponse",
    "BackupListResponse",
    "BackupRestoreResponse",
]
