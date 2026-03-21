"""Pydantic response models for API endpoints."""
from typing import List, Optional
from pydantic import BaseModel


class RoomResponse(BaseModel):
    """Room metadata in search response."""
    room_id: str
    room_type: str
    room_key: str
    room_label: str


class FileTouchedResponse(BaseModel):
    """File touched in search response."""
    path: str
    action: str


class SearchResultResponse(BaseModel):
    """Single search result in API response."""
    conversation_id: str
    project_id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int
    file_path: str
    snippet: str
    score: float
    message_start_index: Optional[int] = None
    message_end_index: Optional[int] = None
    source: str  # WIN or WSL
    tool: Optional[str] = None
    # Sub-scores for hybrid search analysis
    bm25_score: Optional[float] = None
    semantic_score: Optional[float] = None
    # Exchange-level fields (populated by unified search engine)
    exchange_id: Optional[str] = None
    exchange_text: Optional[str] = None
    # Match source indicator: "legacy" | "unified" | "both"
    match_source: Optional[str] = None


class UnifiedSearchResultResponse(BaseModel):
    """Unified search result from palace and verbatim layers."""
    conversation_id: str
    project_id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int
    file_path: str
    combined_score: float
    source: str  # WIN or WSL
    tool: Optional[str] = None

    # Palace layer data (optional)
    palace_score: Optional[float] = None
    palace_summary: Optional[str] = None
    palace_context: Optional[str] = None
    rooms: List[RoomResponse] = []
    files_touched: List[FileTouchedResponse] = []
    ply_start: Optional[int] = None
    ply_end: Optional[int] = None
    object_id: Optional[str] = None

    # Verbatim layer data (optional)
    verbatim_score: Optional[float] = None
    verbatim_snippet: Optional[str] = None
    message_start_index: Optional[int] = None
    message_end_index: Optional[int] = None

    # Sub-scores for analysis (all 4 components)
    palace_bm25_score: Optional[float] = None
    palace_semantic_score: Optional[float] = None
    verbatim_bm25_score: Optional[float] = None
    verbatim_semantic_score: Optional[float] = None

    # Flags
    has_palace: bool = False
    has_verbatim: bool = False
    is_intersection: bool = False

    # Progressive fallback tracking
    fallback_tier: Optional[str] = None  # "scoped" | "related" | "unscoped" | None


class ConversationMessage(BaseModel):
    """Message in conversation response."""
    role: str
    content: str
    timestamp: str
    ply_index: Optional[int] = None


class ConversationResponse(BaseModel):
    """Full conversation details."""
    conversation_id: str
    title: str
    project_id: str
    file_path: str
    tool: Optional[str] = None
    message_count: int
    messages: List[ConversationMessage]


class BackupMetadataResponse(BaseModel):
    """Backup metadata."""
    backup_path: str
    backup_name: str
    created_at: str
    file_count: int
    total_size_bytes: int


class BackupCreateResponse(BaseModel):
    """Backup creation result."""
    message: str
    backup_name: str
    backup_path: str
    file_count: int
    total_size_mb: float


class BackupListResponse(BaseModel):
    """List of available backups."""
    backups: List[BackupMetadataResponse]


class BackupRestoreResponse(BaseModel):
    """Backup restore result."""
    message: str
    backup_name: str
    restored_files: int
