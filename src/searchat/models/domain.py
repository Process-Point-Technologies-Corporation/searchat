"""Domain models for searchat - business logic data structures."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict


@dataclass
class MessageRecord:
    """Record representing a single message in a conversation."""
    sequence: int
    role: str
    content: str
    timestamp: datetime
    has_code: bool
    code_blocks: List[str] = field(default_factory=list)


@dataclass
class ConversationRecord:
    """Record representing a full conversation with metadata."""
    conversation_id: str
    project_id: str
    file_path: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int
    messages: List[MessageRecord]
    full_text: str
    embedding_id: int
    file_hash: str
    indexed_at: datetime
    file_size: int = 0
    mtime_ns: int = 0


@dataclass
class SearchFilters:
    """Filters for search queries."""
    project_ids: Optional[List[str]] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    min_messages: int = 0
    has_code: Optional[bool] = None


@dataclass
class SearchResult:
    """Single search result with metadata and score."""
    conversation_id: str
    project_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int
    file_path: str
    score: float
    snippet: str
    message_start_index: Optional[int] = None
    message_end_index: Optional[int] = None
    bm25_score: Optional[float] = None
    semantic_score: Optional[float] = None
    # Exchange-level fields (populated by unified search engine)
    exchange_id: Optional[str] = None
    exchange_text: Optional[str] = None
    # Match source indicator: "legacy" | "unified" | "both"
    match_source: Optional[str] = None
    # Palace-layer fields (populated by cross-layer and distill searches)
    palace_summary: Optional[str] = None
    palace_context: Optional[str] = None
    files_touched_raw: Optional[List[Dict[str, str]]] = None
    object_id: Optional[str] = None
    # Extra metadata dict for search-specific annotations
    search_metadata: Optional[Dict[str, Any]] = None


@dataclass
class SearchResults:
    """Collection of search results with metadata."""
    results: List[SearchResult]
    total_count: int
    search_time_ms: float
    mode_used: str
    error: Optional[str] = None


class IndexingStats(TypedDict, total=False):
    """Statistics returned by indexing operations."""
    new_conversations: int
    updated_conversations: int
    exchanges_created: int
    embeddings_created: int
    skipped_already_indexed: int
    skipped_errors: int
    skipped_existing: int
    skipped_empty: int
    invalid_transcript_count: int
    invalid_transcript_examples: List[str]
    skipped_known_invalid: int
    append_only_updates: int
    total_files: int
    changed_detected: int
    parse_seconds: float
    encode_seconds: float
    store_seconds: float
    time_seconds: float
    conversations_processed: int


@dataclass
class IndexStats:
    """Statistics about the search index."""
    total_conversations: int
    total_messages: int
    index_time_seconds: float
    parquet_size_mb: float
    faiss_size_mb: float


@dataclass
class UpdateStats:
    """Statistics about an incremental index update."""
    new_conversations: int
    updated_conversations: int
    skipped_conversations: int
    update_time_seconds: float


@dataclass
class DateFilter:
    """Date range filter for search queries."""
    from_date: Optional[datetime]
    to_date: Optional[datetime]


@dataclass
class ParsedQuery:
    """Parsed search query with extracted components."""
    original: str
    must_include: List[str] = field(default_factory=list)
    should_include: List[str] = field(default_factory=list)
    must_exclude: List[str] = field(default_factory=list)
    exact_phrases: List[str] = field(default_factory=list)
    date_filter: Optional[DateFilter] = None


# ============================================================================
# Memory Palace Models (Distillation)
# ============================================================================

@dataclass
class FileTouched:
    """A file referenced in a distilled exchange."""
    path: str
    action: str  # read | modified | created | deleted | discussed


@dataclass
class DistilledObject:
    """A distilled representation of a conversation exchange."""
    object_id: str
    project_id: str
    conversation_id: str
    ply_start: int
    ply_end: int
    files_touched: List[FileTouched]
    exchange_core: str
    specific_context: str
    created_at: datetime
    exchange_at: datetime
    embedding_id: int
    distilled_text: str
    conv_title: Optional[str] = None


@dataclass
class Room:
    """A thematic room in the memory palace - a locus for related distillations."""
    room_id: str
    room_type: str  # file | module | concept | tool | workflow
    room_key: str
    room_label: str
    project_id: Optional[str]
    created_at: datetime
    updated_at: datetime
    object_count: int


@dataclass
class RoomObject:
    """Junction record linking a room to a distilled object."""
    room_id: str
    object_id: str
    relevance: float
    placed_at: datetime


@dataclass
class DistillationStats:
    """Statistics from a distillation run."""
    conversations_processed: int
    objects_created: int
    rooms_created: int
    rooms_updated: int
    distillation_time_seconds: float


# ============================================================================
# Unified Search Models
# ============================================================================

@dataclass
class PalaceSearchResult:
    """Search result from palace layer with full metadata."""
    object_id: str
    conversation_id: str
    project_id: str
    ply_start: int
    ply_end: int
    exchange_core: str
    specific_context: str
    files_touched: List[FileTouched]
    rooms: List[Room]
    score: float
    keyword_score: float = 0.0
    semantic_score: float = 0.0


@dataclass
class UnifiedSearchResult:
    """Merged search result from palace and verbatim layers."""
    conversation_id: str
    project_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int
    file_path: str
    combined_score: float

    # Palace layer data (optional - only if palace hit)
    palace_score: Optional[float] = None
    palace_summary: Optional[str] = None
    palace_context: Optional[str] = None
    rooms: List[Room] = field(default_factory=list)
    files_touched: List[FileTouched] = field(default_factory=list)
    ply_start: Optional[int] = None
    ply_end: Optional[int] = None
    object_id: Optional[str] = None

    # Verbatim layer data (optional - only if verbatim hit)
    verbatim_score: Optional[float] = None
    verbatim_snippet: Optional[str] = None
    message_start_index: Optional[int] = None
    message_end_index: Optional[int] = None

    # Sub-scores for analysis (all 4 components)
    palace_bm25_score: Optional[float] = None
    palace_semantic_score: Optional[float] = None
    verbatim_bm25_score: Optional[float] = None
    verbatim_semantic_score: Optional[float] = None

    # Progressive fallback tracking
    fallback_tier: Optional[str] = None  # "scoped" | "related" | "unscoped" | None

    @property
    def has_palace(self) -> bool:
        """Check if result has palace layer data."""
        return self.palace_score is not None

    @property
    def has_verbatim(self) -> bool:
        """Check if result has verbatim layer data."""
        return self.verbatim_score is not None

    @property
    def is_intersection(self) -> bool:
        """Check if result appears in both layers."""
        return self.has_palace and self.has_verbatim
