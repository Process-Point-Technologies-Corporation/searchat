"""Data models and schemas for searchat."""
from searchat.models.enums import SearchMode, AlgorithmType
from searchat.models.domain import (
    MessageRecord,
    ConversationRecord,
    SearchFilters,
    SearchResult,
    SearchResults,
    IndexStats,
    IndexingStats,
    UpdateStats,
    DateFilter,
    ParsedQuery,
    FileTouched,
    DistilledObject,
    Room,
    RoomObject,
    DistillationStats,
)
from searchat.models.schemas import (
    CONVERSATION_SCHEMA,
    METADATA_SCHEMA,
    DISTILLED_OBJECT_SCHEMA,
    ROOM_SCHEMA,
    ROOM_OBJECT_SCHEMA,
    DISTILLED_METADATA_SCHEMA,
)

__all__ = [
    # Enums
    "SearchMode",
    "AlgorithmType",
    # Domain models
    "MessageRecord",
    "ConversationRecord",
    "SearchFilters",
    "SearchResult",
    "SearchResults",
    "IndexStats",
    "IndexingStats",
    "UpdateStats",
    "DateFilter",
    "ParsedQuery",
    # Palace domain models
    "FileTouched",
    "DistilledObject",
    "Room",
    "RoomObject",
    "DistillationStats",
    # Schemas
    "CONVERSATION_SCHEMA",
    "METADATA_SCHEMA",
    # Palace schemas
    "DISTILLED_OBJECT_SCHEMA",
    "ROOM_SCHEMA",
    "ROOM_OBJECT_SCHEMA",
    "DISTILLED_METADATA_SCHEMA",
]
