"""PyArrow schemas for Parquet storage."""
import pyarrow as pa


CONVERSATION_SCHEMA = pa.schema([
    ('conversation_id', pa.string()),
    ('project_id', pa.string()),
    ('file_path', pa.string()),
    ('title', pa.string()),
    ('created_at', pa.timestamp('us')),
    ('updated_at', pa.timestamp('us')),
    ('message_count', pa.int32()),
    ('messages', pa.list_(
        pa.struct([
            ('sequence', pa.int32()),
            ('role', pa.string()),
            ('content', pa.string()),
            ('timestamp', pa.timestamp('us')),
            ('has_code', pa.bool_()),
            ('code_blocks', pa.list_(pa.string()))
        ])
    )),
    ('full_text', pa.string()),
    ('embedding_id', pa.int64()),
    ('file_hash', pa.string()),
    ('indexed_at', pa.timestamp('us'))
])

METADATA_SCHEMA = pa.schema([
    ('vector_id', pa.int64()),
    ('conversation_id', pa.string()),
    ('project_id', pa.string()),
    ('chunk_index', pa.int32()),
    ('chunk_text', pa.string()),
    ('message_start_index', pa.int32()),
    ('message_end_index', pa.int32()),
    ('created_at', pa.timestamp('us'))
])


# ============================================================================
# Memory Palace Distillation Schemas
# ============================================================================

DISTILLED_OBJECT_SCHEMA = pa.schema([
    ('object_id', pa.string()),
    ('project_id', pa.string()),
    ('conversation_id', pa.string()),
    ('ply_start', pa.int32()),
    ('ply_end', pa.int32()),
    ('files_touched', pa.list_(
        pa.struct([
            ('path', pa.string()),
            ('action', pa.string()),
        ])
    )),
    ('exchange_core', pa.string()),
    ('specific_context', pa.string()),
    ('created_at', pa.timestamp('us')),
    ('exchange_at', pa.timestamp('us')),
    ('embedding_id', pa.int64()),
    ('distilled_text', pa.string()),
])

ROOM_SCHEMA = pa.schema([
    ('room_id', pa.string()),
    ('room_type', pa.string()),
    ('room_key', pa.string()),
    ('room_label', pa.string()),
    ('project_id', pa.string()),
    ('created_at', pa.timestamp('us')),
    ('updated_at', pa.timestamp('us')),
    ('object_count', pa.int32()),
])

ROOM_OBJECT_SCHEMA = pa.schema([
    ('room_id', pa.string()),
    ('object_id', pa.string()),
    ('relevance', pa.float32()),
    ('placed_at', pa.timestamp('us')),
])

DISTILLED_METADATA_SCHEMA = pa.schema([
    ('vector_id', pa.int64()),
    ('object_id', pa.string()),
    ('project_id', pa.string()),
    ('chunk_index', pa.int32()),
    ('chunk_text', pa.string()),
    ('created_at', pa.timestamp('us')),
])
