"""
Custom indexing and storage inspection example for Searchat.

This example shows how to inspect the current unified storage and compare it
against transcript files discoverable from the configured providers.

Usage:
    python examples/custom_indexing.py
"""

from searchat.agents import iter_providers
from searchat.config import Config, PathResolver
from searchat.core.conversation_filter import exclude_automated_conversations
from searchat.core.unified_indexer import UnifiedIndexer


def build_indexer() -> UnifiedIndexer:
    config = Config.load()
    search_dir = PathResolver.get_shared_search_dir(config)
    return UnifiedIndexer(search_dir, config)


def print_storage_summary(indexer: UnifiedIndexer) -> None:
    stats = indexer.storage.get_stats()
    print("Storage Summary")
    print("=" * 70)
    print(f"Conversations: {stats.get('conversations', 0)}")
    print(f"Messages: {stats.get('messages', 0)}")
    print(f"Exchanges: {stats.get('exchanges', 0)}")
    print(f"Palace objects: {stats.get('palace_objects', 0)}")
    print(f"Rooms: {stats.get('rooms', 0)}")
    print()


def list_indexed_conversations(indexer: UnifiedIndexer, limit: int = 10) -> None:
    rows = indexer.storage._get_cursor().execute(
        """
        SELECT conversation_id, project_id, title, message_count, file_path
        FROM conversations
        ORDER BY updated_at DESC
        LIMIT ?
        """,
        [limit],
    ).fetchall()

    print("Recent Indexed Conversations")
    print("=" * 70)
    for idx, row in enumerate(rows, 1):
        print(f"{idx}. {row[2]}")
        print(f"   Conversation ID: {row[0]}")
        print(f"   Project: {row[1]}")
        print(f"   Messages: {row[3]}")
        print(f"   Path: {row[4]}")
        print()


def find_unindexed_source_files(indexer: UnifiedIndexer) -> None:
    config = indexer.config
    discovered: list[str] = []
    for provider in iter_providers():
        for root_dir in provider.discover_dirs(config):
            pattern = "*.json" if provider.agent_id == "vibe" else "*.jsonl"
            files = list(root_dir.glob(pattern)) if provider.agent_id == "vibe" else list(root_dir.rglob(pattern))
            discovered.extend(str(path) for path in files)

    discovered = exclude_automated_conversations(
        discovered,
        config.paths.excluded_conversations_dir,
        config,
    )
    new_files, changed_files = indexer.detect_changed_files(discovered)

    print("Source File Status")
    print("=" * 70)
    print(f"Discovered source files: {len(discovered)}")
    print(f"New files: {len(new_files)}")
    print(f"Changed files: {len(changed_files)}")
    if new_files[:10]:
        print()
        print("Sample new files:")
        for path in new_files[:10]:
            print(f"- {path}")


if __name__ == "__main__":
    idx = build_indexer()
    print_storage_summary(idx)
    list_indexed_conversations(idx)
    find_unindexed_source_files(idx)
