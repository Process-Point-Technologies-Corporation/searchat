"""Batch indexing script invoked by the Rust binary.

Scans for new/changed conversation files, generates embeddings via
sentence-transformers, and writes results to the shared DuckDB database.

Usage (called by Rust subprocess):
    python -m searchat.batch.index_missing --data-dir ~/.searchat/data

Outputs newline-delimited JSON progress to stdout.
Exits with code 0 on success, 1 on error.
"""

import argparse
import json
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Batch index missing conversations")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to ~/.searchat/data containing searchat.duckdb",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum files to process (0 = unlimited)",
    )
    args = parser.parse_args()

    # Import the existing searchat package (must be installed via uv pip install -e .)
    from searchat.config import Config
    from searchat.core.unified_indexer import UnifiedIndexer
    from searchat.core.unified_storage import UnifiedStorage
    from searchat.core.watcher import ConversationWatcher
    from searchat.core.conversation_filter import exclude_automated_conversations

    config = Config.load()
    storage = UnifiedStorage(args.data_dir)
    indexer = UnifiedIndexer(args.data_dir, config=config, storage=storage)

    # Use watcher's cached file scan for speed
    watcher = ConversationWatcher(config=config)
    watcher.scan_all_files()
    known_files = watcher.get_known_files()

    if known_files is None:
        emit({"status": "error", "message": "No files found"})
        sys.exit(1)

    # Pre-scan: exclude automated conversations
    excluded_dir = config.paths.excluded_conversations_dir
    if excluded_dir:
        exclude_automated_conversations(
            list(known_files),
            excluded_dir,
            config,
        )
        # Re-scan after exclusion
        watcher.scan_all_files()
        known_files = watcher.get_known_files()

    file_paths = sorted(known_files) if known_files else []
    if args.limit > 0:
        file_paths = file_paths[: args.limit]

    emit({"status": "scanning", "total_files": len(file_paths)})

    start = time.time()
    stats = indexer.index_from_source_files(file_paths)
    elapsed = time.time() - start

    # Rebuild FTS index after new data
    try:
        storage.create_fts_index()
    except Exception:
        pass

    result = {
        "status": "complete",
        "elapsed_seconds": round(elapsed, 2),
    }
    result.update(stats)
    emit(result)


def emit(obj: dict):
    """Write a JSON line to stdout for the Rust parent process."""
    print(json.dumps(obj), flush=True)


if __name__ == "__main__":
    main()
