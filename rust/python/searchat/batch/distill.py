"""Batch distillation script invoked by the Rust binary.

Runs LLM distillation on undistilled conversations, writing palace
objects to the shared DuckDB database.

Usage (called by Rust subprocess):
    python -m searchat.batch.distill --data-dir ~/.searchat/data

Outputs newline-delimited JSON progress to stdout.
"""

import argparse
import json
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Batch distill conversations")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to ~/.searchat/data containing searchat.duckdb",
    )
    parser.add_argument(
        "--conversation-id",
        type=str,
        default=None,
        help="Distill a single conversation by ID",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum conversations to distill (0 = unlimited)",
    )
    args = parser.parse_args()

    from searchat.config import Config
    from searchat.core.unified_storage import UnifiedStorage
    from searchat.palace.distiller import Distiller
    from searchat.palace.llm import CLIDistillationLLM

    config = Config.load()
    storage = UnifiedStorage(args.data_dir)
    llm = CLIDistillationLLM(config=config)
    distiller = Distiller(
        data_dir=args.data_dir.parent,
        config=config,
        storage=storage,
        llm=llm,
    )

    emit({"status": "starting"})
    start = time.time()

    if args.conversation_id:
        stats = distiller.distill_conversation(args.conversation_id)
    else:
        stats = distiller.distill_all_pending(limit=args.limit or None)

    elapsed = time.time() - start

    result = {
        "status": "complete",
        "elapsed_seconds": round(elapsed, 2),
        "conversations_processed": stats.conversations_processed,
        "objects_created": stats.objects_created,
        "rooms_created": stats.rooms_created,
        "rooms_updated": stats.rooms_updated,
    }
    emit(result)


def emit(obj: dict):
    print(json.dumps(obj), flush=True)


if __name__ == "__main__":
    main()
