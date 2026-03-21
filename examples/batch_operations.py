"""Batch Operations Example.

Shows simple reporting/export tasks using the public UnifiedSearchEngine API.

Usage:
    python examples/batch_operations.py
"""

import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

from searchat import UnifiedSearchEngine
from searchat.config import Config, PathResolver
from searchat.models import AlgorithmType, SearchFilters


def build_engine() -> UnifiedSearchEngine:
    config = Config.load()
    search_dir = PathResolver.get_shared_search_dir(config)
    return UnifiedSearchEngine(search_dir, config)


def export_search_results(query: str, output_file: str) -> None:
    engine = build_engine()
    results = engine.search(query, algorithm=AlgorithmType.CROSS_LAYER, limit=50)

    payload = {
        "query": query,
        "generated_at": datetime.now().isoformat(),
        "total_results": results.total_count,
        "results": [
            {
                "title": r.title,
                "project_id": r.project_id,
                "score": r.score,
                "file_path": r.file_path,
                "snippet": r.snippet,
            }
            for r in results.results
        ],
    }

    Path(output_file).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {results.total_count} results to {output_file}")


def topic_summary() -> None:
    engine = build_engine()
    topics = ["python", "testing", "database", "security"]
    counts = {}
    for topic in topics:
        counts[topic] = engine.search(topic, algorithm=AlgorithmType.CROSS_LAYER, limit=25).total_count

    print("Topic summary")
    print("=" * 70)
    for topic, count in sorted(counts.items(), key=lambda item: item[1], reverse=True):
        print(f"{topic:12} {count}")


def conversations_from_last_week() -> None:
    engine = build_engine()
    filters = SearchFilters(
        date_from=datetime.now() - timedelta(days=7),
        date_to=datetime.now(),
    )
    results = engine.search("", algorithm=AlgorithmType.KEYWORD, filters=filters, limit=100)

    by_project = Counter(r.project_id for r in results.results)
    print()
    print("Last-week conversations by project")
    print("=" * 70)
    for project_id, count in by_project.most_common():
        print(f"{project_id}: {count}")


if __name__ == "__main__":
    export_search_results("python testing", "search_results.json")
    topic_summary()
    conversations_from_last_week()
