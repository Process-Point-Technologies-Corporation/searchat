"""Library Integration Example.

Shows one small wrapper around the public UnifiedSearchEngine API.

Usage:
    python examples/api_integration.py
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from searchat import UnifiedSearchEngine
from searchat.config import Config, PathResolver
from searchat.models import AlgorithmType, SearchFilters, SearchResult


class ConversationSearchAPI:
    """Small wrapper for embedding Searchat in other tools."""

    def __init__(self, config_path: Optional[Path] = None):
        self.config = Config.load(config_path)
        search_dir = PathResolver.get_shared_search_dir(self.config)
        self.engine = UnifiedSearchEngine(search_dir, self.config)

    def quick_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        results = self.engine.search(query, algorithm=AlgorithmType.CROSS_LAYER, limit=limit)
        return [
            {
                "title": r.title,
                "score": round(r.score, 3),
                "snippet": r.snippet,
                "path": r.file_path,
                "project_id": r.project_id,
            }
            for r in results.results
        ]

    def search_by_mode(
        self,
        query: str,
        mode: str = "cross-layer",
        limit: int = 10,
        filters: Optional[SearchFilters] = None,
    ) -> List[SearchResult]:
        mode_map = {
            "cross-layer": AlgorithmType.CROSS_LAYER,
            "verbatim": AlgorithmType.KEYWORD,
            "distill": AlgorithmType.DISTILL,
        }
        algorithm = mode_map[mode]
        return self.engine.search(query, algorithm=algorithm, filters=filters, limit=limit).results


def example_usage() -> None:
    api = ConversationSearchAPI()

    print("Searchat Integration Example")
    print("=" * 70)

    print("1. Quick search")
    for item in api.quick_search("database optimization", limit=3):
        print(f"- {item['title']} ({item['score']})")

    print()
    print("2. Verbatim search")
    for result in api.search_by_mode("error handling", mode="verbatim", limit=3):
        print(f"- {result.title} ({result.score:.3f})")

    print()
    print("3. Distilled search")
    for result in api.search_by_mode("authentication flow", mode="distill", limit=3):
        print(f"- {result.title} ({result.score:.3f})")


if __name__ == "__main__":
    example_usage()
