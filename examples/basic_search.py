"""Basic Search Example.

Run a single cross-layer query against an existing Searchat index.

Usage:
    python examples/basic_search.py
"""

from searchat import UnifiedSearchEngine
from searchat.config import Config, PathResolver
from searchat.models import AlgorithmType


def main() -> None:
    config = Config.load()
    search_dir = PathResolver.get_shared_search_dir(config)
    engine = UnifiedSearchEngine(search_dir, config)

    query = "refactoring"
    results = engine.search(query, algorithm=AlgorithmType.CROSS_LAYER, limit=5)

    print(f"Searching for: {query!r}")
    print("=" * 70)
    for idx, result in enumerate(results.results, 1):
        print(f"{idx}. {result.title}")
        print(f"   Score: {result.score:.3f}")
        print(f"   Project: {result.project_id}")
        print(f"   Path: {result.file_path}")
        print(f"   Snippet: {result.snippet[:160]}...")
        print()

    print(f"Total results: {results.total_count}")


if __name__ == "__main__":
    main()
