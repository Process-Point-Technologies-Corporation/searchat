"""Advanced Search Example.

Demonstrates filters and compares the current public search modes.

Usage:
    python examples/advanced_search.py
"""

from datetime import datetime, timedelta

from searchat import UnifiedSearchEngine
from searchat.config import Config, PathResolver
from searchat.models import AlgorithmType, SearchFilters


def search_with_filters() -> None:
    config = Config.load()
    search_dir = PathResolver.get_shared_search_dir(config)
    engine = UnifiedSearchEngine(search_dir, config)

    query = "API design"
    last_30_days = datetime.now() - timedelta(days=30)
    filters = SearchFilters(
        date_from=last_30_days,
        date_to=datetime.now(),
    )

    results = engine.search(query, algorithm=AlgorithmType.CROSS_LAYER, filters=filters, limit=10)

    print(f"Filtered search for: {query!r}")
    print(f"Date range: {last_30_days.date()} to {datetime.now().date()}")
    print("=" * 70)
    for idx, result in enumerate(results.results, 1):
        print(f"{idx}. {result.title} ({result.score:.3f})")
        print(f"   Updated: {result.updated_at}")
        print(f"   Snippet: {result.snippet[:140]}...")


def compare_modes() -> None:
    config = Config.load()
    search_dir = PathResolver.get_shared_search_dir(config)
    engine = UnifiedSearchEngine(search_dir, config)

    query = "error handling patterns"
    modes = [
        ("cross-layer", AlgorithmType.CROSS_LAYER),
        ("verbatim", AlgorithmType.KEYWORD),
        ("distill", AlgorithmType.DISTILL),
    ]

    print()
    print(f"Mode comparison for: {query!r}")
    print("=" * 70)
    for label, algorithm in modes:
        results = engine.search(query, algorithm=algorithm, limit=3)
        print(f"{label}:")
        for idx, result in enumerate(results.results, 1):
            print(f"  {idx}. {result.title} ({result.score:.3f})")
        print(f"  Total: {results.total_count}")
        print()


if __name__ == "__main__":
    search_with_filters()
    compare_modes()
