#!/usr/bin/env python3
"""Benchmark search modes across both layers.

Compares:
- Verbatim modes: hybrid, keyword, semantic
- Palace modes: hybrid (current), keyword-only, semantic-only
- Unified search with different combinations

Metrics:
- Latency (p50, p95, p99)
- Result counts (total, palace, verbatim, intersection)
- Score distributions
"""
import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import requests

API_BASE = "http://localhost:8000/api"

# Test queries with expected characteristics
TEST_QUERIES = [
    # Exact term queries (keyword should excel)
    {"query": "IndexFlatL2", "type": "exact_term"},
    {"query": "BM25Okapi", "type": "exact_term"},
    {"query": "compactor.py", "type": "file_path"},
    {"query": "palace.duckdb", "type": "file_path"},

    # Conceptual queries (semantic should excel)
    {"query": "vector similarity search", "type": "conceptual"},
    {"query": "conversation compaction", "type": "conceptual"},
    {"query": "search ranking algorithm", "type": "conceptual"},

    # Mixed queries (hybrid should excel)
    {"query": "FAISS vector search", "type": "mixed"},
    {"query": "BM25 keyword ranking", "type": "mixed"},
    {"query": "palace room assignment", "type": "mixed"},

    # Error/code patterns
    {"query": "KeyError object not found", "type": "error"},
    {"query": "TypeError NoneType", "type": "error"},
]


@dataclass
class SearchResult:
    """Single benchmark result."""
    query: str
    mode: str
    latency_ms: float
    total_results: int
    palace_count: int
    verbatim_count: int
    intersection_count: int
    top_score: float
    error: Optional[str] = None


def run_search(query: str, mode: str = "hybrid", limit: int = 50) -> SearchResult:
    """Execute a single search and collect metrics."""
    start = time.perf_counter()

    try:
        resp = requests.get(
            f"{API_BASE}/search",
            params={"q": query, "mode": mode, "limit": limit},
            timeout=30,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        if resp.status_code != 200:
            return SearchResult(
                query=query,
                mode=mode,
                latency_ms=latency_ms,
                total_results=0,
                palace_count=0,
                verbatim_count=0,
                intersection_count=0,
                top_score=0.0,
                error=f"HTTP {resp.status_code}: {resp.text[:100]}",
            )

        data = resp.json()
        results = data.get("results", [])

        intersection_count = sum(1 for r in results if r.get("is_intersection", False))
        top_score = results[0]["combined_score"] if results else 0.0

        return SearchResult(
            query=query,
            mode=mode,
            latency_ms=latency_ms,
            total_results=data.get("total", 0),
            palace_count=data.get("palace_count", 0),
            verbatim_count=data.get("verbatim_count", 0),
            intersection_count=intersection_count,
            top_score=top_score,
        )
    except Exception as e:
        return SearchResult(
            query=query,
            mode=mode,
            latency_ms=(time.perf_counter() - start) * 1000,
            total_results=0,
            palace_count=0,
            verbatim_count=0,
            intersection_count=0,
            top_score=0.0,
            error=str(e),
        )


def run_benchmark(
    queries: List[dict],
    modes: List[str],
    iterations: int = 3,
) -> List[SearchResult]:
    """Run benchmark across all queries and modes."""
    results = []

    total = len(queries) * len(modes) * iterations
    current = 0

    for query_info in queries:
        query = query_info["query"]
        for mode in modes:
            for i in range(iterations):
                current += 1
                print(f"\r[{current}/{total}] {mode}: {query[:30]}...", end="", flush=True)

                result = run_search(query, mode)
                results.append(result)

                # Small delay to avoid overwhelming the server
                time.sleep(0.1)

    print()
    return results


def analyze_results(results: List[SearchResult]) -> dict:
    """Analyze benchmark results by mode."""
    by_mode = {}

    for r in results:
        if r.mode not in by_mode:
            by_mode[r.mode] = []
        by_mode[r.mode].append(r)

    analysis = {}
    for mode, mode_results in by_mode.items():
        successful = [r for r in mode_results if r.error is None]
        latencies = [r.latency_ms for r in successful]

        if not latencies:
            continue

        analysis[mode] = {
            "count": len(mode_results),
            "errors": len(mode_results) - len(successful),
            "latency": {
                "mean": statistics.mean(latencies),
                "median": statistics.median(latencies),
                "p95": sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) >= 20 else max(latencies),
                "min": min(latencies),
                "max": max(latencies),
            },
            "results": {
                "avg_total": statistics.mean([r.total_results for r in successful]),
                "avg_palace": statistics.mean([r.palace_count for r in successful]),
                "avg_verbatim": statistics.mean([r.verbatim_count for r in successful]),
                "avg_intersection": statistics.mean([r.intersection_count for r in successful]),
            },
            "scores": {
                "avg_top": statistics.mean([r.top_score for r in successful if r.top_score > 0]),
            },
        }

    return analysis


def print_comparison(analysis: dict):
    """Print formatted comparison table."""
    print("\n" + "=" * 80)
    print("SEARCH MODE COMPARISON")
    print("=" * 80)

    # Latency comparison
    print("\n### Latency (ms)")
    print(f"{'Mode':<12} {'Mean':>10} {'Median':>10} {'P95':>10} {'Min':>10} {'Max':>10}")
    print("-" * 62)
    for mode, data in sorted(analysis.items()):
        lat = data["latency"]
        print(f"{mode:<12} {lat['mean']:>10.1f} {lat['median']:>10.1f} {lat['p95']:>10.1f} {lat['min']:>10.1f} {lat['max']:>10.1f}")

    # Results comparison
    print("\n### Average Results")
    print(f"{'Mode':<12} {'Total':>10} {'Palace':>10} {'Verbatim':>10} {'Intersect':>10}")
    print("-" * 52)
    for mode, data in sorted(analysis.items()):
        res = data["results"]
        print(f"{mode:<12} {res['avg_total']:>10.1f} {res['avg_palace']:>10.1f} {res['avg_verbatim']:>10.1f} {res['avg_intersection']:>10.1f}")

    # Score comparison
    print("\n### Average Top Score")
    print(f"{'Mode':<12} {'Avg Top Score':>15}")
    print("-" * 27)
    for mode, data in sorted(analysis.items()):
        print(f"{mode:<12} {data['scores']['avg_top']:>15.4f}")

    print("\n" + "=" * 80)


def print_query_breakdown(results: List[SearchResult], queries: List[dict]):
    """Print per-query breakdown by type."""
    print("\n### Results by Query Type")

    query_types = {}
    for q in queries:
        qtype = q["type"]
        if qtype not in query_types:
            query_types[qtype] = []
        query_types[qtype].append(q["query"])

    for qtype, qlist in query_types.items():
        print(f"\n**{qtype}** queries:")

        for query in qlist:
            query_results = [r for r in results if r.query == query and r.error is None]
            if not query_results:
                continue

            print(f"  '{query[:40]}'")
            for mode in ["hybrid", "keyword", "semantic"]:
                mode_results = [r for r in query_results if r.mode == mode]
                if mode_results:
                    avg_total = statistics.mean([r.total_results for r in mode_results])
                    avg_lat = statistics.mean([r.latency_ms for r in mode_results])
                    print(f"    {mode:<10}: {avg_total:>5.0f} results, {avg_lat:>6.1f}ms")


def main():
    parser = argparse.ArgumentParser(description="Benchmark search modes")
    parser.add_argument("--iterations", "-n", type=int, default=3, help="Iterations per query/mode")
    parser.add_argument("--modes", "-m", nargs="+", default=["hybrid", "keyword", "semantic"])
    parser.add_argument("--output", "-o", type=str, help="JSON output file")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print(f"Running benchmark with {args.iterations} iterations per query/mode...")
    print(f"Modes: {args.modes}")
    print(f"Queries: {len(TEST_QUERIES)}")
    print()

    # Check server is running
    try:
        resp = requests.get(f"{API_BASE}/projects", timeout=5)
        if resp.status_code != 200:
            print(f"Server not responding correctly: {resp.status_code}")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print(f"Cannot connect to {API_BASE}. Is the server running?")
        sys.exit(1)

    # Run benchmark
    results = run_benchmark(TEST_QUERIES, args.modes, args.iterations)

    # Analyze
    analysis = analyze_results(results)

    # Print comparison
    print_comparison(analysis)

    if args.verbose:
        print_query_breakdown(results, TEST_QUERIES)

    # Save to file
    if args.output:
        output_data = {
            "config": {
                "iterations": args.iterations,
                "modes": args.modes,
                "query_count": len(TEST_QUERIES),
            },
            "analysis": analysis,
            "raw_results": [
                {
                    "query": r.query,
                    "mode": r.mode,
                    "latency_ms": r.latency_ms,
                    "total_results": r.total_results,
                    "palace_count": r.palace_count,
                    "verbatim_count": r.verbatim_count,
                    "intersection_count": r.intersection_count,
                    "top_score": r.top_score,
                    "error": r.error,
                }
                for r in results
            ],
        }
        Path(args.output).write_text(json.dumps(output_data, indent=2))
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
