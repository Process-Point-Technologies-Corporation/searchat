#!/usr/bin/env python3
"""Quick test of search modes to compare results.

Run after starting server with: searchat-web
"""
import json
import sys
import time

import requests

API = "http://localhost:8000/api"


def test_search(query: str, mode: str = "hybrid") -> dict:
    """Run a search and return summary."""
    start = time.perf_counter()
    resp = requests.get(f"{API}/search", params={"q": query, "mode": mode, "limit": 10})
    elapsed = (time.perf_counter() - start) * 1000

    if resp.status_code != 200:
        return {"error": f"HTTP {resp.status_code}"}

    data = resp.json()
    results = data.get("results", [])

    # Check if unified search is active
    is_unified = "palace_count" in data

    summary = {
        "mode": mode,
        "latency_ms": round(elapsed, 1),
        "total": data.get("total", 0),
        "is_unified": is_unified,
    }

    if is_unified:
        summary["palace_count"] = data.get("palace_count", 0)
        summary["verbatim_count"] = data.get("verbatim_count", 0)

        palace_only = sum(1 for r in results if r.get("has_palace") and not r.get("has_verbatim"))
        verbatim_only = sum(1 for r in results if r.get("has_verbatim") and not r.get("has_palace"))
        intersection = sum(1 for r in results if r.get("is_intersection"))

        summary["top10_palace_only"] = palace_only
        summary["top10_verbatim_only"] = verbatim_only
        summary["top10_intersection"] = intersection

    if results:
        summary["top_score"] = round(results[0].get("combined_score", results[0].get("score", 0)), 4)
        summary["top_title"] = results[0].get("title", "")[:60]

    return summary


def main():
    queries = [
        ("FAISS", "exact term - library name"),
        ("vector similarity search", "conceptual query"),
        ("compactor.py", "file path"),
        ("BM25 keyword ranking", "mixed query"),
    ]

    modes = ["hybrid", "keyword", "semantic"]

    print("=" * 80)
    print("SEARCH MODE COMPARISON")
    print("=" * 80)

    # Check server
    try:
        resp = requests.get(f"{API}/projects", timeout=5)
        if resp.status_code != 200:
            print(f"Server error: {resp.status_code}")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("Server not running. Start with: searchat-web")
        sys.exit(1)

    for query, description in queries:
        print(f"\n### Query: '{query}' ({description})")
        print("-" * 60)

        for mode in modes:
            result = test_search(query, mode)

            if "error" in result:
                print(f"  {mode:<10}: ERROR - {result['error']}")
                continue

            if result["is_unified"]:
                print(f"  {mode:<10}: {result['total']:>3} results, {result['latency_ms']:>6.1f}ms "
                      f"| P:{result['palace_count']:>2} V:{result['verbatim_count']:>2} "
                      f"| top10: P-only:{result['top10_palace_only']} V-only:{result['top10_verbatim_only']} Both:{result['top10_intersection']}")
            else:
                print(f"  {mode:<10}: {result['total']:>3} results, {result['latency_ms']:>6.1f}ms "
                      f"| score={result.get('top_score', 'N/A')}")

    print("\n" + "=" * 80)

    # Check if unified search is active
    test = test_search("test", "hybrid")
    if not test.get("is_unified"):
        print("\nWARNING: Unified search not active!")
        print("Server is running old code. Restart server to enable unified search.")
        print("Run: taskkill /F /IM python.exe && searchat-web")


if __name__ == "__main__":
    main()
