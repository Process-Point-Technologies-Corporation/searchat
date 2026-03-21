# Searchat Test Suite

This directory contains tests for the public Searchat tool: API routes, indexing, unified search, palace storage/query behavior, hooks, and configuration handling.

## Quick Start

```bash
pip install -e ".[dev]"
pytest
```

## Layout

- `tests/api/`: FastAPI route coverage
- `tests/core/`: unified indexer, search, normalization, and storage
- `tests/hooks/`: hook invocation and transcript distillation behavior
- `tests/palace/`: distilled-memory storage, query, and index tests
- `tests/unit/`: focused config and low-level unit coverage
- `tests/fixtures/`: synthetic transcript fixtures used by the suite
- `tests/conftest.py`: shared mocks and fixtures

## Notes

- Heavy dependencies are mocked where practical so most tests run locally without external services.
- Coverage is opt-in. Use `pytest --cov=searchat --cov-report=term-missing` when you want a coverage report.
- Some tests exercise optional distillation and hook features because they are part of the public tool surface.

## Useful Commands

```bash
pytest tests/api/test_search_routes.py
pytest tests/core/test_unified_search.py
pytest -m "not slow"
pytest --cov=searchat --cov-report=html
```
