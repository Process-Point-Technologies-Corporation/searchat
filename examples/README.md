# Searchat Examples

These examples are intended for a repo checkout with Searchat already configured.
Run `python -m searchat.setup` first and make sure you already have indexed conversations.

## Current Examples

- `basic_search.py`
  Runs one cross-layer search using `UnifiedSearchEngine`.

- `advanced_search.py`
  Demonstrates filters and compares `cross-layer`, `verbatim`, and `distill` modes.

- `api_integration.py`
  Shows a small wrapper around the public library API for embedding Searchat in another tool.

- `batch_operations.py`
  Demonstrates simple export/reporting workflows using the current engine.

- `custom_indexing.py`
  Inspects the unified storage and reports newly discovered source files.

## Config Examples

The `config_examples/` directory contains sample env/TOML files for local setup patterns.
Treat them as templates, not drop-in production configs.

## Notes

- These examples are repo examples, not installed CLI commands.
- They should reflect the current public API surface.
- If an example stops matching the product, it should be rewritten or removed.
