# Searchat Architecture

Searchat is a local-first search service for AI coding-agent transcripts. The public repo ships the product runtime: transcript discovery, indexing, unified storage, search APIs, and a small web UI.

## Main Runtime Pieces

- `src/searchat/agents/`: transcript providers for Claude Code, Codex, and Mistral Vibe
- `src/searchat/core/`: unified indexing, DuckDB-backed search, ranking, and watcher logic
- `src/searchat/palace/`: distilled-memory storage, retrieval, and helper indexes
- `src/searchat/api/`: FastAPI app, routers, and response models
- `src/searchat/web/`: static frontend served by the API
- `src/hooks/`: optional Claude-oriented hook utilities for per-turn distillation workflows

## Data Layout

Searchat stores local data under `~/.searchat/` by default:

```text
~/.searchat/
├── data/
│   ├── searchat.duckdb
│   ├── palace.duckdb
│   └── watcher_file_cache.json
└── config/
    ├── settings.toml
    └── .env
```

`searchat.duckdb` holds transcript, exchange, and search metadata. `palace.duckdb` holds distilled objects and related room/file metadata used by the distillation-aware search modes.

## Search Modes

- `cross-layer`: combines verbatim keyword search with distilled semantic retrieval
- `verbatim`: searches raw conversation text only
- `distill`: searches distilled objects only

The API also exposes deeper diagnostic endpoints for unified, scoped, fallback, and facet-weighted search flows.

## Operational Notes

- Indexing is append-safe by default and can watch transcript directories for changes.
- Distillation is optional. If palace storage does not exist yet, the verbatim paths still work.
