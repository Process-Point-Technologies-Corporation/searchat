<p align="center">
  <img src="assets/logo.svg" alt="searchat" width="400">
</p>

> [!NOTE]
> **Next version in progress.** Searchat v2 is under active development. The current release remains functional but the upcoming version includes significant changes. Watch this repo or check back for the release.

<p align="center">

[![Built with Claude Code](https://img.shields.io/badge/Built_with-Claude_Code-D97757?logo=claude&logoColor=fff)](https://claude.ai/code)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</p>

<p align="center">Semantic search for AI coding agent conversations. Find past solutions by meaning, not just keywords.</p>

## Supported Agents

| Agent | Location | Format |
|-------|----------|--------|
| Claude Code | `~/.claude/projects/**/*.jsonl` | JSONL |
| Mistral Vibe | `~/.vibe/logs/session/*.json` | JSON |

## Features

- **Hybrid Search** — BM25 keyword + FAISS semantic vectors
- **Multi-Agent** — Search across Claude Code and Mistral Vibe sessions
- **Live Indexing** — Auto-indexes new/modified files (5min debounce for in-progress)
- **Append-Only** — Never deletes existing data, safe for long-term use
- **Self-Search** — Agents can search their own history via API
- **Safe Shutdown** — Detects ongoing indexing, prevents data corruption
- **Cross-Platform** — Windows, WSL, Linux, macOS
- **Local-First** — All data stays on your machine

## Quick Start

```bash
git clone https://github.com/Process-Point-Technologies-Corporation/searchat.git
cd searchat
pip install -e .

# First-time setup: build search index
python scripts/setup-index

# Start web server
searchat-web
```

Open http://localhost:8000

The setup script indexes all conversations from Claude Code and Mistral Vibe. On subsequent runs, the web server automatically indexes new conversations via live file watching.

## Enable Claude Self-Search

Add to `~/.claude/CLAUDE.md`:

```markdown
## Conversation History Search

Search past Claude Code conversations via local API (requires server running).

**Search:**
\`\`\`bash
curl -s "http://localhost:8000/api/search?q=QUERY&limit=5" | jq '.results[] | {id: .conversation_id, title, snippet}'
\`\`\`

**Get full conversation:**
\`\`\`bash
curl -s "http://localhost:8000/api/conversation/CONVERSATION_ID" | jq '.messages[] | {role, content: .content[:500]}'
\`\`\`

**When to use:**
- User asks "did we discuss X before" or "find that conversation about Y"
- Looking for previous solutions to similar problems
- Checking how something was implemented in past sessions

**Start server:** `searchat-web` from the searchat directory
```

See `CLAUDE.example.md` for the full template.

## Usage

### Web UI

```bash
searchat-web
```

Features:
- Search modes: hybrid/semantic/keyword
- Filter by project, date range
- View full conversations
- Add missing conversations button (safe append)
- Stop server button (checks for ongoing indexing)
- Helpful tips sidebars (search tips + API integration guide)

### CLI

```bash
searchat "search query"
searchat  # interactive mode
```

### API

```bash
# Search
curl "http://localhost:8000/api/search?q=authentication&mode=hybrid&limit=10"

# Get conversation
curl "http://localhost:8000/api/conversation/{conversation_id}"

# List projects
curl "http://localhost:8000/api/projects"

# Statistics
curl "http://localhost:8000/api/statistics"

# Watcher status
curl "http://localhost:8000/api/watcher/status"

# Index missing conversations (append-only)
curl -X POST "http://localhost:8000/api/index_missing"

# Safe shutdown (checks for ongoing indexing)
curl -X POST "http://localhost:8000/api/shutdown"

# Force shutdown (override safety check)
curl -X POST "http://localhost:8000/api/shutdown?force=true"
```

### Utilities

```bash
# Add missing conversations to index
python scripts/index-missing

# Initial setup (interactive, safe options)
python scripts/setup-index

# Convert Vibe plaintext history to searchable sessions
python utils/vibe_converter.py
```

### As Library

```python
from searchat.search_engine import SearchEngine
from searchat.config import Config

config = Config.load()
engine = SearchEngine(config.paths.search_directory, config)

results = engine.search("python async", mode="hybrid")
for r in results.results[:5]:
    print(f"{r.title}: {r.score:.3f}")
```

## Architecture

**Code Organization:**
- `src/searchat/api/` - FastAPI app with 6 modular routers (15 endpoints)
- `src/searchat/core/` - Business logic (indexer, search_engine, watcher)
- `src/searchat/web/` - Modular frontend (HTML + CSS modules + ES6 JS)
- `tests/api/` - Comprehensive API tests (62 tests)

**Data Flow:**
```
~/.claude/projects/**/*.jsonl     (source conversations)
        │
        ▼ index_append_only()
        │
~/.searchat/data/
├── conversations/*.parquet       (conversation data, DuckDB queryable)
└── indices/
    ├── embeddings.faiss          (semantic vectors)
    ├── embeddings.metadata.parquet
    └── index_metadata.json
```

**Search Flow:**
1. Query → BM25 keyword search + FAISS semantic search
2. Results merged via Reciprocal Rank Fusion
3. Hybrid ranking returns best of both approaches

**Live Watching:**
- `watchdog` monitors conversation directories
- New files → indexed immediately
- Modified files → re-indexed after 5min debounce (configurable)
- `index_append_only()` adds to existing index
- Never deletes existing data

**Documentation:**
- `docs/architecture.md` - System design and components
- `docs/api-reference.md` - Complete API endpoint documentation
- `docs/terminal-launching.md` - Platform-specific terminal launching

## Configuration

Create `~/.searchat/config/settings.toml`:

```toml
[paths]
search_directory = "~/.searchat"
claude_directory_windows = "~/.claude/projects"
claude_directory_wsl = "//wsl$/Ubuntu/home/{username}/.claude/projects"

[indexing]
batch_size = 1000
auto_index = true
reindex_on_modification = true  # Re-index modified conversations
modification_debounce_minutes = 5  # Wait time before re-indexing

[search]
default_mode = "hybrid"
max_results = 100
snippet_length = 200

[embedding]
model = "all-MiniLM-L6-v2"
batch_size = 32

[performance]
memory_limit_mb = 3000
query_cache_size = 100
```

Or use environment variables:

```bash
export SEARCHAT_DATA_DIR=~/.searchat
export SEARCHAT_PORT=8000
export SEARCHAT_EMBEDDING_MODEL=all-MiniLM-L6-v2
export SEARCHAT_REINDEX_ON_MODIFICATION=true
export SEARCHAT_MODIFICATION_DEBOUNCE_MINUTES=5
```

## Requirements

- Python 3.9+
- ~2-3GB RAM (embeddings model + FAISS index)
- ~10MB disk per 1K conversations

### Dependencies

| Package | Purpose |
|---------|---------|
| sentence-transformers | Embeddings (all-MiniLM-L6-v2) |
| faiss-cpu | Vector similarity search |
| pyarrow | Parquet storage |
| duckdb | SQL queries on parquet |
| fastapi + uvicorn | Web API |
| watchdog | File system monitoring |
| rich | CLI formatting |

## Safety

**Append-only indexing:** Never deletes existing data.

```python
indexer.index_append_only(file_paths)  # Safe: only adds new data
indexer.index_all()                     # Blocked if index exists
indexer.index_all(force=True)           # Explicit override required
```

**Safe shutdown:** Detects ongoing indexing operations.

```bash
# Check status, wait if indexing in progress
curl -X POST "http://localhost:8000/api/shutdown"

# Override safety check (may corrupt data)
curl -X POST "http://localhost:8000/api/shutdown?force=true"
```

Protects against:
- Data loss from deleted/moved source files
- Corrupted Parquet/FAISS files during indexing
- Inconsistent metadata from interrupted operations

## Performance

| Metric | Value |
|--------|-------|
| Search latency | <100ms (hybrid), <50ms (semantic), <30ms (keyword) |
| Filtered queries | <20ms (DuckDB predicate pushdown) |
| Index build | ~60s per 100K conversations |
| Embedding | Batched (CPU: 0.1s/conv, GPU: 0.008s/conv) |
| Memory | ~2-3GB |
| Startup | <3s |

## Troubleshooting

**Port in use:**
```bash
SEARCHAT_PORT=8001 searchat-web
```

**No conversations found:**
```bash
ls ~/.claude/projects/  # Verify conversations exist
```

**WSL not tracked:**
Configure `claude_directory_wsl` in `~/.searchat/config/settings.toml`:
```toml
claude_directory_wsl = "//wsl.localhost/Ubuntu/home/username/.claude/projects"
```

**Missing conversations after setup:**
```bash
python scripts/index-missing  # Index files not yet in search index
```

**Slow on WSL:**
Run from Windows Python or move repo to WSL filesystem (`~/projects/`).

**Import errors:**
```bash
pip install -e . --force-reinstall
```

**Empty environment variables override config:**
Remove empty values from `~/.searchat/config/.env` or set proper values.

## License

MIT
