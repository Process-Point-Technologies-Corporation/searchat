<p align="center">
  <img src="assets/logo.svg" alt="searchat" width="400">
</p>

<p align="center">Semantic search for AI coding agent conversations. Find past solutions by meaning, not just keywords.</p>

## Supported Agents

| Agent | Location | Format |
|-------|----------|--------|
| Claude Code | `~/.claude/projects/**/*.jsonl` | JSONL |
| Codex | `~/.codex/sessions/**/*.jsonl` | JSONL |
| Mistral Vibe | `~/.vibe/logs/session/*.json` | JSON |

## Features

- **Cross-Layer Search** - Verbatim + distilled retrieval with unified ranking
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
pip install .

# First-time setup: create local config
python -m searchat.setup

# Start web server (startup catch-up indexing runs automatically)
searchat-web
```

Open http://localhost:8000

The setup wizard creates local configuration. On startup, the web server performs catch-up indexing for discovered transcripts and then keeps up with new conversations via live file watching.

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

**Start server:** `searchat-web`
```

See `CLAUDE.example.md` for the full template.

## Usage

### Web UI

```bash
searchat-web
```

Features:
- Search modes: cross-layer/verbatim/distill
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
curl "http://localhost:8000/api/search?q=authentication&mode=cross-layer&limit=10"

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

Installed commands:

```bash
# Initial setup (interactive configuration wizard)
python -m searchat.setup

# Web server
searchat-web

# Hardware profile detection
searchat-hardware --show
```

Repo-only helper scripts:

```bash
# Add missing conversations to index from a repo checkout
python scripts/index-missing

# Convert Vibe plaintext history to searchable sessions from a repo checkout
python utils/vibe_converter.py
```

### As Library

```python
from pathlib import Path

from searchat import UnifiedSearchEngine
from searchat.config import Config
from searchat.models import AlgorithmType

config = Config.load()
engine = UnifiedSearchEngine(Path(config.paths.search_directory), config)

results = engine.search("python async", algorithm=AlgorithmType.CROSS_LAYER, limit=5)
for r in results.results:
    print(f"{r.title}: {r.score:.3f}")
```

## Architecture

**Code Organization:**
- `src/searchat/api/` - FastAPI app with modular routers
- `src/searchat/core/` - Business logic (indexer, unified_search, watcher)
- `src/searchat/web/` - Modular frontend (HTML + CSS modules + ES6 JS)
- `tests/api/` - API and watcher behavior tests

**Data Flow:**
```
~/.searchat/data/
├── searchat.duckdb               (conversations, messages, exchanges, embeddings)
├── palace.duckdb                 (distilled objects and rooms)
└── watcher_file_cache.json       (startup bootstrap cache)
```

**Search Flow:**
1. Query → DuckDB keyword search + semantic vector search
2. Results merged in the unified engine
3. Cross-layer ranking returns the best combined results

**Live Watching:**
- `watchdog` monitors conversation directories
- New files → indexed immediately
- Modified files → re-indexed after 5min debounce (configurable)
- Changed Claude/Codex JSONL files use append-only tail reindex when possible
- Never deletes source conversations; index updates remain append-safe

**Documentation:**
- `docs/architecture.md` - System design and components
- `docs/api-reference.md` - Public API and search endpoint reference
- `docs/terminal-launching.md` - Platform-specific terminal launching

## Configuration

Searchat uses these config layers:
- `config/settings.default.toml` - full internal defaults
- `config/settings.template.toml` - simple user-facing starter config
- `~/.searchat/config/settings.toml` - your actual local settings

For most people, this is enough:

```toml
[paths]
search_directory = "~/.searchat"
claude_directory_windows = "~/.claude/projects"
claude_directory_wsl = ""

[indexing]
reindex_on_modification = true
modification_debounce_minutes = 5

[search]
default_mode = "cross-layer"
max_results = 100

[embedding]
device = "auto"
batch_size = 32

[performance]
startup_warmup_mode = "keyword"

[distillation]
provider = "claude"
cli_model = "claude-haiku-4-5-20251001"
```

Notes:
- Leave `claude_directory_wsl = ""` unless you actually use WSL transcripts.
- `startup_warmup_mode = "keyword"` is the best default for modest hardware.
- `device = "auto"` will use GPU if available, otherwise CPU.
- Distillation can use either subscription-backed CLI family:
  - `provider = "claude"` uses `claude --print`
  - `provider = "openai"` uses `codex exec`
- Good model picks:
  - Claude fast default: `claude-haiku-4-5-20251001`
  - OpenAI recommended default: `gpt-5.3-codex`
  - OpenAI fallback: `gpt-5`
  - OpenAI cheaper/faster fallback: `gpt-5.1-codex-mini`
- Advanced tuning still exists, but most users should not need to edit the full defaults surface.

The setup wizard writes this simple template automatically:

```bash
python -m searchat.setup
```

Environment variables still override config when needed:

```bash
export SEARCHAT_DATA_DIR=~/.searchat
export SEARCHAT_PORT=8000
export SEARCHAT_EMBEDDING_DEVICE=cpu
export SEARCHAT_REINDEX_ON_MODIFICATION=true
export SEARCHAT_MODIFICATION_DEBOUNCE_MINUTES=5
export SEARCHAT_STARTUP_WARMUP_MODE=none
```

## Requirements

- Python 3.9+
- ~2-3GB RAM for comfortable semantic search/distillation
- ~10MB disk per 1K conversations

### Dependencies

| Package | Purpose |
|---------|---------|
| sentence-transformers | Embeddings (all-MiniLM-L6-v2) |
| duckdb | Unified storage + SQL + FTS/VSS |
| fastapi + uvicorn | Web API |
| watchdog | File system monitoring |
| rich | CLI formatting |

## Safety

**Append-safe indexing:** Searchat never deletes your source transcripts. New files are indexed directly, and changed Claude/Codex transcripts use append-only tail reindex when possible.

**Safe shutdown:** Detects ongoing indexing operations.

```bash
# Check status, wait if indexing in progress
curl -X POST "http://localhost:8000/api/shutdown"

# Override safety check (may corrupt data)
curl -X POST "http://localhost:8000/api/shutdown?force=true"
```

Protects against:
- Data loss from deleted/moved source files
- Corrupted search index files during indexing
- Inconsistent metadata from interrupted operations

## Performance

| Metric | Value |
|--------|-------|
| Search latency | <100ms (cross-layer), <50ms (distill), <30ms (verbatim) |
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
Set `claude_directory_wsl` in `~/.searchat/config/settings.toml`:
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
pip install . --force-reinstall
```

**Empty environment variables override config:**
Remove empty values from `~/.searchat/config/.env` or set proper values.

## License

MIT






