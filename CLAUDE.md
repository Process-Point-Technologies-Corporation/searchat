# Searchat - Project Instructions

## CRITICAL: DATA SAFETY
Source JSONLs are **LOST**. Parquet files contain **IRREPLACEABLE** conversation data.

**NEVER:**
- Run `index_all()` — blocked with RuntimeError
- Run `index_incremental()` — would delete orphaned data
- Click "Reindex" button — disabled, returns 403
- Modify `.searchat/data/` without backup

**Backups:**
Backups are stored in `~/.searchat/backups/` by default.

## Run (Windows PowerShell)
```powershell
cd path\to\searchat
pip install -e .                 # first time only
searchat-web                     # Web UI at localhost:8000
searchat                         # CLI alternative
```

## Run (WSL)
```bash
cd /path/to/searchat
source .venv/bin/activate
searchat-web
```

## Architecture
- **Storage:** Parquet (DuckDB queries)
- **Search:** Hybrid (BM25 keyword + FAISS semantic)
- **Embeddings:** all-MiniLM-L6-v2
- **Web:** FastAPI + modular routers + ES6 modules
- **Data:** `~/.searchat/` (Windows: `%USERPROFILE%\.searchat\`)
- **Tests:** 62 API endpoint tests

## Key Files
| Path | Purpose |
|------|---------|
| `src/searchat/core/indexer.py` | Indexing (BLOCKED) |
| `src/searchat/core/search_engine.py` | Search queries |
| `src/searchat/api/app.py` | FastAPI app factory |
| `src/searchat/api/routers/` | 6 API routers (15 endpoints) |
| `src/searchat/cli/main.py` | Terminal interface |
| `src/searchat/backup.py` | Backup/restore manager |
| `src/searchat/web/` | Modular HTML/CSS/JS |
| `tests/api/` | API endpoint tests (62 tests) |
| `docs/api-reference.md` | Complete API documentation |

## Safety Guards (2026-01-10)
- `index_all()` requires `force=True` if existing index detected
- `index_append_only()` is the safe method (never deletes)
- Dangerous methods removed: `index_incremental()`, `_mark_deleted_files()`
- `/api/reindex` returns 403

## Backup & Restore (2026-01-21)
**New built-in backup functionality in left sidebar:**

**Web UI:**
- **Create Backup** button - Creates timestamped backup of index + config
- **Manage Backups** button - Lists all backups with details
- Located in left sidebar (doesn't interfere with existing controls)

**Storage:**
- Default location: `~/.searchat/backups/` (Windows: `%USERPROFILE%\.searchat\backups\`)
- Format: `backup_YYYYMMDD_HHMMSS/`
- Backs up: `data/` (parquet files, FAISS index), `config/`
- Metadata stored in `backup_metadata.json`

**Safety:**
- Automatic pre-restore backup before any restore operation
- Backup validation before restore (checks for parquet files)
- Manual backups kept indefinitely (no auto-deletion)

**API Endpoints:**
- `POST /api/backup/create` - Create new backup
- `GET /api/backup/list` - List all backups
- `POST /api/backup/restore` - Restore from backup
- `DELETE /api/backup/delete/{name}` - Delete backup

**Architecture:**
- Web UI modularized: `src/searchat/web/index.html`
- Backup manager: `src/searchat/backup.py`
- All existing functionality preserved (no changes to middle content)

## Git Workflow

**CRITICAL: Work with Fork, Not Upstream**

This is a fork of the upstream repository. Always push and create PRs against the fork.

**Repository Setup:**
- **Fork (origin):** `https://github.com/Mathews-Tom/searchat.git` ← **PUSH HERE**
- **Upstream:** `https://github.com/Process-Point-Technologies-Corporation/searchat.git` (read-only)

**Workflow:**
```bash
# Push to fork (origin)
git push origin <branch-name>

# Create PR on fork
gh pr create --repo Mathews-Tom/searchat --base master --head <branch-name>

# Never push directly to upstream
# ❌ git push upstream <branch-name>
# ❌ gh pr create --repo Process-Point-Technologies-Corporation/searchat
```

**Remote Configuration:**
```bash
git remote -v
# origin    https://github.com/Mathews-Tom/searchat.git (push/fetch)
# upstream  https://github.com/Process-Point-Technologies-Corporation/searchat.git (fetch only)
```

**Syncing with Upstream:**
```bash
# Fetch upstream changes
git fetch upstream

# Merge upstream/master into local master
git checkout master
git merge upstream/master

# Push updated master to fork
git push origin master
```

## Config
Configuration file: `~/.searchat/config/settings.toml`
