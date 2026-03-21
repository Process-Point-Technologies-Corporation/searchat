# Contributing to Searchat

## Development Setup

### Prerequisites

- Python 3.9+
- Git
- A virtual environment tool (`venv`, `uv`, `conda`, or similar)

### Initial Setup

1. Clone the repository.
   ```bash
   git clone https://github.com/Process-Point-Technologies-Corporation/searchat.git
   cd searchat
   ```

2. Create and activate a virtual environment.
   ```bash
   python -m venv .venv

   # Windows
   .venv\Scripts\activate

   # Unix/macOS
   source .venv/bin/activate
   ```

3. Install the package with development dependencies.
   ```bash
   pip install -e ".[dev]"
   ```

4. Run first-time setup.
   ```bash
   python -m searchat.setup
   ```

## Development Workflow

1. Create a feature branch.
   ```bash
   git checkout -b feature/your-change
   ```

2. Make focused changes.
   - Keep user-facing behavior coherent.
   - Add or update tests when behavior changes.
   - Update docs/examples if commands or API behavior change.

3. Run relevant tests.
   ```bash
   # Full suite
   pytest

   # Focused suites
   pytest tests/api/
   pytest tests/core/
   pytest tests/palace/
   pytest tests/unit/

   # One file
   pytest tests/api/test_search_routes.py
   ```

4. Run a manual smoke check when relevant.
   ```bash
   python -m searchat.setup
   searchat-web
   searchat "test query"
   searchat-hardware --show
   ```

5. Commit with a clear message.
   ```bash
   git add .
   git commit -m "Describe the change"
   ```

## Project Shape

Public product areas:
- `src/searchat/api/` - FastAPI app and routers
- `src/searchat/core/` - indexing, search, storage, watcher logic
- `src/searchat/palace/` - distilled retrieval components
- `src/searchat/agents/` - provider detection and integrations
- `src/searchat/web/` - packaged web UI assets
- `tests/` - public tool tests only

Key user-facing search modes:
- `cross-layer`
- `verbatim`
- `distill`

## Code Guidelines

- Follow existing style and keep changes localized.
- Use type hints on public functions and data models.
- Keep constants in configuration/constants modules rather than scattering literals.
- Prefer explicit errors over silent fallback behavior.
- Update examples and docs when changing CLI/API/config behavior.

## Testing Guidance

Current test layout:
- `tests/api/` - API route behavior
- `tests/core/` - unified storage/index/search internals
- `tests/palace/` - distillation and distilled retrieval logic
- `tests/unit/` - smaller isolated units
- `tests/hooks/` - hook-specific behavior

Guidelines:
- Use synthetic fixtures only.
- Do not add tests that depend on non-public data or external local state.
- Keep platform-path fixtures generic.
- Mock external CLIs and network-like interactions.

Useful commands:
```bash
pytest --cov=searchat --cov-report=html
pytest -m "not slow"
```

## Pull Requests

Before opening a PR:
- Tests for changed behavior pass locally.
- Docs/examples/config are updated if needed.
- No local junk, generated artifacts, or machine-specific paths are included.
- Public API/docs stay tool-focused.

