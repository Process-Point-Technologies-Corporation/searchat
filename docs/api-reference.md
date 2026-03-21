# API Reference

Base URL: `http://localhost:8000`

## Core Endpoints

### `GET /api/search`

Primary public search endpoint.

Parameters:

- `q`: query string
- `mode`: `cross-layer`, `verbatim`, or `distill`
- `project`: optional project filter
- `date`, `date_from`, `date_to`: optional date filtering
- `sort_by`: `relevance`, `date_newest`, `date_oldest`, or `messages`
- `limit`: 1-100

### `GET /api/projects`

Returns the indexed project list.

### `GET /api/conversations/all`

Lists indexed conversations with optional filtering and sorting.

### `GET /api/conversation/{conversation_id}`

Returns one conversation with its messages.

### `POST /api/resume`

Launches the source tool session for a conversation when the underlying provider supports it.

### `GET /api/statistics`

Returns index-level statistics.

### `POST /api/index_missing`

Append-safe indexing of transcripts not yet in storage.

### `POST /api/shutdown`

Graceful server shutdown with indexing-safety checks.

## Backup Endpoints

- `POST /api/backup/create`
- `GET /api/backup/list`
- `POST /api/backup/restore`
- `DELETE /api/backup/delete/{backup_name}`

## Admin Endpoints

- `GET /api/watcher/status`
- `GET /api/indexing/status`

## Notes

- The API is designed for local use on a trusted machine.
- Response shapes are defined in `src/searchat/api/models/`.

