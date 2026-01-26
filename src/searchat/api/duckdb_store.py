"""DuckDB-on-demand queries over the parquet index.

This module is used by API endpoints that should be fast even when the search
engine/model are still warming.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class IndexStatistics:
    total_conversations: int
    total_messages: int
    avg_messages: float
    total_projects: int
    earliest_date: str | None
    latest_date: str | None


class DuckDBStore:
    def __init__(self, search_dir: Path, *, memory_limit_mb: int | None = None) -> None:
        self._search_dir = search_dir
        self._conversations_dir = search_dir / "data" / "conversations"
        self._memory_limit_mb = memory_limit_mb

    def _conversation_parquets(self) -> list[Path]:
        if not self._conversations_dir.exists():
            return []
        return sorted(self._conversations_dir.glob("*.parquet"))

    def _connect(self):
        import duckdb

        con = duckdb.connect(database=":memory:")
        if self._memory_limit_mb is not None:
            con.execute(f"PRAGMA memory_limit='{int(self._memory_limit_mb)}MB'")
        return con

    def list_projects(self) -> list[str]:
        parquets = self._conversation_parquets()
        if not parquets:
            return []

        con = self._connect()
        try:
            # Project only the required column.
            query = """
            SELECT DISTINCT project_id
            FROM parquet_scan($1)
            WHERE message_count > 0
            ORDER BY project_id
            """
            rows = con.execute(query, [str(self._conversations_dir / "*.parquet")]).fetchall()
            return [r[0] for r in rows]
        finally:
            con.close()

    def get_statistics(self) -> IndexStatistics:
        parquets = self._conversation_parquets()
        if not parquets:
            return IndexStatistics(
                total_conversations=0,
                total_messages=0,
                avg_messages=0.0,
                total_projects=0,
                earliest_date=None,
                latest_date=None,
            )

        con = self._connect()
        try:
            query = """
            SELECT
              COUNT(*)::BIGINT AS total_conversations,
              COALESCE(SUM(message_count), 0)::BIGINT AS total_messages,
              COALESCE(AVG(message_count), 0)::DOUBLE AS avg_messages,
              COUNT(DISTINCT project_id)::BIGINT AS total_projects,
              MIN(created_at) AS earliest_date,
              MAX(updated_at) AS latest_date
            FROM parquet_scan($1)
            """
            row = con.execute(query, [str(self._conversations_dir / "*.parquet")]).fetchone()
            if row is None:
                return IndexStatistics(
                    total_conversations=0,
                    total_messages=0,
                    avg_messages=0.0,
                    total_projects=0,
                    earliest_date=None,
                    latest_date=None,
                )

            earliest = row[4]
            latest = row[5]

            def _to_iso(value):
                if value is None:
                    return None
                if isinstance(value, datetime):
                    return value.isoformat()
                return str(value)

            return IndexStatistics(
                total_conversations=int(row[0]),
                total_messages=int(row[1]),
                avg_messages=float(row[2]),
                total_projects=int(row[3]),
                earliest_date=_to_iso(earliest),
                latest_date=_to_iso(latest),
            )
        finally:
            con.close()
