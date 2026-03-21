"""Safe parquet file access — skips corrupted files."""
import logging
import time
from pathlib import Path
from typing import List, Tuple

import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

_cache: dict[str, Tuple[float, float, List[Path]]] = {}
_CACHE_TTL_SECONDS = 30.0


def get_valid_parquet_files(directory: Path) -> List[Path]:
    """Return parquet files in directory, excluding any with corrupted metadata.

    Results are cached for 30 seconds per directory. Cache invalidates
    when the directory mtime changes or TTL expires.
    """
    key = str(directory)
    now = time.monotonic()

    try:
        dir_mtime = directory.stat().st_mtime
    except OSError:
        dir_mtime = 0.0

    if key in _cache:
        cached_time, cached_mtime, cached_result = _cache[key]
        if (now - cached_time) < _CACHE_TTL_SECONDS and cached_mtime == dir_mtime:
            return cached_result

    all_files = sorted(directory.glob("*.parquet"))
    valid = []
    for f in all_files:
        try:
            pq.ParquetFile(f)
            valid.append(f)
        except Exception as e:
            logger.error("Skipping corrupted parquet file %s: %s", f, e)

    _cache[key] = (now, dir_mtime, valid)
    return valid


def invalidate_parquet_cache(directory: Path = None):
    """Invalidate cached parquet file lists.

    Args:
        directory: Specific directory to invalidate. If None, clears all.
    """
    if directory is None:
        _cache.clear()
    else:
        _cache.pop(str(directory), None)


def duckdb_file_list(files: List[Path]) -> str:
    """Format a list of paths as a DuckDB read_parquet() argument.

    Returns a string like ``['C:/a.parquet', 'C:/b.parquet']`` suitable
    for embedding directly in a SQL query.
    """
    escaped = ", ".join(f"'{str(f).replace(chr(92), '/')}'" for f in files)
    return f"[{escaped}]"
