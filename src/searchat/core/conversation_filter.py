"""Pre-scan filter for automated `claude --print` conversations.

Detects automated conversations by two independent signals:
1. Prefix match — existing excluded_prompt_prefixes logic (distillation, eval, compaction)
2. Structural detection — single-turn conversations (1 user + 1 assistant, no
   file-history-snapshot, no tool_use blocks)

Matched files are moved to an excluded directory rather than deleted, preserving
them for audit. Files that pass both checks are returned for indexing.
"""

import logging
import shutil
from pathlib import Path
from typing import List, Tuple

from searchat.config import Config
from searchat.utils.jsonl import load_jsonl

logger = logging.getLogger(__name__)


def _is_prefix_match(entries: list, prefixes: tuple) -> bool:
    """Check if the first non-empty user message starts with an excluded prefix."""
    for entry in entries:
        if entry.get("type") != "user":
            continue
        raw_content = entry.get("message", {}).get("content", "")
        if isinstance(raw_content, str):
            text = raw_content
        elif isinstance(raw_content, list):
            text = " ".join(
                block.get("text", "")
                for block in raw_content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        else:
            text = ""
        if not text.strip():
            continue
        return any(text.startswith(p) for p in prefixes)
    return False


def _is_single_turn_automated(entries: list) -> bool:
    """Detect structurally automated conversations.

    A `claude --print` conversation has ALL of:
    - Exactly 1 user entry and 1 assistant entry
    - No file-history-snapshot entries
    - No tool_use blocks in assistant message content
    """
    user_count = 0
    assistant_count = 0
    has_file_history = False
    has_tool_use = False

    for entry in entries:
        entry_type = entry.get("type", "")

        if entry_type == "user":
            user_count += 1
        elif entry_type == "assistant":
            assistant_count += 1
            # Check for tool_use blocks in content
            content = entry.get("message", {}).get("content", "")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        has_tool_use = True
                        break
        elif entry_type == "file-history-snapshot":
            has_file_history = True

    return (
        user_count == 1
        and assistant_count == 1
        and not has_file_history
        and not has_tool_use
    )


def _parse_jsonl_entries(file_path: Path) -> list:
    """Parse JSONL entries from a file. Returns list of parsed dicts."""
    result = load_jsonl(file_path)
    if result.invalid_count > 0:
        logger.debug(
            "Malformed JSONL in %s during filtering; parsed %d entries and skipped %d invalid lines: %s",
            file_path,
            result.valid_count,
            result.invalid_count,
            result.describe_issues(),
        )
    return result.entries


def _move_to_excluded(file_path: Path, excluded_dir: Path, reason: str) -> None:
    """Move a file to the excluded directory, preserving project context."""
    # Use parent directory name as project subdirectory
    # e.g. /home/syd/.claude/projects/-home-syd-projects-pramana/abc.jsonl
    #   -> excluded_dir/-home-syd-projects-pramana/abc.jsonl
    project_dir_name = file_path.parent.name
    dest_dir = excluded_dir / project_dir_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / file_path.name

    shutil.move(str(file_path), str(dest_path))
    logger.info("Excluded [%s]: %s -> %s", reason, file_path, dest_path)


def exclude_automated_conversations(
    file_paths: List[str],
    excluded_dir: str,
    config: Config,
) -> List[str]:
    """Filter out automated conversations, moving them to excluded_dir.

    Args:
        file_paths: List of source file paths to check
        excluded_dir: Directory to move excluded files into
        config: Config object (provides excluded_prompt_prefixes)

    Returns:
        List of file paths that passed filtering (should be indexed)
    """
    if not excluded_dir:
        return file_paths

    excluded_path = Path(excluded_dir)
    excluded_path.mkdir(parents=True, exist_ok=True)
    prefixes = config.indexing.excluded_prompt_prefixes

    kept: List[str] = []
    excluded_count = 0

    for fp in file_paths:
        path = Path(fp)

        # Only filter JSONL files (Claude Code conversations)
        if path.suffix != ".jsonl":
            kept.append(fp)
            continue

        if not path.exists():
            kept.append(fp)
            continue

        try:
            entries = _parse_jsonl_entries(path)
        except Exception as e:
            logger.warning("Failed to parse %s for filtering: %s", fp, e)
            kept.append(fp)
            continue

        if not entries:
            kept.append(fp)
            continue

        # Check 1: prefix match
        if _is_prefix_match(entries, prefixes):
            _move_to_excluded(path, excluded_path, "prefix_match")
            excluded_count += 1
            continue

        # Check 2: structural detection (single-turn automated)
        if _is_single_turn_automated(entries):
            _move_to_excluded(path, excluded_path, "single_turn_automated")
            excluded_count += 1
            continue

        kept.append(fp)

    if excluded_count > 0:
        logger.info(
            "Conversation filter: %d excluded, %d kept out of %d total",
            excluded_count, len(kept), len(file_paths),
        )

    return kept
