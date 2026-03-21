"""JSONL parsing helpers with explicit diagnostics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class JSONLParseIssue:
    """A single malformed JSONL line."""

    line_number: int
    message: str


@dataclass(frozen=True)
class JSONLLoadResult:
    """Parsed JSONL content plus parse diagnostics."""

    entries: List[dict]
    issues: List[JSONLParseIssue]

    @property
    def valid_count(self) -> int:
        return len(self.entries)

    @property
    def invalid_count(self) -> int:
        return len(self.issues)

    def describe_issues(self, limit: int = 5) -> str:
        """Summarize malformed line numbers and parser errors."""
        if not self.issues:
            return "no parse issues"

        sample = ", ".join(
            f"line {issue.line_number} ({issue.message})"
            for issue in self.issues[:limit]
        )
        remaining = self.invalid_count - min(self.invalid_count, limit)
        if remaining > 0:
            sample = f"{sample}, +{remaining} more"
        return sample


def _parse_jsonl_lines(raw_text: str, starting_line_number: int = 1) -> JSONLLoadResult:
    """Parse JSONL text and retain malformed-line diagnostics."""
    entries: List[dict] = []
    issues: List[JSONLParseIssue] = []

    for line_number, raw_line in enumerate(
        raw_text.splitlines(),
        start=starting_line_number,
    ):
        if not raw_line.strip():
            continue
        try:
            parsed = json.loads(raw_line, strict=False)
        except json.JSONDecodeError as exc:
            issues.append(JSONLParseIssue(line_number=line_number, message=exc.msg))
            continue
        if not isinstance(parsed, dict):
            issues.append(
                JSONLParseIssue(
                    line_number=line_number,
                    message=f"expected JSON object, got {type(parsed).__name__}",
                )
            )
            continue
        entries.append(parsed)

    return JSONLLoadResult(entries=entries, issues=issues)


def load_jsonl(file_path: Path) -> JSONLLoadResult:
    """Parse a JSONL file and retain malformed-line diagnostics."""
    return _parse_jsonl_lines(file_path.read_text(encoding="utf-8"))


def load_jsonl_text(raw_text: str, starting_line_number: int = 1) -> JSONLLoadResult:
    """Parse JSONL text that may represent an appended transcript tail."""
    return _parse_jsonl_lines(raw_text, starting_line_number=starting_line_number)
