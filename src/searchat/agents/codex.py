"""Codex provider."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from searchat.config import PathResolver
from searchat.models.domain import ConversationRecord, MessageRecord
from searchat.utils.jsonl import JSONLLoadResult, load_jsonl

from .base import AgentProvider


class CodexProvider(AgentProvider):
    """Provider for Codex session JSONL transcripts."""

    agent_id = "codex"
    label = "Codex"

    def discover_dirs(self, config=None) -> List[Path]:
        return PathResolver.resolve_codex_dirs()

    def matches_file(self, file_path: Path) -> bool:
        if file_path.suffix != ".jsonl":
            return False
        normalized = str(file_path).replace("\\", "/").lower()
        if "/.codex/" in normalized:
            return True
        if not file_path.exists():
            return False
        for entry in self._load_jsonl(file_path).entries:
            return entry.get("type") == "session_meta"
        return False

    def parse_conversation(self, file_path: Path, project_id: Optional[str] = None) -> ConversationRecord:
        st = file_path.stat()
        raw_bytes = file_path.read_bytes()
        mtime_ns = st.st_mtime_ns
        load_result = self._load_jsonl(file_path)
        lines = load_result.entries
        self._raise_if_unparseable(file_path, load_result)

        session_meta = next(
            (entry.get("payload", {}) for entry in lines if entry.get("type") == "session_meta"),
            {},
        )
        conversation_id = session_meta.get("id", file_path.stem)
        cwd = session_meta.get("cwd", "")
        actual_project_id = Path(cwd).name if cwd else "codex-session"

        created_at = self._parse_timestamp(session_meta.get("timestamp")) or datetime.now()
        updated_at = created_at
        messages: List[MessageRecord] = []
        full_text_parts: List[str] = []
        title = "Untitled Codex Session"

        for entry in lines:
            if entry.get("type") != "response_item":
                continue
            payload = entry.get("payload", {})
            if payload.get("type") != "message":
                continue
            role = payload.get("role")
            if role not in ("user", "assistant"):
                continue
            content = self._extract_text(payload.get("content", []))
            if not content:
                continue
            if role == "user" and title == "Untitled Codex Session":
                title = content[:100].replace("\n", " ").strip()
            timestamp = self._parse_timestamp(entry.get("timestamp")) or datetime.now()
            updated_at = timestamp
            code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", content, re.DOTALL)
            messages.append(
                MessageRecord(
                    sequence=len(messages),
                    role=role,
                    content=content,
                    timestamp=timestamp,
                    has_code=bool(code_blocks),
                    code_blocks=code_blocks,
                )
            )
            full_text_parts.append(content)

        if messages:
            created_at = messages[0].timestamp
            updated_at = messages[-1].timestamp

        return ConversationRecord(
            conversation_id=conversation_id,
            project_id=f"codex-{actual_project_id}",
            file_path=str(file_path),
            title=title,
            created_at=created_at,
            updated_at=updated_at,
            message_count=len(messages),
            messages=messages,
            full_text="\n\n".join(full_text_parts),
            embedding_id=-1,
            file_hash="",
            indexed_at=datetime.now(),
            file_size=len(raw_bytes),
            mtime_ns=mtime_ns,
        )

    def load_messages(self, file_path: Path) -> List["ConversationMessage"]:
        from searchat.api.models import ConversationMessage

        messages = []
        load_result = self._load_jsonl(file_path)
        self._raise_if_unparseable(file_path, load_result)
        for entry in load_result.entries:
            if entry.get("type") != "response_item":
                continue
            payload = entry.get("payload", {})
            if payload.get("type") != "message":
                continue
            role = payload.get("role")
            if role not in ("user", "assistant"):
                continue
            content = self._extract_text(payload.get("content", []))
            if content:
                messages.append(
                    ConversationMessage(
                        role=role,
                        content=content,
                        timestamp=entry.get("timestamp", ""),
                        ply_index=len(messages),
                    )
                )
        return messages

    def extract_cwd(self, file_path: Path) -> Optional[str]:
        for entry in self._load_jsonl(file_path).entries:
            if entry.get("type") == "session_meta":
                return entry.get("payload", {}).get("cwd")
        return None

    def build_resume_command(self, session_id: str) -> str:
        return f"codex resume {session_id}"

    @staticmethod
    def _load_jsonl(file_path: Path) -> JSONLLoadResult:
        return load_jsonl(file_path)

    @staticmethod
    def _raise_if_unparseable(file_path: Path, load_result: JSONLLoadResult) -> None:
        if load_result.valid_count == 0:
            raise ValueError(
                f"No valid JSON objects in {file_path}; malformed lines: "
                f"{load_result.describe_issues()}"
            )
        if load_result.invalid_count > 0:
            raise ValueError(
                f"Malformed JSONL in {file_path}; parsed {load_result.valid_count} entries "
                f"but found {load_result.invalid_count} invalid lines: "
                f"{load_result.describe_issues()}"
            )

    @staticmethod
    def _extract_text(content_blocks) -> str:
        parts = []
        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            if block.get("type") in ("input_text", "output_text"):
                text = block.get("text", "")
                if text:
                    parts.append(text)
        return "\n\n".join(parts).strip()

    @staticmethod
    def _parse_timestamp(timestamp_str: Optional[str]) -> Optional[datetime]:
        if not timestamp_str:
            return None
        return datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
