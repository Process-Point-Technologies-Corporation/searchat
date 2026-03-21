"""Claude Code provider."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from searchat.config import PathResolver
from searchat.models.domain import ConversationRecord, MessageRecord
from searchat.utils.jsonl import JSONLLoadResult, load_jsonl

from .base import AgentProvider


class ClaudeProvider(AgentProvider):
    """Provider for Claude Code JSONL transcripts."""

    agent_id = "claude"
    label = "Claude Code"

    def discover_dirs(self, config=None) -> List[Path]:
        return PathResolver.resolve_claude_dirs(config)

    def matches_file(self, file_path: Path) -> bool:
        if file_path.suffix != ".jsonl":
            return False
        normalized = str(file_path).replace("\\", "/").lower()
        if "/.codex/" in normalized:
            return False
        if not file_path.exists():
            return True
        for entry in self._load_jsonl(file_path).entries:
            return entry.get("type") != "session_meta"
        return True

    def parse_conversation(self, file_path: Path, project_id: Optional[str] = None) -> ConversationRecord:
        st = file_path.stat()
        raw_bytes = file_path.read_bytes()
        mtime_ns = st.st_mtime_ns
        load_result = self._load_jsonl(file_path)
        lines = load_result.entries
        self._raise_if_unparseable(file_path, load_result)

        conversation_id = file_path.stem
        actual_project_id = project_id or file_path.parent.name
        title = "Untitled"
        for entry in lines:
            text = self._extract_text(entry.get("message", {}).get("content", ""))
            if text:
                title = text[:100]
                break

        messages: List[MessageRecord] = []
        full_text_parts: List[str] = []
        for entry in lines:
            msg_type = entry.get("type")
            if msg_type not in ("user", "assistant"):
                continue

            content = self._extract_text(entry.get("message", {}).get("content", ""))
            code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", content, re.DOTALL)
            timestamp_str = entry.get("timestamp")
            timestamp = datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.now()
            messages.append(
                MessageRecord(
                    sequence=len(messages),
                    role=msg_type,
                    content=content,
                    timestamp=timestamp,
                    has_code=bool(code_blocks),
                    code_blocks=code_blocks,
                )
            )
            full_text_parts.append(content)

        created_at = messages[0].timestamp if messages else datetime.now()
        updated_at = messages[-1].timestamp if messages else datetime.now()

        return ConversationRecord(
            conversation_id=conversation_id,
            project_id=actual_project_id,
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
        for ply_index, entry in enumerate(load_result.entries):
            msg_type = entry.get("type")
            if msg_type not in ("user", "assistant"):
                continue
            content = self._extract_text(entry.get("message", {}).get("content", ""))
            if content:
                messages.append(
                    ConversationMessage(
                        role=msg_type,
                        content=content,
                        timestamp=entry.get("timestamp", ""),
                        ply_index=ply_index,
                    )
                )
        return messages

    def extract_cwd(self, file_path: Path) -> Optional[str]:
        for entry in self._load_jsonl(file_path).entries:
            cwd = entry.get("cwd")
            if cwd:
                return cwd
        return None

    def build_resume_command(self, session_id: str) -> str:
        return f"claude --resume {session_id}"

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
    def _extract_text(raw_content) -> str:
        if isinstance(raw_content, str):
            return raw_content
        if isinstance(raw_content, list):
            return "\n\n".join(
                block.get("text", "")
                for block in raw_content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        return ""
