"""Mistral Vibe provider."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from searchat.config import PathResolver
from searchat.models.domain import ConversationRecord, MessageRecord

from .base import AgentProvider


class VibeProvider(AgentProvider):
    """Provider for Vibe JSON transcripts."""

    agent_id = "vibe"
    label = "Vibe"

    def discover_dirs(self, config=None) -> List[Path]:
        return PathResolver.resolve_vibe_dirs()

    def matches_file(self, file_path: Path) -> bool:
        if file_path.suffix != ".json":
            return False
        normalized = str(file_path).replace("\\", "/").lower()
        if "/.vibe/" in normalized:
            return True
        if not file_path.exists():
            return False
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return "metadata" in data and "messages" in data

    def parse_conversation(self, file_path: Path, project_id: Optional[str] = None) -> ConversationRecord:
        st = file_path.stat()
        raw_bytes = file_path.read_bytes()
        mtime_ns = st.st_mtime_ns
        data = json.loads(raw_bytes.decode("utf-8"))
        metadata = data.get("metadata", {})
        session_id = metadata.get("session_id", file_path.stem)
        env = metadata.get("environment", {})
        working_dir = env.get("working_directory", "")
        actual_project_id = Path(working_dir).name if working_dir else "vibe-session"

        start_time_str = metadata.get("start_time")
        end_time_str = metadata.get("end_time")
        created_at = datetime.fromisoformat(start_time_str) if start_time_str else datetime.now()
        updated_at = datetime.fromisoformat(end_time_str) if end_time_str else created_at

        messages: List[MessageRecord] = []
        full_text_parts: List[str] = []
        title = "Untitled Vibe Session"
        for msg in data.get("messages", []):
            role = msg.get("role")
            content = msg.get("content", "")
            if role not in ("user", "assistant") or not content:
                continue
            if role == "user" and title == "Untitled Vibe Session":
                title = content[:100].replace("\n", " ").strip()
            code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", content, re.DOTALL)
            messages.append(
                MessageRecord(
                    sequence=len(messages),
                    role=role,
                    content=content,
                    timestamp=created_at,
                    has_code=bool(code_blocks),
                    code_blocks=code_blocks,
                )
            )
            full_text_parts.append(content)

        return ConversationRecord(
            conversation_id=session_id,
            project_id=f"vibe-{actual_project_id}",
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

        data = json.loads(file_path.read_text(encoding="utf-8"))
        messages = []
        for msg in data.get("messages", []):
            role = msg.get("role")
            content = msg.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append(
                    ConversationMessage(
                        role=role,
                        content=content,
                        timestamp="",
                        ply_index=len(messages),
                    )
                )
        return messages

    def extract_cwd(self, file_path: Path) -> Optional[str]:
        data = json.loads(file_path.read_text(encoding="utf-8"))
        return data.get("metadata", {}).get("environment", {}).get("working_directory")

    def build_resume_command(self, session_id: str) -> str:
        return f"vibe --resume {session_id}"
