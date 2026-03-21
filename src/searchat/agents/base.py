"""Base interfaces for agent-specific conversation providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional
from searchat.models.domain import ConversationRecord

if TYPE_CHECKING:
    from searchat.api.models import ConversationMessage


class AgentProvider(ABC):
    """Interface for agent-specific transcript handling."""

    agent_id: str
    label: str

    @abstractmethod
    def discover_dirs(self, config=None) -> List[Path]:
        """Return root directories that may contain this provider's transcripts."""

    @abstractmethod
    def matches_file(self, file_path: Path) -> bool:
        """Return True when the file belongs to this provider."""

    @abstractmethod
    def parse_conversation(self, file_path: Path, project_id: Optional[str] = None) -> ConversationRecord:
        """Parse a source file into a conversation record."""

    @abstractmethod
    def load_messages(self, file_path: Path) -> List["ConversationMessage"]:
        """Load UI-visible messages from a source file."""

    @abstractmethod
    def extract_cwd(self, file_path: Path) -> Optional[str]:
        """Extract the original working directory for session resume."""

    @abstractmethod
    def build_resume_command(self, session_id: str) -> str:
        """Build the CLI command used to resume the session."""
