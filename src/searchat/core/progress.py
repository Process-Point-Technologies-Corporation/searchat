"""Progress reporting system with multiple adapters."""

from __future__ import annotations

import logging
import sys
from typing import Protocol


class ProgressCallback(Protocol):
    """Protocol for progress reporting callbacks."""

    def update_phase(self, phase: str) -> None:
        """Update current phase/stage."""
        ...

    def update_file_progress(self, current: int, total: int, filename: str) -> None:
        """Update file processing progress."""
        ...

    def update_embedding_progress(self, current: int, total: int) -> None:
        """Update embedding generation progress."""
        ...

    def update_stats(self, conversations: int, chunks: int, embeddings: int) -> None:
        """Update processing statistics."""
        ...

    def finish(self) -> None:
        """Mark progress as complete."""
        ...


class NullProgressAdapter:
    """No-op progress adapter for tests."""

    def update_phase(self, phase: str) -> None:
        pass

    def update_file_progress(self, current: int, total: int, filename: str) -> None:
        pass

    def update_embedding_progress(self, current: int, total: int) -> None:
        pass

    def update_stats(self, conversations: int, chunks: int, embeddings: int) -> None:
        pass

    def finish(self) -> None:
        pass


class LoggingProgressAdapter:
    """Progress adapter that logs at 10% intervals."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self._last_file_pct = -1
        self._last_embed_pct = -1

    def update_phase(self, phase: str) -> None:
        """Log phase changes."""
        self.logger.info(f"Phase: {phase}")

    def update_file_progress(self, current: int, total: int, filename: str) -> None:
        """Log file progress at 10% intervals."""
        if total == 0:
            return

        pct = int((current / total) * 100)

        # Log at 10% intervals
        if pct // 10 > self._last_file_pct // 10:
            self.logger.info(f"Processing files: {current}/{total} ({pct}%) - {filename}")
            self._last_file_pct = pct

    def update_embedding_progress(self, current: int, total: int) -> None:
        """Log embedding progress at 10% intervals."""
        if total == 0:
            return

        pct = int((current / total) * 100)

        # Log at 10% intervals
        if pct // 10 > self._last_embed_pct // 10:
            self.logger.info(f"Generating embeddings: {current}/{total} ({pct}%)")
            self._last_embed_pct = pct

    def update_stats(self, conversations: int, chunks: int, embeddings: int) -> None:
        """Log current statistics."""
        self.logger.info(
            f"Stats: {conversations} conversations, {chunks} chunks, {embeddings} embeddings"
        )

    def finish(self) -> None:
        """Log completion."""
        self.logger.info("Progress complete")


class RichProgressAdapter:
    """Progress adapter using Rich library for interactive display."""

    def __init__(self) -> None:
        from rich.progress import (
            Progress,
            SpinnerColumn,
            TextColumn,
            BarColumn,
            TaskProgressColumn,
            TimeRemainingColumn,
        )
        from rich.console import Console

        self.console = Console()
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        )

        # Create tasks
        self.file_task = None
        self.embed_task = None
        self.phase_text = "Initializing..."

        # Stats
        self.stats = {
            "conversations": 0,
            "chunks": 0,
            "embeddings": 0,
        }

        # Live display
        self.live = None

    def _start_live(self) -> None:
        """Start live display if not already started."""
        if self.live is None:
            from rich.live import Live

            self.live = Live(self._make_layout(), console=self.console)
            self.live.start()

    def _make_layout(self):
        """Create layout with progress and stats."""
        from rich.table import Table
        from rich.console import Group

        # Stats table
        stats_table = Table.grid(padding=(0, 2))
        stats_table.add_column(style="cyan", justify="right")
        stats_table.add_column(style="magenta")

        stats_table.add_row("Phase:", self.phase_text)
        stats_table.add_row("Conversations:", str(self.stats["conversations"]))
        stats_table.add_row("Chunks:", str(self.stats["chunks"]))
        stats_table.add_row("Embeddings:", str(self.stats["embeddings"]))

        # Combine with progress
        return Group(
            self.progress,
            stats_table,
        )

    def update_phase(self, phase: str) -> None:
        """Update current phase."""
        self.phase_text = phase
        self._start_live()
        if self.live:
            self.live.update(self._make_layout())

    def update_file_progress(self, current: int, total: int, filename: str) -> None:
        """Update file processing progress."""
        self._start_live()

        if self.file_task is None:
            self.file_task = self.progress.add_task(
                "[cyan]Processing files...",
                total=total,
            )

        self.progress.update(
            self.file_task,
            completed=current,
            description=f"[cyan]Processing: {filename}",
        )

        if self.live:
            self.live.update(self._make_layout())

    def update_embedding_progress(self, current: int, total: int) -> None:
        """Update embedding generation progress."""
        self._start_live()

        if self.embed_task is None:
            self.embed_task = self.progress.add_task(
                "[green]Generating embeddings...",
                total=total,
            )

        self.progress.update(self.embed_task, completed=current)

        if self.live:
            self.live.update(self._make_layout())

    def update_stats(self, conversations: int, chunks: int, embeddings: int) -> None:
        """Update statistics."""
        self.stats["conversations"] = conversations
        self.stats["chunks"] = chunks
        self.stats["embeddings"] = embeddings

        if self.live:
            self.live.update(self._make_layout())

    def finish(self) -> None:
        """Stop live display."""
        if self.live:
            self.live.stop()
            self.live = None


def create_progress(use_rich: bool | None = None) -> ProgressCallback:
    """
    Create appropriate progress adapter based on context.

    Args:
        use_rich: Force Rich adapter (True) or logging adapter (False).
                 If None, auto-detect based on TTY.

    Returns:
        Progress adapter instance
    """
    if use_rich is None:
        # Auto-detect TTY
        use_rich = sys.stdout.isatty()

    if use_rich:
        try:
            return RichProgressAdapter()
        except ImportError:
            # Fall back to logging if Rich not available
            return LoggingProgressAdapter()
    else:
        return LoggingProgressAdapter()
