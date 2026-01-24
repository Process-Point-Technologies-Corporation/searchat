"""Centralized logging configuration for searchat."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from logging.handlers import RotatingFileHandler


@dataclass
class LogConfig:
    """Logging configuration."""

    level: str = "INFO"
    file_enabled: bool = True
    file_path: str = "~/.searchat/logs/searchat.log"
    file_max_bytes: int = 10485760  # 10MB
    file_backup_count: int = 5
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    use_rich_console: bool = True


def setup_logging(config: LogConfig) -> None:
    """
    Configure logging with file rotation and optional Rich console output.

    Args:
        config: Logging configuration
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.level.upper()))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(config.format)

    # Add file handler if enabled
    if config.file_enabled:
        # Expand user path
        log_path = Path(config.file_path).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Create rotating file handler
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=config.file_max_bytes,
            backupCount=config.file_backup_count,
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Add console handler
    if config.use_rich_console:
        try:
            from rich.logging import RichHandler

            console_handler = RichHandler(rich_tracebacks=True)
            console_handler.setFormatter(logging.Formatter("%(message)s"))
        except ImportError:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)
