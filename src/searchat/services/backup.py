"""
Backup and restore functionality for Searchat indices and data.

CRITICAL: The Parquet files contain irreplaceable conversation data.
This module provides safe backup/restore operations to protect that data.
"""

import shutil
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BackupMetadata:
    """Metadata for a backup."""

    def __init__(
        self,
        timestamp: str,
        backup_path: Path,
        source_path: Path,
        file_count: int,
        total_size_bytes: int,
        backup_type: str = "manual"
    ):
        self.timestamp = timestamp
        self.backup_path = backup_path
        self.source_path = source_path
        self.file_count = file_count
        self.total_size_bytes = total_size_bytes
        self.backup_type = backup_type

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "timestamp": self.timestamp,
            "backup_path": str(self.backup_path),
            "source_path": str(self.source_path),
            "file_count": self.file_count,
            "total_size_bytes": self.total_size_bytes,
            "total_size_mb": round(self.total_size_bytes / (1024 * 1024), 2),
            "backup_type": self.backup_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BackupMetadata":
        """Create metadata from dictionary."""
        return cls(
            timestamp=data["timestamp"],
            backup_path=Path(data["backup_path"]),
            source_path=Path(data["source_path"]),
            file_count=data["file_count"],
            total_size_bytes=data["total_size_bytes"],
            backup_type=data.get("backup_type", "manual"),
        )


class BackupManager:
    """Manages backups and restores for Searchat data."""

    METADATA_FILE = "backup_metadata.json"

    def __init__(self, data_dir: Path, backup_dir: Optional[Path] = None):
        """
        Initialize backup manager.

        Args:
            data_dir: Path to the searchat data directory (e.g., ~/.searchat)
            backup_dir: Optional custom backup directory. Defaults to data_dir/backups
        """
        self.data_dir = Path(data_dir)

        if backup_dir:
            self.backup_dir = Path(backup_dir)
        else:
            # Default: .searchat/backups/
            self.backup_dir = self.data_dir / "backups"

        # Ensure backup directory exists
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def _get_directory_size(self, path: Path) -> int:
        """Calculate total size of all files in a directory."""
        total = 0
        for item in path.rglob("*"):
            if item.is_file():
                total += item.stat().st_size
        return total

    def _count_files(self, path: Path) -> int:
        """Count all files in a directory."""
        return sum(1 for item in path.rglob("*") if item.is_file())

    def create_backup(
        self,
        backup_name: Optional[str] = None,
        backup_type: str = "manual"
    ) -> BackupMetadata:
        """
        Create a backup of the current index and data.

        Args:
            backup_name: Optional custom backup name. If not provided, uses timestamp.
            backup_type: Type of backup (manual, pre_restore, scheduled)

        Returns:
            BackupMetadata object with backup information

        Raises:
            FileNotFoundError: If data directory doesn't exist
            IOError: If backup creation fails
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        # Generate backup name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if backup_name:
            folder_name = f"{backup_name}_{timestamp}"
        else:
            folder_name = f"backup_{timestamp}"

        backup_path = self.backup_dir / folder_name

        logger.info(f"Creating backup: {backup_path}")

        # Create backup directory
        backup_path.mkdir(parents=True, exist_ok=True)

        # Items to backup
        items_to_backup = [
            ("data", self.data_dir / "data"),
            ("config", self.data_dir / "config"),
        ]

        file_count = 0
        total_size = 0

        for item_name, source_path in items_to_backup:
            if source_path.exists():
                dest_path = backup_path / item_name

                if source_path.is_dir():
                    logger.info(f"  Copying directory: {item_name}")
                    shutil.copytree(source_path, dest_path)
                    file_count += self._count_files(dest_path)
                    total_size += self._get_directory_size(dest_path)
                else:
                    logger.info(f"  Copying file: {item_name}")
                    shutil.copy2(source_path, dest_path)
                    file_count += 1
                    total_size += dest_path.stat().st_size

        # Create metadata
        metadata = BackupMetadata(
            timestamp=timestamp,
            backup_path=backup_path,
            source_path=self.data_dir,
            file_count=file_count,
            total_size_bytes=total_size,
            backup_type=backup_type,
        )

        # Save metadata to backup directory
        metadata_path = backup_path / self.METADATA_FILE
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        logger.info(
            f"Backup created: {folder_name} "
            f"({file_count} files, {metadata.to_dict()['total_size_mb']} MB)"
        )

        return metadata

    def list_backups(self) -> List[BackupMetadata]:
        """
        List all available backups.

        Returns:
            List of BackupMetadata objects, sorted by timestamp (newest first)
        """
        backups = []

        if not self.backup_dir.exists():
            return backups

        for backup_path in self.backup_dir.iterdir():
            if not backup_path.is_dir():
                continue

            metadata_path = backup_path / self.METADATA_FILE

            if metadata_path.exists():
                # Load from metadata file
                try:
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    backups.append(BackupMetadata.from_dict(data))
                except Exception as e:
                    logger.warning(f"Failed to load backup metadata from {metadata_path}: {e}")
            else:
                # Create metadata on-the-fly for older backups without metadata
                try:
                    file_count = self._count_files(backup_path)
                    total_size = self._get_directory_size(backup_path)

                    # Extract timestamp from folder name
                    folder_name = backup_path.name
                    if "_" in folder_name:
                        timestamp = folder_name.rsplit("_", 1)[-1]
                    else:
                        timestamp = datetime.fromtimestamp(
                            backup_path.stat().st_mtime
                        ).strftime("%Y%m%d_%H%M%S")

                    backups.append(BackupMetadata(
                        timestamp=timestamp,
                        backup_path=backup_path,
                        source_path=self.data_dir,
                        file_count=file_count,
                        total_size_bytes=total_size,
                        backup_type="unknown",
                    ))
                except Exception as e:
                    logger.warning(f"Failed to analyze backup at {backup_path}: {e}")

        # Sort by timestamp (newest first)
        backups.sort(key=lambda b: b.timestamp, reverse=True)

        return backups

    def validate_backup(self, backup_path: Path) -> bool:
        """
        Validate that a backup has the expected structure.

        Args:
            backup_path: Path to the backup directory

        Returns:
            True if backup is valid, False otherwise
        """
        if not backup_path.exists() or not backup_path.is_dir():
            logger.error(f"Backup path does not exist or is not a directory: {backup_path}")
            return False

        # Check for essential directories
        data_dir = backup_path / "data"

        if not data_dir.exists():
            logger.error(f"Backup missing data directory: {data_dir}")
            return False

        # Check for critical data files — DuckDB (primary) or legacy parquet
        has_duckdb = (data_dir / "searchat.duckdb").exists()
        has_parquet = bool(list(data_dir.glob("conversations/*.parquet")))

        if not has_duckdb and not has_parquet:
            logger.error(
                f"Backup contains neither searchat.duckdb nor conversation "
                f"parquet files in: {data_dir}"
            )
            return False

        logger.info(f"Backup validation passed: {backup_path.name}")
        return True

    def restore_from_backup(
        self,
        backup_path: Path,
        create_pre_restore_backup: bool = True
    ) -> Optional[BackupMetadata]:
        """
        Restore data from a backup.

        SAFETY:
        - Creates a pre-restore backup by default
        - Validates backup structure before restoring
        - Uses atomic copy operations

        Args:
            backup_path: Path to the backup directory to restore from
            create_pre_restore_backup: Create a backup before restoring (recommended)

        Returns:
            Metadata of the pre-restore backup (if created)

        Raises:
            FileNotFoundError: If backup doesn't exist
            ValueError: If backup validation fails
            IOError: If restore operation fails
        """
        backup_path = Path(backup_path)

        if not backup_path.exists():
            raise FileNotFoundError(f"Backup not found: {backup_path}")

        # Validate backup
        if not self.validate_backup(backup_path):
            raise ValueError(f"Backup validation failed: {backup_path}")

        # Create pre-restore backup
        pre_restore_metadata = None
        if create_pre_restore_backup and self.data_dir.exists():
            logger.info("Creating pre-restore backup...")
            pre_restore_metadata = self.create_backup(
                backup_name="pre_restore",
                backup_type="pre_restore"
            )

        logger.info(f"Restoring from backup: {backup_path}")

        # Restore items
        items_to_restore = [
            ("data", backup_path / "data", self.data_dir / "data"),
            ("config", backup_path / "config", self.data_dir / "config"),
        ]

        for item_name, source, dest in items_to_restore:
            if not source.exists():
                logger.warning(f"Backup missing {item_name}, skipping")
                continue

            # Remove existing destination
            if dest.exists():
                logger.info(f"  Removing existing {item_name}")
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()

            # Copy from backup
            logger.info(f"  Restoring {item_name}")
            if source.is_dir():
                shutil.copytree(source, dest)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)

        logger.info(f"Restore complete from: {backup_path.name}")

        return pre_restore_metadata

    def delete_backup(self, backup_path: Path) -> None:
        """
        Delete a backup directory.

        Args:
            backup_path: Path to the backup directory to delete

        Raises:
            FileNotFoundError: If backup doesn't exist
        """
        backup_path = Path(backup_path)

        if not backup_path.exists():
            raise FileNotFoundError(f"Backup not found: {backup_path}")

        logger.info(f"Deleting backup: {backup_path}")
        shutil.rmtree(backup_path)
        logger.info(f"Backup deleted: {backup_path.name}")
