"""Backup endpoints - create, list, restore, and delete backups."""
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException

from searchat.api.dependencies import (
    get_backup_manager,
    get_unified_indexer,
    reset_projects_cache,
)


router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/create")
async def create_backup(backup_name: Optional[str] = None):
    """Create a new backup of the index and data."""
    try:
        backup_manager = get_backup_manager()
        logger.info(f"Creating backup: {backup_name or 'auto'}")
        metadata = backup_manager.create_backup(backup_name=backup_name)

        return {
            "success": True,
            "backup": metadata.to_dict(),
            "message": f"Backup created: {metadata.backup_path.name}"
        }

    except Exception as e:
        logger.error(f"Failed to create backup: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
async def list_backups():
    """List all available backups."""
    try:
        backup_manager = get_backup_manager()
        backups = backup_manager.list_backups()

        return {
            "backups": [b.to_dict() for b in backups],
            "total": len(backups),
            "backup_directory": str(backup_manager.backup_dir)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/restore")
async def restore_backup(backup_name: str):
    """Restore from a backup."""
    try:
        backup_manager = get_backup_manager()

        backup_path = backup_manager.backup_dir / backup_name

        if not backup_path.exists():
            raise HTTPException(status_code=404, detail=f"Backup not found: {backup_name}")

        logger.info(f"Restoring from backup: {backup_name}")

        pre_restore_metadata = backup_manager.restore_from_backup(
            backup_path=backup_path,
            create_pre_restore_backup=True
        )

        # Rebuild DuckDB from restored parquet files
        unified_indexer = get_unified_indexer()
        if unified_indexer is not None:
            unified_indexer.index_from_parquet()

        # Clear projects cache
        reset_projects_cache()

        result = {
            "success": True,
            "restored_from": backup_name,
            "message": f"Successfully restored from backup: {backup_name}"
        }

        if pre_restore_metadata:
            result["pre_restore_backup"] = pre_restore_metadata.to_dict()

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to restore backup: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/delete/{backup_name}")
async def delete_backup(backup_name: str):
    """Delete a backup."""
    try:
        backup_manager = get_backup_manager()
        backup_path = backup_manager.backup_dir / backup_name

        if not backup_path.exists():
            raise HTTPException(status_code=404, detail=f"Backup not found: {backup_name}")

        logger.info(f"Deleting backup: {backup_name}")
        backup_manager.delete_backup(backup_path)

        return {
            "success": True,
            "deleted": backup_name,
            "message": f"Backup deleted: {backup_name}"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete backup: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
