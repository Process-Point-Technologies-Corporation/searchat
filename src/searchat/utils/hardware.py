"""Hardware detection and optimization for searchat.

Auto-detects GPU VRAM, CPU cores, and sets optimal batch sizes.
"""
import logging
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import psutil
import torch

logger = logging.getLogger(__name__)


@dataclass
class HardwareProfile:
    """Hardware capabilities and optimal settings."""

    # Detection results
    has_cuda: bool
    cuda_device_count: int
    cuda_device_name: Optional[str]
    vram_mb: int
    cpu_cores: int
    ram_gb: int

    # Optimal settings
    embedding_batch_size: int
    embedding_device: str
    indexing_workers: int

    def to_dict(self) -> dict:
        """Convert to dict for config storage."""
        return {
            "has_cuda": self.has_cuda,
            "cuda_device_count": self.cuda_device_count,
            "cuda_device_name": self.cuda_device_name,
            "vram_mb": self.vram_mb,
            "cpu_cores": self.cpu_cores,
            "ram_gb": self.ram_gb,
            "embedding_batch_size": self.embedding_batch_size,
            "embedding_device": self.embedding_device,
            "indexing_workers": self.indexing_workers,
        }


def detect_gpu_vram() -> tuple[bool, int, Optional[str]]:
    """Detect CUDA availability and VRAM.

    Returns:
        (has_cuda, vram_mb, device_name)
    """
    if not torch.cuda.is_available():
        return False, 0, None

    try:
        device_count = torch.cuda.device_count()
        if device_count == 0:
            return False, 0, None

        # Get first GPU properties
        props = torch.cuda.get_device_properties(0)
        vram_mb = props.total_memory // (1024 * 1024)
        device_name = props.name

        logger.info(
            "Detected CUDA GPU: %s with %d MB VRAM",
            device_name, vram_mb
        )
        return True, vram_mb, device_name

    except Exception as e:
        logger.warning("Failed to detect GPU: %s", e)
        return False, 0, None


def calculate_optimal_batch_size(vram_mb: int) -> int:
    """Calculate optimal embedding batch size based on VRAM.

    Conservative estimates for all-MiniLM-L6-v2 (384 dim embeddings):
    - Model: ~90 MB
    - Per sample: ~2 KB (text) + ~1.5 KB (embedding) = ~4 KB
    - Overhead: 20% safety margin

    Args:
        vram_mb: Available VRAM in MB

    Returns:
        Optimal batch size
    """
    if vram_mb == 0:
        # CPU fallback
        return 32

    # VRAM tiers with conservative batch sizes
    if vram_mb >= 12000:  # 12+ GB (e.g., 4080)
        return 512
    elif vram_mb >= 8000:  # 8-12 GB (e.g., 3070)
        return 256
    elif vram_mb >= 6000:  # 6-8 GB (e.g., 3060)
        return 128
    elif vram_mb >= 4000:  # 4-6 GB (e.g., 1660 Ti)
        return 64
    else:  # < 4 GB
        return 32


def calculate_optimal_workers(cpu_cores: int) -> int:
    """Calculate optimal worker count for indexing.

    Args:
        cpu_cores: Number of CPU cores

    Returns:
        Optimal worker count
    """
    # Use half of cores, min 2, max 8
    workers = max(2, min(8, cpu_cores // 2))
    return workers


def detect_hardware() -> HardwareProfile:
    """Detect hardware and calculate optimal settings.

    Returns:
        HardwareProfile with detection results and optimal settings
    """
    logger.info("Detecting hardware configuration...")

    # GPU detection
    has_cuda, vram_mb, device_name = detect_gpu_vram()
    device_count = torch.cuda.device_count() if has_cuda else 0

    # CPU/RAM detection
    cpu_cores = psutil.cpu_count(logical=False) or 4
    ram_bytes = psutil.virtual_memory().total
    ram_gb = ram_bytes // (1024 ** 3)

    # Calculate optimal settings
    batch_size = calculate_optimal_batch_size(vram_mb)
    device = "cuda" if has_cuda else "cpu"
    workers = calculate_optimal_workers(cpu_cores)

    profile = HardwareProfile(
        has_cuda=has_cuda,
        cuda_device_count=device_count,
        cuda_device_name=device_name,
        vram_mb=vram_mb,
        cpu_cores=cpu_cores,
        ram_gb=ram_gb,
        embedding_batch_size=batch_size,
        embedding_device=device,
        indexing_workers=workers,
    )

    logger.info(
        "Hardware profile: %s, %d MB VRAM, %d cores, %d GB RAM",
        device_name or "CPU",
        vram_mb,
        cpu_cores,
        ram_gb,
    )
    logger.info(
        "Optimal settings: batch_size=%d, device=%s, workers=%d",
        batch_size,
        device,
        workers,
    )

    return profile


def save_hardware_profile(profile: HardwareProfile, config_dir: Path) -> None:
    """Save hardware profile to config file.

    Args:
        profile: Hardware profile to save
        config_dir: Config directory path
    """
    config_dir.mkdir(parents=True, exist_ok=True)
    hardware_file = config_dir / "hardware.toml"

    data = profile.to_dict()

    # Write TOML manually (simple structure, no nested tables)
    lines = [
        "# Hardware profile - auto-detected settings",
        "# Run 'searchat-hardware' to regenerate",
        "",
        f"has_cuda = {str(data['has_cuda']).lower()}",
        f"cuda_device_count = {data['cuda_device_count']}",
        f"cuda_device_name = {repr(data['cuda_device_name']) if data['cuda_device_name'] else 'null'}",
        f"vram_mb = {data['vram_mb']}",
        f"cpu_cores = {data['cpu_cores']}",
        f"ram_gb = {data['ram_gb']}",
        "",
        "# Optimal settings based on hardware",
        f"embedding_batch_size = {data['embedding_batch_size']}",
        f"embedding_device = {repr(data['embedding_device'])}",
        f"indexing_workers = {data['indexing_workers']}",
        "",
    ]

    with open(hardware_file, "w") as f:
        f.write("\n".join(lines))

    logger.info("Saved hardware profile to %s", hardware_file)


def load_hardware_profile(config_dir: Path) -> Optional[HardwareProfile]:
    """Load hardware profile from config file.

    Args:
        config_dir: Config directory path

    Returns:
        HardwareProfile if exists, None otherwise
    """
    import tomli

    hardware_file = config_dir / "hardware.toml"
    if not hardware_file.exists():
        return None

    try:
        with open(hardware_file, "rb") as f:
            data = tomli.load(f)

        profile = HardwareProfile(
            has_cuda=data.get("has_cuda", False),
            cuda_device_count=data.get("cuda_device_count", 0),
            cuda_device_name=data.get("cuda_device_name"),
            vram_mb=data.get("vram_mb", 0),
            cpu_cores=data.get("cpu_cores", 4),
            ram_gb=data.get("ram_gb", 8),
            embedding_batch_size=data.get("embedding_batch_size", 32),
            embedding_device=data.get("embedding_device", "cpu"),
            indexing_workers=data.get("indexing_workers", 4),
        )

        logger.info("Loaded hardware profile from %s", hardware_file)
        return profile

    except Exception as e:
        logger.warning("Failed to load hardware profile: %s", e)
        return None


def get_or_detect_hardware(config_dir: Path, force_detect: bool = False) -> HardwareProfile:
    """Get cached hardware profile or detect if not exists.

    Args:
        config_dir: Config directory path
        force_detect: Force re-detection even if cached

    Returns:
        HardwareProfile
    """
    if not force_detect:
        profile = load_hardware_profile(config_dir)
        if profile is not None:
            return profile

    # Detect and save
    profile = detect_hardware()
    save_hardware_profile(profile, config_dir)
    return profile
