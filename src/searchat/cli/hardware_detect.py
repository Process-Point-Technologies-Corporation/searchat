"""Hardware detection CLI command.

Detects GPU, CPU, RAM and sets optimal batch sizes and worker counts.
"""
import argparse
import sys
from pathlib import Path

from searchat.utils.hardware import detect_hardware, save_hardware_profile
from searchat.config.constants import DEFAULT_DATA_DIR, DEFAULT_CONFIG_SUBDIR


def main():
    """Run hardware detection and save profile."""
    parser = argparse.ArgumentParser(
        description="Detect hardware and configure optimal settings"
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_DATA_DIR / DEFAULT_CONFIG_SUBDIR,
        help="Config directory (default: ~/.searchat/config)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show hardware profile without saving",
    )

    args = parser.parse_args()

    print("Detecting hardware...")
    print()

    profile = detect_hardware()

    print("=" * 60)
    print("HARDWARE PROFILE")
    print("=" * 60)
    print()
    print("GPU:")
    if profile.has_cuda:
        print(f"  Device: {profile.cuda_device_name}")
        print(f"  VRAM: {profile.vram_mb:,} MB ({profile.vram_mb / 1024:.1f} GB)")
        print(f"  Count: {profile.cuda_device_count}")
    else:
        print("  No CUDA GPU detected")
    print()
    print("CPU/RAM:")
    print(f"  Cores: {profile.cpu_cores}")
    print(f"  RAM: {profile.ram_gb} GB")
    print()
    print("=" * 60)
    print("OPTIMAL SETTINGS")
    print("=" * 60)
    print()
    print(f"  Embedding batch size: {profile.embedding_batch_size}")
    print(f"  Embedding device: {profile.embedding_device}")
    print(f"  Indexing workers: {profile.indexing_workers}")
    print()

    if args.show:
        print("(not saved, use without --show to save)")
        return 0

    save_hardware_profile(profile, args.config_dir)
    print()
    print(f"Saved to {args.config_dir / 'hardware.toml'}")
    print()
    print("These settings will be used automatically by searchat.")
    print("You can override them in ~/.searchat/config/settings.toml")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
