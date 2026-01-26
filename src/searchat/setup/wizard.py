"""
Interactive setup wizard for Claude Search.

Run with: python -m searchat.setup
"""

import os
import sys
import shutil
from pathlib import Path
from typing import Optional, List

from searchat.config.constants import (
    DEFAULT_DATA_DIR,
    DEFAULT_CONFIG_SUBDIR,
    DEFAULT_DATA_SUBDIR,
    DEFAULT_LOGS_SUBDIR,
    SETTINGS_FILE,
    DEFAULT_SETTINGS_FILE,
    ENV_FILE,
    CLAUDE_DIR_NAME,
    CLAUDE_PROJECTS_SUBDIR,
)
from searchat.config import PathResolver


class SetupManager:
    """Manages interactive setup for first-time users."""

    def __init__(self):
        self.platform = PathResolver.detect_platform()
        self.claude_dirs: List[Path] = []
        self.data_dir = DEFAULT_DATA_DIR
        self.config_dir = self.data_dir / DEFAULT_CONFIG_SUBDIR

    def run_interactive_setup(self) -> bool:
        """
        Run the interactive setup wizard.

        Returns:
            True if setup completed successfully
        """
        print("=" * 70)
        print("Claude Search - Interactive Setup")
        print("=" * 70)
        print()

        # Step 1: Detect platform
        self._show_platform_info()

        # Step 2: Find Claude directories
        if not self._detect_claude_directories():
            return False

        # Step 3: Create directory structure
        if not self._create_directory_structure():
            return False

        # Step 4: Create configuration files
        if not self._create_config_files():
            return False

        # Step 5: Show next steps
        self._show_completion_message()

        return True

    def _show_platform_info(self):
        """Display detected platform information."""
        print(f"Detected platform: {self.platform}")
        print()

        platform_notes = {
            "windows": "Running on Windows. Will search for both Windows and WSL Claude directories.",
            "wsl": "Running under WSL. Will search for Linux paths and Windows mount points.",
            "linux": "Running on Linux. Will search for standard Linux paths.",
            "macos": "Running on macOS. Will search for standard macOS paths.",
        }

        note = platform_notes.get(self.platform, "Unknown platform")
        print(f"Note: {note}")
        print()

    def _detect_claude_directories(self) -> bool:
        """
        Attempt to auto-detect Claude conversation directories.

        Returns:
            True if at least one directory found
        """
        print("Searching for Claude conversation directories...")
        print()

        # Try common locations
        candidates = [
            Path.home() / CLAUDE_DIR_NAME / CLAUDE_PROJECTS_SUBDIR,
            Path.home() / CLAUDE_DIR_NAME,
        ]

        # Platform-specific additional candidates
        if self.platform == "windows":
            # Check for WSL paths
            wsl_paths = [
                Path("\\\\wsl$\\Ubuntu\\home") / os.getenv("USER", "user") / CLAUDE_DIR_NAME,
                Path("\\\\wsl.localhost\\Ubuntu\\home") / os.getenv("USER", "user") / CLAUDE_DIR_NAME,
            ]
            candidates.extend(wsl_paths)

        elif self.platform in ("wsl", "linux"):
            # Check Windows mount points
            username = os.getenv("USER", "user")
            mount_paths = [
                Path(f"/mnt/c/Users/{username}") / CLAUDE_DIR_NAME,
            ]
            candidates.extend(mount_paths)

        # Find existing directories
        found_dirs = []
        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                found_dirs.append(candidate)
                print(f"  ✓ Found: {candidate}")

        if not found_dirs:
            print("  ✗ No Claude directories found automatically")
            print()
            print("Please ensure Claude CLI is installed and has created conversations.")
            print()
            response = input("Would you like to specify a custom path? (y/n): ").strip().lower()

            if response == "y":
                custom_path = input("Enter path to Claude directory: ").strip()
                custom_path_obj = Path(custom_path)
                if custom_path_obj.exists():
                    found_dirs.append(custom_path_obj)
                    print(f"  ✓ Using custom path: {custom_path_obj}")
                else:
                    print(f"  ✗ Path not found: {custom_path_obj}")
                    return False
            else:
                print()
                print("Setup cannot continue without a Claude directory.")
                print("Please install Claude CLI and create at least one conversation.")
                return False

        self.claude_dirs = found_dirs
        print()
        return True

    def _create_directory_structure(self) -> bool:
        """
        Create necessary directory structure.

        Returns:
            True if successful
        """
        print("Creating directory structure...")
        print()

        directories = [
            self.data_dir,
            self.config_dir,
            self.data_dir / DEFAULT_DATA_SUBDIR,
            self.data_dir / DEFAULT_LOGS_SUBDIR,
        ]

        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                print(f"  ✓ Created: {directory}")
            except PermissionError:
                print(f"  ✗ Permission denied: {directory}")
                return False
            except Exception as e:
                print(f"  ✗ Error creating {directory}: {e}")
                return False

        print()
        return True

    def _create_config_files(self) -> bool:
        """
        Create initial configuration files.

        Returns:
            True if successful
        """
        print("Creating configuration files...")
        print()

        # 1. Copy default settings to user config
        user_config = self.config_dir / SETTINGS_FILE
        default_config = Path(__file__).parent.parent / "config" / DEFAULT_SETTINGS_FILE

        if user_config.exists():
            response = input(f"Config file already exists at {user_config}. Overwrite? (y/n): ").strip().lower()
            if response != "y":
                print("  - Skipped settings.toml (already exists)")
                return True

        try:
            if default_config.exists():
                shutil.copy(default_config, user_config)
                print(f"  ✓ Created: {user_config}")

                # Update with detected paths
                self._update_config_paths(user_config)
            else:
                print(f"  ✗ Default config not found: {default_config}")
                return False
        except Exception as e:
            print(f"  ✗ Error creating config: {e}")
            return False

        # 2. Create .env file if it doesn't exist
        env_file = self.config_dir / ENV_FILE
        env_example = Path(__file__).parent.parent / ".env.example"

        if not env_file.exists():
            try:
                if env_example.exists():
                    shutil.copy(env_example, env_file)
                    print(f"  ✓ Created: {env_file}")
                else:
                    # Create minimal .env
                    env_file.write_text(
                        "# Searchat Environment Variables\n"
                        "# Uncomment and modify as needed\n\n"
                        f"# SEARCHAT_DATA_DIR={self.data_dir}\n"
                        "# SEARCHAT_PORT=8000\n"
                    )
                    print(f"  ✓ Created: {env_file}")
            except Exception as e:
                print(f"  ✗ Error creating .env: {e}")
                # Non-fatal, continue

        print()
        return True

    def _update_config_paths(self, config_path: Path):
        """Update config file with detected paths."""
        try:
            content = config_path.read_text()

            # Replace {username} placeholder
            username = os.getenv("USERNAME") or os.getenv("USER") or "user"
            content = content.replace("{username}", username)

            # Update search directory to match data_dir
            content = content.replace(
                'search_directory = "C:/Users/{username}/.searchat"',
                f'search_directory = "{str(self.data_dir).replace(chr(92), "/")}"'
            )

            config_path.write_text(content)
            print(f"  ✓ Updated config with your username: {username}")
        except Exception as e:
            print(f"  ⚠ Warning: Could not update config paths: {e}")

    def _show_completion_message(self):
        """Display setup completion message and next steps."""
        print("=" * 70)
        print("Setup Complete!")
        print("=" * 70)
        print()
        print("Configuration files created:")
        print(f"  - Config: {self.config_dir / SETTINGS_FILE}")
        print(f"  - Env:    {self.config_dir / ENV_FILE}")
        print()
        print("Data directory:")
        print(f"  - {self.data_dir}")
        print()
        print("Next steps:")
        print()
        print("1. Start the web interface:")
        print("   python csw.py")
        print()
        print("2. Or use the CLI:")
        print("   python cs.py \"your search query\"")
        print()
        print("3. Customize configuration:")
        print(f"   Edit {self.config_dir / SETTINGS_FILE}")
        print(f"   Edit {self.config_dir / ENV_FILE}")
        print()
        print("For more information, see README.md")
        print()


def main():
    """Entry point for setup wizard."""
    setup = SetupManager()

    try:
        success = setup.run_interactive_setup()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print()
        print()
        print("Setup cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print()
        print(f"Setup failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
