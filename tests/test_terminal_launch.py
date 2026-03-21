#!/usr/bin/env python3
"""
Standalone test script for cross-platform terminal launching.

Tests terminal launching independently before integrating with session resume.
Run this manually to verify terminal launching works on your platform.

Usage:
    python tests/test_terminal_launch.py
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from searchat.services import PlatformManager
from searchat.config import PathResolver


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def test_platform_detection():
    """Test platform detection."""
    print_section("Platform Detection")

    platform = PathResolver.detect_platform()
    print(f"Detected platform: {platform}")

    pm = PlatformManager()
    print(f"PlatformManager.system: {pm.system}")
    print(f"PlatformManager.is_windows: {pm.is_windows}")
    print(f"PlatformManager.is_linux: {pm.is_linux}")
    print(f"PlatformManager.is_macos: {pm.is_macos}")

    assert platform is not None


def test_path_translation():
    """Test path translation between platforms."""
    print_section("Path Translation")

    test_paths = [
        "D:\\projects\\test",
        "C:\\Users\\Syd\\test",
        "/mnt/d/projects/test",
        "/home/user/test",
        "/Users/name/test",
    ]

    for path in test_paths:
        try:
            translated = PathResolver.translate_claude_path(path)
            print(f"  {path}")
            print(f"  -> {translated}\n")
        except Exception as e:
            print(f"  {path}")
            print(f"  -> ERROR: {e}\n")


def get_test_paths(platform: str) -> dict:
    """Get test paths based on current platform."""
    paths = {
        'native': None,
        'foreign': None,
        'none': None,
    }

    if platform == 'windows':
        paths['native'] = "D:\\projects\\searchat"
        paths['foreign'] = "/mnt/d/projects/searchat"
    elif platform == 'wsl':
        paths['native'] = "/mnt/d/projects/searchat"
        paths['foreign'] = "D:\\projects\\searchat"
    elif platform == 'linux':
        paths['native'] = "/home/user/projects"
        paths['foreign'] = None  # No foreign path for Linux
    elif platform == 'macos':
        paths['native'] = "/Users/Syd/projects"
        paths['foreign'] = None  # No foreign path for macOS

    return paths


def display_command_preview(pm: PlatformManager, command: str, cwd: str = None):
    """Display what command will be executed without actually launching."""
    print(f"\nTest case:")
    print(f"  Command: {command}")
    print(f"  CWD: {cwd or '(none)'}")

    # Show what would be executed
    if pm.is_windows:
        if cwd and cwd.startswith('/'):
            print(f"  Type: Windows + WSL path")
            cmd_args = [
                'cmd.exe', '/c', 'start',
                'cmd.exe', '/k',
                'wsl.exe', '--cd', cwd,
                'bash', '-i', '-c', command
            ]
        elif cwd:
            print(f"  Type: Windows + Windows path")
            combined = f'cd /d {cwd} && {command}'
            cmd_args = ['cmd.exe', '/c', 'start', 'cmd.exe', '/k', combined]
        else:
            print(f"  Type: Windows + no cwd")
            cmd_args = ['cmd.exe', '/c', 'start', 'cmd.exe', '/k', command]

        print(f"  Will execute: {cmd_args}")

    elif pm.is_linux:
        if cwd:
            bash_cmd = f'cd {cwd} && {command}'
        else:
            bash_cmd = command

        cmd_args = ['gnome-terminal', '--', 'bash', '-c', f'{bash_cmd}; exec bash']
        print(f"  Type: Linux")
        print(f"  Will execute: {cmd_args}")

    elif pm.is_macos:
        if cwd:
            script = f'tell application "Terminal" to do script "cd {cwd} && {command}"'
        else:
            script = f'tell application "Terminal" to do script "{command}"'

        cmd_args = ['osascript', '-e', script]
        print(f"  Type: macOS")
        print(f"  Will execute: {cmd_args}")


def manual_terminal_launch(pm: PlatformManager, platform: str, interactive: bool = True):
    """Test terminal launching with various path scenarios."""
    print_section("Terminal Launch Tests")

    paths = get_test_paths(platform)
    test_command = 'echo "Terminal test successful" && pwd'

    test_cases = [
        ("Native path", paths['native'], test_command),
        ("Foreign path", paths['foreign'], test_command),
        ("No path", None, test_command),
    ]

    for test_name, cwd, command in test_cases:
        if cwd is None and test_name == "Foreign path":
            print(f"\n{test_name}: N/A for {platform}")
            continue

        print(f"\n{'-'*60}")
        print(f"Test: {test_name}")
        print(f"{'-'*60}")

        display_command_preview(pm, command, cwd)

        if interactive:
            response = input("\nLaunch terminal? [y/N]: ").strip().lower()
            if response == 'y':
                try:
                    process = pm.open_terminal_with_command(command, cwd)
                    print(f"[OK] Terminal launched successfully (PID: {process.pid})")
                    print("  Check that:")
                    print("    1. Terminal window opened")
                    print("    2. Command executed")
                    print("    3. Working directory is correct (check pwd output)")
                except Exception as e:
                    print(f"[ERROR] Failed to launch terminal: {e}")
            else:
                print("  Skipped")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("  Cross-Platform Terminal Launch Test")
    print("="*60)

    # Test 1: Platform detection
    platform = test_platform_detection()

    # Test 2: Path translation
    test_path_translation()

    # Test 3: Terminal launching
    pm = PlatformManager()

    print_section("Interactive Terminal Tests")
    print("This script will show you what commands will be executed")
    print("and ask if you want to launch each test terminal.")
    print("\nPress Ctrl+C to exit at any time.")

    try:
        manual_terminal_launch(pm, platform, interactive=True)
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user.")
        return

    print_section("Tests Complete")
    print("Verify that:")
    print("  1. Platform detection is correct")
    print("  2. Path translation works as expected")
    print("  3. Terminals open in the correct directory")
    print("  4. Commands execute successfully")


if __name__ == '__main__':
    main()
