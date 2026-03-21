# Cross-Platform Terminal Launching

## Overview

Searchat implements robust cross-platform terminal launching for session resumption across Windows, WSL, Linux, and macOS. The implementation correctly distinguishes between WSL and native Linux, and handles path translation automatically.

## Architecture

### Key Components

1. **PathResolver** (`src/searchat/config/path_resolver.py`)
   - Accurate platform detection (distinguishes WSL from Linux)
   - Path translation between Windows and Unix formats
   - Detection method: checks `/proc/version` for "microsoft" on Linux

2. **PlatformManager** (`src/searchat/platform_utils.py`)
   - Platform-specific terminal launching
   - Automatic path translation before launching
   - Clean subprocess argument lists (no shell injection risks)

3. **Resume API** (`src/searchat/api/routers/conversations.py`)
   - `POST /api/resume` endpoint for session resumption
   - Extracts cwd from conversation files
   - Launches terminal with agent command

## Platform Detection

### Four Distinct Platforms

| Platform | Detection Method | sys.platform | platform.system() |
|----------|------------------|--------------|-------------------|
| Windows  | `sys.platform == 'win32'` | `win32` | `Windows` |
| WSL      | `/proc/version` contains "microsoft" | `linux` | `Linux` |
| Linux    | `sys.platform == 'linux'` + not WSL | `linux` | `Linux` |
| macOS    | `sys.platform == 'darwin'` | `darwin` | `Darwin` |

### Why WSL Detection Matters

Standard `platform.system()` returns "Linux" for both WSL and native Linux. WSL requires different terminal launching strategies:
- Can launch Windows GUI applications via WSL interop
- Requires `cmd.exe /c` wrapper for Windows Terminal execution aliases
- Supports bidirectional path translation (/mnt/c ↔ C:\)

## Path Translation

### Automatic Translation Rules

The `PlatformManager._translate_cwd_if_needed()` method automatically translates paths:

| Current Platform | Input Path | Output Path | Notes |
|------------------|------------|-------------|-------|
| Windows | `D:\projects` | `D:\projects` | No translation |
| Windows | `/mnt/d/projects` | `/mnt/d/projects` | Kept as-is, handled by wsl.exe |
| WSL/Linux | `C:\projects` | `/mnt/c/projects` | Translated to Unix mount |
| WSL/Linux | `/mnt/d/projects` | `/mnt/d/projects` | No translation |
| macOS | `/Users/name/projects` | `/Users/name/projects` | No translation |

## Terminal Launching Logic

### Windows Native

```python
# Windows path
cmd.exe /c start cmd.exe /k "cd /d D:\projects && command"

# WSL path (starts with /)
cmd.exe /c start cmd.exe /k wsl.exe --cd /mnt/d/projects bash -i -c "command"
```

**Key features:**
- `/c start` creates new visible window
- `/k` keeps window open after command
- `/d` changes drive when using cd
- `wsl.exe --cd` changes directory within WSL before running command

### WSL

```python
# Unix path
bash -c "cd /mnt/d/projects && command; exec bash"

# Windows path (translated to /mnt/c/... first)
bash -c "cd /mnt/c/projects && command; exec bash"
```

**Key features:**
- Path translation happens before launching
- `exec bash` replaces the process to keep terminal open
- `start_new_session=True` prevents zombie processes

### Linux Native

```python
gnome-terminal -- bash -c "cd /home/user/projects && command; exec bash"
```

**Key features:**
- Uses gnome-terminal (attempts konsole, xterm as fallbacks)
- `--` separates terminal args from command
- `exec bash` keeps terminal open

### macOS

```python
osascript -e 'tell application "Terminal" to do script "cd /Users/name/projects && command"'
```

**Key features:**
- AppleScript provides full control over Terminal.app
- Command includes cd and stays open by default
- No need for exec bash trick

## Session Resume Flow

1. User clicks "Resume" in web UI
2. API endpoint `/api/resume` receives conversation_id
3. Extract working directory from conversation file:
   - Claude Code: read `.jsonl` until `cwd` field found
   - Vibe: read `.json` metadata.environment.working_directory
4. Build command: `claude --resume {id}` or `vibe --resume {id}`
5. Call `platform_manager.open_terminal_with_command(command, cwd)`:
   - Platform detection (windows/wsl/linux/macos)
   - Path translation if needed
   - Platform-specific terminal launch
6. Return success response with details

## Testing

### Test Scripts

1. **Basic Test** (`tests/test_terminal_basic.py`)
   - Non-interactive verification of platform detection
   - Path translation tests
   - CWD translation preview
   - Run: `python tests/test_terminal_basic.py`

2. **Interactive Test** (`tests/test_terminal_launch.py`)
   - Manual terminal launching tests
   - Tests native paths, foreign paths, no path scenarios
   - Prompts before each launch for verification
   - Run: `python tests/test_terminal_launch.py`

### Test Matrix

| Platform | Native Path | Foreign Path | No Path | Status |
|----------|-------------|--------------|---------|--------|
| Windows | D:\test | /mnt/d/test | None | ✓ Tested |
| WSL | /mnt/d/test | D:\test | None | Manual test needed |
| Linux | /home/test | N/A | None | Manual test needed |
| macOS | /Users/test | N/A | None | Manual test needed |

## Implementation Files

| File | Purpose |
|------|---------|
| `src/searchat/platform_utils.py` | Terminal launcher, platform manager |
| `src/searchat/config/path_resolver.py` | Platform detection, path translation |
| `src/searchat/api/routers/conversations.py` | Resume endpoint implementation |
| `tests/test_platform_utils.py` | Platform detection and terminal launch tests |

## Security Considerations

### No Shell Injection

All terminal launching uses subprocess argument lists, not `shell=True`:

```python
# Safe - subprocess handles quoting
subprocess.Popen(['cmd.exe', '/c', 'start', 'cmd.exe', '/k', command])

# Unsafe - shell parsing vulnerabilities
subprocess.Popen(f'cmd.exe /c start cmd.exe /k {command}', shell=True)
```

### Path Validation

- Paths are normalized and translated, not executed directly
- No user input is evaluated as shell commands
- Working directories are read from trusted conversation files

## Known Limitations

1. **WSL Terminal Launcher**: Currently launches bash within WSL. Could be enhanced to launch Windows Terminal from WSL using `cmd.exe /c start wt.exe` pattern.

2. **Linux Terminal Emulators**: Only tests gnome-terminal. Fallback to konsole/xterm is implemented but not tested.

3. **Windows Terminal Detection**: Uses cmd.exe for compatibility. Could detect and prefer Windows Terminal if available.

4. **Path Validation**: No validation that translated paths actually exist. Assumes conversation files contain valid paths.

## Future Enhancements

1. **Prefer Windows Terminal**: Detect if `wt.exe` is available and use it instead of cmd.exe on Windows
2. **WSL ↔ Windows**: From WSL, detect Windows paths and launch Windows Terminal instead of bash
3. **Terminal Preference**: Allow users to configure preferred terminal emulator
4. **Path Validation**: Verify paths exist before launching terminal
5. **Error Handling**: Better error messages for missing terminal emulators or invalid paths


