#!/usr/bin/env python3
"""
distill-turn: Per-turn distillation hook for Claude Code.

DISABLED (2026-02-15): Removed from ~/.claude/settings.local.json Stop hooks.
Claude Code does not kill child processes on session exit/crash. Each invocation
spawns `claude --print` via subprocess (~670MB with loaded models). Orphaned
processes accumulate across crashed sessions — 90+ instances / ~19GB RAM observed.
See: https://github.com/anthropics/claude-code/issues/25963

Re-enable when: Claude Code implements process group cleanup on session exit,
OR this hook is rewritten to use Anthropic SDK directly (no subprocess).

Stop hook that extracts the last user-assistant exchange from the transcript,
sends it to Haiku for distillation, and stores the result in the memory palace.

Usage:
    Configured as a Stop hook in ~/.claude/settings.local.json
    Receives JSON on stdin with transcript_path, stop_hook_active, session_id, cwd
"""
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

from searchat.config import Config


# Use placeholders instead of format strings to avoid tool issues
USER_PLACEHOLDER = "<<USER_TEXT>>"
ASSISTANT_PLACEHOLDER = "<<ASSISTANT_TEXT>>"

# Maximum characters to send to Haiku (prevents timeout on large exchanges)
MAX_TEXT_LENGTH = 4000

# Load distillation prompt from config
config = Config.load()
DISTILLATION_PROMPT = config.distillation.perturn_prompt.replace("{user_text}", USER_PLACEHOLDER).replace("{assistant_text}", ASSISTANT_PLACEHOLDER)


def read_transcript(path: Path) -> list:
    """Read transcript JSONL and return list of messages."""
    messages = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            # Skip non-message entries
            if entry.get("type") not in ("user", "assistant"):
                continue
            msg = entry.get("message", {})
            if msg.get("role") in ("user", "assistant"):
                messages.append(msg)
    return messages


def extract_text_from_content(content) -> str:
    """Extract text from message content (string or array of blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    texts.append(block.get("text", ""))
                elif block.get("type") == "thinking":
                    pass  # Skip thinking blocks
                # Skip tool_use and tool_result blocks
            elif isinstance(block, str):
                texts.append(block)
        return " ".join(texts)
    return ""


def truncate_text(text: str, max_length: int = MAX_TEXT_LENGTH) -> str:
    """Truncate text to max_length, adding ellipsis if truncated."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "... [truncated]"


def extract_last_exchange(messages: list) -> tuple:
    """Extract the last user-assistant text exchange."""
    if not messages:
        return "", ""

    last_user = ""
    last_assistant = ""

    # Find last user message with text content
    for msg in reversed(messages):
        if msg.get("role") == "user":
            text = extract_text_from_content(msg.get("content", ""))
            if text.strip():
                last_user = truncate_text(text)
                break

    # Find last assistant message with text content
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            text = extract_text_from_content(msg.get("content", ""))
            if text.strip():
                last_assistant = truncate_text(text)
                break

    return last_user, last_assistant


def parse_response(raw: str) -> dict:
    """Parse LLM response, extracting JSON from markdown fencing or raw text."""
    import re

    text = raw.strip()

    # Try to extract JSON from markdown code blocks
    json_match = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', text)
    if json_match:
        text = json_match.group(1).strip()

    # If still not valid JSON, try to find JSON object in the text
    if not text.startswith('{'):
        brace_match = re.search(r'\{[\s\S]*\}', text)
        if brace_match:
            text = brace_match.group(0)

    if not text:
        raise ValueError(f"No JSON found. Raw: {raw[:500]}")

    return json.loads(text)


def invoke_haiku(prompt: str) -> str:
    """Invoke claude CLI with Haiku model."""
    claude_cmd = shutil.which("claude")
    if claude_cmd is None:
        raise RuntimeError("claude CLI not found in PATH")

    cmd = [claude_cmd, "--print", "--model", "haiku"]
    use_shell = sys.platform == "win32" and claude_cmd.lower().endswith(".cmd")

    result = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        shell=use_shell,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"claude CLI failed (exit {result.returncode}): {result.stderr}"
        )
    output = result.stdout.strip()
    if not output:
        raise RuntimeError(f"claude CLI returned empty output. stderr: {result.stderr}")
    return output


def should_exit_early(hook_input: dict) -> bool:
    """Check if we should exit early (loop prevention)."""
    return hook_input.get("stop_hook_active", False)


def get_learnings_path(cwd: str) -> Path:
    """Get path to learnings JSONL for this project."""
    project_name = Path(cwd).name
    learnings_dir = Path.home() / ".claude" / "learnings"
    learnings_dir.mkdir(parents=True, exist_ok=True)
    return learnings_dir / f"{project_name}.jsonl"


def store_distillation(result: dict, cwd: str, session_id: str):
    """Store distilled result to learnings file."""
    path = get_learnings_path(cwd)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "project": Path(cwd).name,
        **result,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    """Main hook entry point."""
    # Read hook input from stdin
    stdin_data = sys.stdin.read().strip()
    if not stdin_data:
        sys.exit(0)

    hook_input = json.loads(stdin_data)

    # Loop prevention
    if should_exit_early(hook_input):
        sys.exit(0)

    transcript_path = hook_input.get("transcript_path")
    if not transcript_path or not Path(transcript_path).exists():
        sys.exit(0)

    session_id = hook_input.get("session_id", "unknown")
    cwd = hook_input.get("cwd", ".")

    # Read and extract last exchange
    messages = read_transcript(Path(transcript_path))
    user_text, assistant_text = extract_last_exchange(messages)

    if not user_text or not assistant_text:
        sys.exit(0)

    # Build and send prompt
    prompt = DISTILLATION_PROMPT.replace(USER_PLACEHOLDER, user_text).replace(ASSISTANT_PLACEHOLDER, assistant_text)

    raw_response = invoke_haiku(prompt)
    result = parse_response(raw_response)

    # Store result
    store_distillation(result, cwd, session_id)


if __name__ == "__main__":
    main()
