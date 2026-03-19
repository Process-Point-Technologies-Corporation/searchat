"""Tests for LLM interface and response parsing."""
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from searchat.palace.llm import (
    CLIDistillationLLM,
    DistillationInput,
    DistillationOutput,
    RoomAssignment,
)


class TestPromptFormatting:
    def test_build_prompt_contains_messages(self):
        llm = CLIDistillationLLM(model="haiku")
        inp = DistillationInput(
            conversation_id="conv-1",
            project_id="proj-1",
            messages=[
                {"role": "user", "sequence": 0, "content": "hello world"},
                {"role": "assistant", "sequence": 1, "content": "hi there"},
            ],
            ply_start=0,
            ply_end=1,
        )
        prompt = llm._build_prompt(inp)
        assert "hello world" in prompt
        assert "hi there" in prompt
        assert "proj-1" in prompt
        assert "0-1" in prompt

    def test_build_prompt_handles_missing_fields(self):
        llm = CLIDistillationLLM(model="haiku")
        inp = DistillationInput(
            conversation_id="conv-1",
            project_id="proj-1",
            messages=[{"content": "text only"}],
            ply_start=0,
            ply_end=0,
        )
        prompt = llm._build_prompt(inp)
        assert "text only" in prompt


class TestResponseParsing:
    def test_parse_valid_json(self):
        llm = CLIDistillationLLM(model="haiku")
        raw = json.dumps({
            "exchange_core": "Implemented feature X",
            "specific_context": "Used pattern Y",
            "room_assignments": [
                {
                    "room_type": "file",
                    "room_key": "src/main.py",
                    "room_label": "main.py",
                    "relevance": 0.9,
                }
            ],
        })
        result = llm._parse_response(raw)
        assert isinstance(result, DistillationOutput)
        assert result.exchange_core == "Implemented feature X"
        assert len(result.room_assignments) == 1
        assert result.room_assignments[0].room_type == "file"

    def test_parse_json_with_markdown_fencing(self):
        llm = CLIDistillationLLM(model="haiku")
        raw = '```json\n{"exchange_core": "test", "specific_context": "ctx", "room_assignments": []}\n```'
        result = llm._parse_response(raw)
        assert result.exchange_core == "test"

    def test_parse_malformed_json_raises(self):
        llm = CLIDistillationLLM(model="haiku")
        with pytest.raises(RuntimeError, match="Malformed JSON"):
            llm._parse_response("this is not json at all")

    def test_parse_missing_required_field_raises(self):
        llm = CLIDistillationLLM(model="haiku")
        raw = json.dumps({"exchange_core": "test"})
        with pytest.raises(KeyError):
            llm._parse_response(raw)

    def test_parse_empty_arrays(self):
        llm = CLIDistillationLLM(model="haiku")
        raw = json.dumps({
            "exchange_core": "test",
            "specific_context": "ctx",
            "room_assignments": [],
        })
        result = llm._parse_response(raw)
        assert result.room_assignments == []

    def test_parse_tolerates_extra_fields(self):
        """LLM may still return files_touched — parser ignores extra fields."""
        llm = CLIDistillationLLM(model="haiku")
        raw = json.dumps({
            "exchange_core": "test",
            "specific_context": "ctx",
            "files_touched": [{"path": "fake.py", "action": "modified"}],
            "room_assignments": [],
        })
        result = llm._parse_response(raw)
        assert result.exchange_core == "test"
        assert not hasattr(result, "files_touched")


class TestCliInvocationErrors:
    def test_invoke_cli_uses_stdout_when_stderr_is_empty(self):
        llm = CLIDistillationLLM(model="haiku")

        completed = subprocess.CompletedProcess(
            args=["claude", "--print", "--model", "haiku"],
            returncode=1,
            stdout="Failed to authenticate. OAuth token expired.",
            stderr="",
        )

        with patch("searchat.palace.llm.shutil.which", return_value="claude"):
            with patch.object(CLIDistillationLLM, "_get_session_dir", return_value=Path("D:/searchat-fixtures/.missing-claude-sessions")):
                with patch("searchat.palace.llm.subprocess.run", return_value=completed):
                    with pytest.raises(RuntimeError, match="OAuth token expired"):
                        llm._invoke_cli("prompt")

    def test_invoke_cli_reports_missing_output_with_manual_check_hint(self):
        llm = CLIDistillationLLM(provider="claude", model="claude-haiku-4-5-20251001")

        completed = subprocess.CompletedProcess(
            args=["claude", "--print", "--model", "claude-haiku-4-5-20251001"],
            returncode=1,
            stdout="",
            stderr="",
        )

        with patch("searchat.palace.llm.shutil.which", return_value="claude"):
            with patch.object(CLIDistillationLLM, "_get_session_dir", return_value=Path("D:/searchat-fixtures/.missing-claude-sessions")):
                with patch("searchat.palace.llm.subprocess.run", return_value=completed):
                    with pytest.raises(RuntimeError, match=r"claude --print --model claude-haiku"):
                        llm._invoke_cli("prompt")

    def test_invoke_claude_cli_uses_structured_output_flags(self):
        llm = CLIDistillationLLM(provider="claude", model="claude-haiku-4-5-20251001")
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='{"exchange_core":"test","specific_context":"ctx","room_assignments":[]}',
            stderr="",
        )

        with patch("searchat.palace.llm.shutil.which", return_value="claude"):
            with patch.object(CLIDistillationLLM, "_get_session_dir", return_value=Path("D:/searchat-fixtures/.missing-claude-sessions")):
                with patch("searchat.palace.llm.subprocess.run", return_value=completed) as run_mock:
                    llm._invoke_cli("prompt")

        cmd = run_mock.call_args.args[0]
        assert "--output-format" in cmd
        assert "--json-schema" in cmd
        assert "--no-session-persistence" in cmd

    def test_invoke_openai_cli_reads_output_file(self):
        llm = CLIDistillationLLM(provider="openai", model="gpt-5")

        def _fake_run(cmd, **kwargs):
            output_index = cmd.index("--output-last-message") + 1
            output_path = cmd[output_index]
            with open(output_path, "w", encoding="utf-8") as handle:
                handle.write('{"exchange_core":"test","specific_context":"ctx","room_assignments":[]}')
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        with patch("searchat.palace.llm.shutil.which", side_effect=lambda name: "codex" if name == "codex" else "pwsh"):
            with patch.object(CLIDistillationLLM, "_get_session_dir", return_value=Path("D:/searchat-fixtures/.missing-codex-sessions")):
                with patch("searchat.palace.llm.subprocess.run", side_effect=_fake_run) as run_mock:
                    raw = llm._invoke_cli("prompt")

        cmd = run_mock.call_args.args[0]
        assert cmd[0] == "codex"
        assert "exec" in cmd
        assert "--output-schema" in cmd
        assert "--ephemeral" in cmd
        assert raw.startswith('{"exchange_core":"test"')

    def test_invoke_openai_cli_reports_codex_command_on_failure(self):
        llm = CLIDistillationLLM(provider="openai", model="gpt-5")
        completed = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="")

        with patch("searchat.palace.llm.shutil.which", side_effect=lambda name: "codex" if name == "codex" else "pwsh"):
            with patch.object(CLIDistillationLLM, "_get_session_dir", return_value=Path("D:/searchat-fixtures/.missing-codex-sessions")):
                with patch("searchat.palace.llm.subprocess.run", return_value=completed):
                    with pytest.raises(RuntimeError, match=r"codex exec --model gpt-5"):
                        llm._invoke_cli("prompt")

