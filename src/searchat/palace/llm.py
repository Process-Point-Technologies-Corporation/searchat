"""LLM interface for conversation distillation.

Two execution modes:
- Interactive: current agent session drives distillation directly.
- Batch: CLIDistillationLLM invokes a subscription-backed CLI subprocess.

Distillation extracts the essence of an exchange and places it in the memory palace.
The distillation prompt is loaded from config (settings.toml [distillation] prompt).
"""
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

from searchat.config.constants import (
    DEFAULT_DISTILLATION_CLI_MODEL,
    DEFAULT_DISTILLATION_CLI_MODEL_OPENAI,
    DEFAULT_DISTILLATION_PROMPT,
)


@dataclass
class DistillationInput:
    """Input to the distillation LLM."""
    conversation_id: str
    project_id: str
    messages: List[dict]  # [{role, content, sequence, timestamp}, ...]
    ply_start: int
    ply_end: int


@dataclass
class RoomAssignment:
    """A room assignment produced by the LLM."""
    room_type: str
    room_key: str
    room_label: str
    relevance: float


@dataclass
class DistillationOutput:
    """Output from the distillation LLM."""
    exchange_core: str
    specific_context: str
    room_assignments: List[RoomAssignment] = field(default_factory=list)


class DistillationLLM(ABC):
    """Abstract base for distillation LLM providers."""

    @abstractmethod
    def distill(self, inputs: List[DistillationInput]) -> List[DistillationOutput]:
        """Distill a batch of exchanges into structured outputs."""
        ...


DISTILLATION_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "exchange_core": {"type": "string"},
        "specific_context": {"type": "string"},
        "room_assignments": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "room_type": {"type": "string", "enum": ["file", "concept", "workflow"]},
                    "room_key": {"type": "string"},
                    "room_label": {"type": "string"},
                    "relevance": {"type": "number"},
                },
                "required": ["room_type", "room_key", "room_label", "relevance"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["exchange_core", "specific_context", "room_assignments"],
    "additionalProperties": False,
}


class CLIDistillationLLM(DistillationLLM):
    """Batch mode: invokes a subscription-backed CLI subprocess."""

    _DEFAULT_MODELS = {
        "claude": DEFAULT_DISTILLATION_CLI_MODEL,
        "openai": DEFAULT_DISTILLATION_CLI_MODEL_OPENAI,
    }
    _CLI_NAMES = {"claude": "claude", "openai": "codex"}

    def __init__(
        self,
        provider: str = "auto",
        model: str = DEFAULT_DISTILLATION_CLI_MODEL,
        prompt_template: Optional[str] = None,
    ):
        requested = (provider or "auto").strip().lower()
        if requested not in {"claude", "openai", "auto"}:
            raise ValueError(f"Unsupported distillation provider: {provider!r}")
        self.provider, self.model = self._resolve_provider(requested, model)
        self.prompt_template = prompt_template or DEFAULT_DISTILLATION_PROMPT

    def distill(self, inputs: List[DistillationInput]) -> List[DistillationOutput]:
        # Cache CLI path and session snapshot across the batch
        cli_path = self._find_cli()
        session_dir = self._get_session_dir()
        before_jsonls = self._snapshot_jsonl_files(session_dir)

        results = []
        try:
            for inp in inputs:
                prompt = self._build_prompt(inp)
                try:
                    raw = self._invoke_cli(prompt, cli_path=cli_path)
                except subprocess.TimeoutExpired as e:
                    raise RuntimeError(
                        f"{self.provider} distillation CLI timed out for exchange plies "
                        f"{inp.ply_start}-{inp.ply_end}"
                    ) from e
                output = self._parse_response(raw)
                results.append(output)
        finally:
            self._cleanup_side_effect_jsonls(session_dir, before_jsonls)
        return results

    def _build_prompt(self, inp: DistillationInput) -> str:
        messages_text = "\n".join(
            f"[{m.get('role', 'unknown')}] (seq {m.get('sequence', '?')}): {m.get('content', '')}"
            for m in inp.messages
        )
        return self.prompt_template.format(
            project_id=inp.project_id,
            ply_start=inp.ply_start,
            ply_end=inp.ply_end,
            messages_text=messages_text,
        )

    def _invoke_cli(self, prompt: str, cli_path: str | None = None) -> str:
        if cli_path is None:
            cli_path = self._find_cli()
        raw = ""
        output_path: Optional[Path] = None
        schema_path: Optional[Path] = None

        try:
            cmd, output_path, schema_path = self._build_command(cli_path)
            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=300,
            )
            raw = self._extract_output(result, output_path)
        finally:
            for temp_path in (output_path, schema_path):
                if temp_path is None:
                    continue
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass

        if result.returncode != 0:
            failure_details = self._format_failure_output(
                stdout=result.stdout,
                stderr=result.stderr,
                output_text=raw,
            )
            raise RuntimeError(
                f"{self._provider_label()} CLI failed (exit {result.returncode}) using "
                f"'{self._display_command()}': {failure_details}"
            )

        if not raw.strip():
            raise RuntimeError(
                f"{self._provider_label()} CLI returned empty output using "
                f"'{self._display_command()}'."
            )

        return raw.strip()

    @classmethod
    def _resolve_provider(cls, requested: str, configured_model: str) -> tuple[str, str]:
        """Resolve provider and model by probing which CLI is available.

        Returns (provider, model) where provider is "claude" or "openai".
        """
        if requested == "auto":
            order = ["claude", "openai"]
        else:
            other = "openai" if requested == "claude" else "claude"
            order = [requested, other]

        for provider in order:
            cli_name = cls._CLI_NAMES[provider]
            if shutil.which(cli_name) is None:
                continue
            model = cls._pick_model(provider, configured_model)
            if provider != order[0]:
                logger.warning(
                    "%s CLI not found; using %s instead (model: %s)",
                    cls._CLI_NAMES[order[0]],
                    cli_name,
                    model,
                )
            else:
                logger.info("Distillation provider: %s (model: %s)", provider, model)
            return provider, model

        tried = ", ".join(cls._CLI_NAMES[p] for p in order)
        raise RuntimeError(f"No distillation CLI found on PATH (tried: {tried})")

    @classmethod
    def _pick_model(cls, provider: str, configured_model: str) -> str:
        """Return configured_model if it matches the provider, else the default."""
        if provider == "claude" and configured_model.startswith("claude-"):
            return configured_model
        if provider == "openai" and configured_model.startswith("gpt-"):
            return configured_model
        return cls._DEFAULT_MODELS[provider]

    def _find_cli(self) -> str:
        cli_name = self._CLI_NAMES[self.provider]
        cli_path = shutil.which(cli_name)
        if cli_path is None:
            raise RuntimeError(f"{self._provider_label()} CLI not found in PATH")
        return cli_path

    def _build_command(self, cli_path: str) -> tuple[list[str], Optional[Path], Optional[Path]]:
        base_command = self._wrap_cli_command(cli_path)
        if self.provider == "claude":
            return (
                base_command
                + [
                    "--print",
                    "--model",
                    self.model,
                    "--output-format",
                    "json",
                    "--json-schema",
                    json.dumps(DISTILLATION_OUTPUT_SCHEMA),
                    "--no-session-persistence",
                ],
                None,
                None,
            )

        schema_file = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            encoding="utf-8",
            delete=False,
        )
        output_file = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            encoding="utf-8",
            delete=False,
        )
        try:
            json.dump(DISTILLATION_OUTPUT_SCHEMA, schema_file)
            schema_file.flush()
        finally:
            schema_file.close()
            output_file.close()

        return (
            base_command
            + [
                "exec",
                "--model",
                self.model,
                "--sandbox",
                "read-only",
                "--skip-git-repo-check",
                "--ephemeral",
                "--color",
                "never",
                "--output-schema",
                schema_file.name,
                "--output-last-message",
                output_file.name,
                "-",
            ],
            Path(output_file.name),
            Path(schema_file.name),
        )

    def _wrap_cli_command(self, cli_path: str) -> list[str]:
        suffix = Path(cli_path).suffix.lower()
        if sys.platform == "win32" and suffix == ".ps1":
            powershell = (
                shutil.which("pwsh")
                or shutil.which("powershell")
                or shutil.which("powershell.exe")
            )
            if powershell is None:
                raise RuntimeError(
                    f"PowerShell is required to run {self._provider_label()} wrapper script: {cli_path}"
                )
            return [powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", cli_path]

        if sys.platform == "win32" and suffix in {".cmd", ".bat"}:
            comspec = os.environ.get("COMSPEC", "cmd.exe")
            return [comspec, "/c", cli_path]

        return [cli_path]

    def _provider_label(self) -> str:
        return "Claude" if self.provider == "claude" else "Codex"

    def _display_command(self) -> str:
        if self.provider == "claude":
            return f"claude --print --model {self.model}"
        return f"codex exec --model {self.model}"

    def _get_session_dir(self) -> Path:
        if self.provider == "claude":
            return Path.home() / ".claude" / "projects"
        return Path.home() / ".codex" / "sessions"

    def _snapshot_jsonl_files(self, session_dir: Path) -> set[Path]:
        if not session_dir.exists():
            return set()
        return set(session_dir.rglob("*.jsonl"))

    def _cleanup_side_effect_jsonls(self, session_dir: Path, before_jsonls: set[Path]) -> None:
        if not session_dir.exists():
            return
        after_jsonls = set(session_dir.rglob("*.jsonl"))
        new_jsonls = after_jsonls - before_jsonls
        for jsonl_path in new_jsonls:
            try:
                jsonl_path.unlink()
                logger.debug("Deleted %s side-effect JSONL: %s", self.provider, jsonl_path)
            except OSError:
                pass

    def _extract_output(
        self,
        result: subprocess.CompletedProcess[str],
        output_path: Optional[Path],
    ) -> str:
        if self.provider == "claude":
            raw = (result.stdout or "").strip()
            # Claude CLI with --output-format json wraps the response in a
            # metadata envelope. Extract structured_output or result field.
            if raw.startswith("{"):
                try:
                    envelope = json.loads(raw)
                    if "structured_output" in envelope and envelope["structured_output"]:
                        return json.dumps(envelope["structured_output"])
                    if "result" in envelope and envelope["result"]:
                        return envelope["result"]
                except (json.JSONDecodeError, KeyError):
                    pass
            return raw

        if output_path is None or not output_path.exists():
            return ""
        return output_path.read_text(encoding="utf-8", errors="replace").strip()

    @staticmethod
    def _format_failure_output(stdout: str, stderr: str, output_text: str = "") -> str:
        stderr_text = (stderr or "").strip()
        stdout_text = (stdout or "").strip()
        output_text = (output_text or "").strip()

        if stderr_text and stdout_text:
            return f"stderr: {stderr_text[:500]} | stdout: {stdout_text[:500]}"
        if stderr_text:
            return f"stderr: {stderr_text[:500]}"
        if stdout_text:
            return f"stdout: {stdout_text[:500]}"
        if output_text:
            return f"output: {output_text[:500]}"
        return "no stdout/stderr captured."

    def _parse_response(self, raw: str) -> DistillationOutput:
        # Strip markdown fencing if present
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (fencing)
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Malformed JSON response from distillation LLM: {e}\nRaw: {raw[:500]}"
            )

        if not isinstance(data, dict):
            raise RuntimeError(
                f"Expected JSON object from distillation LLM, got {type(data).__name__}. "
                f"Raw: {raw[:500]}"
            )

        rooms = [
            RoomAssignment(
                room_type=r["room_type"],
                room_key=r["room_key"],
                room_label=r["room_label"],
                relevance=r["relevance"],
            )
            for r in data.get("room_assignments", [])
        ]

        return DistillationOutput(
            exchange_core=data["exchange_core"],
            specific_context=data["specific_context"],
            room_assignments=rooms,
        )
