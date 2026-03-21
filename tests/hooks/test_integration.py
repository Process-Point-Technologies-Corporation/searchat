"""Integration tests with real Haiku calls."""
import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def has_claude_cli():
    """Check if claude CLI is available."""
    return shutil.which("claude") is not None


@pytest.mark.skipif(not has_claude_cli(), reason="claude CLI not in PATH")
@pytest.mark.slow
class TestHaikuIntegration:
    """Tests that make real Haiku API calls."""

    def test_haiku_returns_valid_json(self):
        """Real Haiku call, parse response."""
        from hooks.distill_turn import invoke_haiku, parse_response

        prompt = """Distill this conversation exchange into JSON:

- "exchange_core": 1-2 sentences. What was accomplished or decided? Use specific terms from the text.
- "specific_context": One concrete detail: number, error message, parameter, or file path. Copy exactly.
- "tags": 2-4 keywords for retrieval (lowercase, underscore-separated).

User: How do I implement a binary search algorithm in Python?
Assistant: Binary search works by repeatedly dividing the search interval in half. The key is to maintain left and right pointers and compare the middle element with the target. Time complexity is O(log n).

Respond with ONLY valid JSON."""

        raw_response = invoke_haiku(prompt)
        result = parse_response(raw_response)

        assert "exchange_core" in result
        assert "specific_context" in result
        assert "tags" in result
        assert isinstance(result["tags"], list)

    def test_haiku_minimal_input(self):
        """Edge case: very short exchange."""
        from hooks.distill_turn import invoke_haiku, parse_response

        prompt = """Distill this conversation exchange into JSON:

- "exchange_core": 1-2 sentences. What was accomplished or decided? Use specific terms from the text.
- "specific_context": One concrete detail: number, error message, parameter, or file path. Copy exactly.
- "tags": 2-4 keywords for retrieval (lowercase, underscore-separated).

User: hi
Assistant: Hello! How can I help you today?

Respond with ONLY valid JSON."""

        raw_response = invoke_haiku(prompt)
        result = parse_response(raw_response)

        assert "exchange_core" in result
        # Even minimal exchanges should produce valid output
        assert isinstance(result.get("tags", []), list)
