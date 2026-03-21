"""Query classification for adaptive hybrid search weights.

Classifies queries into categories to determine optimal BM25/semantic weight balance:
- KEYWORD: File paths, line numbers, exact identifiers → high BM25 weight
- SEMANTIC: Conceptual terms, architectural discussions → high semantic weight
- BALANCED: Mixed queries → equal weights
"""

import re
from dataclasses import dataclass
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class QueryClassification:
    """Result of query classification."""
    query_type: str  # "keyword", "semantic", or "balanced"
    bm25_weight: float
    semantic_weight: float
    confidence: float  # How confident we are in this classification (0-1)
    reasoning: str  # Why this classification was chosen


class QueryClassifier:
    """Classifies queries to determine optimal hybrid search weights."""

    # Patterns that indicate keyword-heavy queries
    KEYWORD_PATTERNS = [
        r'\.\w+:\d+',  # file.py:142 (file with line number)
        r'\b\w+\.(py|ts|js|tsx|jsx|java|cpp|c|h|go|rs|rb|php|md|txt|json|toml|yaml|yml|sql|sh|bat|ps1|css|html)\b',  # file extensions
        r'\bline\s+\d+',  # "line 42"
        r'error\s+code\s+\d+',  # "error code 500"
        r'\b[A-Z_]{5,}\b',  # CONSTANT_NAMES (all caps with underscores, 5+ chars)
        r'\bfunction\s+\w+',  # "function foo"
        r'\bclass\s+\w+',  # "class Bar"
        r'\bdef\s+\w+',  # "def baz"
        r'\bimport\s+[\w.]+',  # "import xyz" or "import x.y.z"
        r'\bfrom\s+[\w.]+\s+import',  # "from abc import"
        r'\bTODO\b|\bFIXME\b|\bNOTE\b|\bBUG\b',  # code comment markers
        r'[/_]\w+[/_]',  # path separators like /abc/ or_abc_
    ]

    # Terms that indicate semantic/conceptual queries
    SEMANTIC_TERMS = [
        'architecture', 'pattern', 'design', 'approach', 'strategy', 'concept',
        'principle', 'methodology', 'framework', 'paradigm', 'philosophy',
        'rationale', 'tradeoff', 'pros and cons', 'advantages', 'disadvantages',
        'best practice', 'anti-pattern', 'recommendation', 'guideline',
        'overview', 'introduction', 'explanation', 'understanding',
        'why', 'how does', 'what is', 'when to', 'should i',
        'comparison', 'difference between', 'versus', 'vs',
        'alternative', 'option', 'choice', 'decide',
    ]

    # Technical action verbs (balanced - could be either)
    TECHNICAL_VERBS = [
        'implement', 'refactor', 'optimize', 'debug', 'fix', 'add', 'remove',
        'update', 'configure', 'setup', 'deploy', 'build', 'test',
    ]

    def __init__(self):
        self.keyword_regex = re.compile('|'.join(f'({p})' for p in self.KEYWORD_PATTERNS), re.IGNORECASE)

    def classify(self, query: str) -> QueryClassification:
        """Classify a query and return optimal weights.

        Args:
            query: Search query string

        Returns:
            QueryClassification with type, weights, and reasoning
        """
        query_lower = query.lower()

        # Count keyword pattern matches
        keyword_matches = len(self.keyword_regex.findall(query))

        # Count semantic term matches
        semantic_matches = sum(1 for term in self.SEMANTIC_TERMS if term in query_lower)

        # Count technical verbs (neutral indicators)
        verb_matches = sum(1 for verb in self.TECHNICAL_VERBS if verb in query_lower)

        # Check for quoted exact phrases (indicates precise keyword search)
        has_quotes = '"' in query or "'" in query

        # Check for file paths (strong keyword indicator)
        has_path = '/' in query or '\\' in query or '.' in query

        # Scoring logic
        keyword_score = keyword_matches * 2.5  # Each pattern match is strong signal
        if has_quotes:
            keyword_score += 2.0
        if has_path and not any(term in query_lower for term in ['architecture', 'design', 'pattern']):
            keyword_score += 1.5

        semantic_score = semantic_matches * 2.0  # Each term is moderate signal

        # Short queries (<15 chars) tend to be keywords if they have file indicators
        if len(query) < 15 and (has_path or keyword_matches > 0):
            keyword_score += 1.0

        # Long queries (>80 chars) with question words tend to be semantic
        if len(query) > 80 and any(q in query_lower for q in ['how', 'why', 'what', 'when', 'where']):
            semantic_score += 2.0

        # Question words at start are strong semantic signal
        has_question_start = any(query_lower.startswith(q) for q in ['how ', 'why ', 'what ', 'when ', 'where ', 'should ', 'can '])
        has_question_word = any(q in query_lower for q in ['how ', 'why ', 'what ', 'when ', 'where '])

        if has_question_start:
            semantic_score += 4.0
        elif has_question_word:
            semantic_score += 2.0

        # Determine classification with different threshold for semantic due to question words
        total_score = keyword_score + semantic_score
        if total_score == 0:
            # No clear signals - default to balanced
            query_type = "balanced"
            bm25_weight = 0.5
            semantic_weight = 0.5
            confidence = 0.3
            reasoning = "No clear keyword or semantic signals detected"
        elif keyword_score > semantic_score * 2 and not has_question_start:
            # Strong keyword signals (but not if starts with question)
            query_type = "keyword"
            bm25_weight = 0.8
            semantic_weight = 0.2
            confidence = min(0.9, keyword_score / (total_score + 1))
            reasoning = f"File paths/identifiers/code patterns detected ({keyword_matches} matches)"
        elif semantic_score > keyword_score * 1.5 or has_question_start:
            # Strong semantic signals or starts with question
            query_type = "semantic"
            bm25_weight = 0.2
            semantic_weight = 0.8
            confidence = min(0.9, semantic_score / (total_score + 1))
            reasoning = f"Conceptual/architectural terms detected ({semantic_matches} matches)" if semantic_matches > 0 else "Question query detected"
        else:
            # Mixed signals or close scores
            query_type = "balanced"
            bm25_weight = 0.5
            semantic_weight = 0.5
            confidence = 0.5
            reasoning = f"Mixed signals: {keyword_matches} keyword patterns, {semantic_matches} semantic terms"

        # Log classification
        logger.info(
            "Query classified as '%s' (confidence=%.2f): kw_score=%.1f, sem_score=%.1f, weights=(%.1f, %.1f)",
            query_type, confidence, keyword_score, semantic_score, bm25_weight, semantic_weight
        )

        return QueryClassification(
            query_type=query_type,
            bm25_weight=bm25_weight,
            semantic_weight=semantic_weight,
            confidence=confidence,
            reasoning=reasoning,
        )
