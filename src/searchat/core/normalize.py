"""Score normalization utilities."""

from typing import List, Sequence
import numpy as np


def percentile_normalize(scores: Sequence[float], percentile: int = 95, min_samples: int = 10) -> float:
    """Get normalization divisor using percentile-based approach.

    For result sets with >= min_samples items, uses the specified percentile
    as the divisor. For smaller sets, falls back to max normalization.

    Args:
        scores: Sequence of raw scores to normalize
        percentile: Percentile to use as divisor (default 95)
        min_samples: Minimum samples needed for percentile normalization (default 10)

    Returns:
        Divisor value (never 0, minimum 1e-10 for numerical stability)
    """
    if not scores:
        return 1.0

    scores_array = np.array(list(scores))

    if len(scores_array) >= min_samples:
        divisor = float(np.percentile(scores_array, percentile))
    else:
        divisor = float(np.max(scores_array))

    return max(divisor, 1e-10)


def normalize_score(score: float, divisor: float) -> float:
    """Normalize a single score, capping at 1.0.

    Args:
        score: Raw score to normalize
        divisor: Divisor from percentile_normalize()

    Returns:
        Normalized score in range [0, 1]
    """
    return min(score / divisor, 1.0)
