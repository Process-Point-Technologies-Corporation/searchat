"""Tests for score normalization utilities."""

import pytest
import numpy as np

from searchat.core.normalize import percentile_normalize, normalize_score


class TestPercentileNormalize:
    """Tests for percentile_normalize function."""

    def test_empty_scores_returns_one(self):
        assert percentile_normalize([]) == 1.0

    def test_single_score_uses_max(self):
        assert percentile_normalize([5.0]) == 5.0

    def test_few_scores_uses_max(self):
        # Below min_samples (10), should use max
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert percentile_normalize(scores) == 5.0

    def test_many_scores_uses_percentile(self):
        # Above min_samples, should use 95th percentile
        scores = list(range(1, 101))  # 1 to 100
        result = percentile_normalize(scores)
        expected = np.percentile(scores, 95)
        assert result == expected

    def test_custom_percentile(self):
        scores = list(range(1, 101))
        result = percentile_normalize(scores, percentile=90)
        expected = np.percentile(scores, 90)
        assert result == expected

    def test_custom_min_samples(self):
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        # With min_samples=5, should use percentile
        result = percentile_normalize(scores, min_samples=5)
        expected = np.percentile(scores, 95)
        assert result == expected

    def test_never_returns_zero(self):
        # Even with all zeros, should return minimum threshold
        scores = [0.0, 0.0, 0.0]
        result = percentile_normalize(scores)
        assert result > 0


class TestNormalizeScore:
    """Tests for normalize_score function."""

    def test_basic_normalization(self):
        assert normalize_score(5.0, 10.0) == 0.5

    def test_caps_at_one(self):
        # Score above divisor should cap at 1.0
        assert normalize_score(15.0, 10.0) == 1.0

    def test_zero_score(self):
        assert normalize_score(0.0, 10.0) == 0.0

    def test_equal_to_divisor(self):
        assert normalize_score(10.0, 10.0) == 1.0


class TestIntegration:
    """Integration tests for normalization flow."""

    def test_sparse_results_stability(self):
        # Simulate sparse results where max could inflate scores
        scores = [0.01, 0.02, 100.0]  # One outlier
        divisor = percentile_normalize(scores)
        # With only 3 items, uses max (100.0)
        assert divisor == 100.0

        # Small scores normalize to small values
        assert normalize_score(0.01, divisor) < 0.01
        assert normalize_score(0.02, divisor) < 0.01

    def test_typical_results_with_outlier(self):
        # 20 items, 19 normal + 1 outlier
        scores = [0.5 + i * 0.02 for i in range(19)] + [10.0]
        divisor = percentile_normalize(scores)

        # p95 should ignore the extreme outlier
        assert divisor < 10.0
        # Most normal scores should normalize to reasonable range
        normalized = [normalize_score(s, divisor) for s in scores[:-1]]
        # Average should be around 0.5-0.8
        avg = sum(normalized) / len(normalized)
        assert 0.4 <= avg <= 0.9
