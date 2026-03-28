/// Get normalization divisor using percentile-based approach.
///
/// For result sets with >= min_samples items, uses the specified percentile
/// as the divisor. For smaller sets, falls back to max normalization.
///
/// Returns a divisor value (never 0, minimum 1e-10 for numerical stability).
pub fn percentile_normalize(scores: &[f64], percentile: f64, min_samples: usize) -> f64 {
    if scores.is_empty() {
        return 1.0;
    }

    let divisor = if scores.len() >= min_samples {
        percentile_value(scores, percentile)
    } else {
        scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    };

    divisor.max(1e-10)
}

/// Normalize a single score, capping at 1.0.
pub fn normalize_score(score: f64, divisor: f64) -> f64 {
    (score / divisor).min(1.0)
}

/// Compute the p-th percentile value from an unsorted slice (0–100).
fn percentile_value(scores: &[f64], p: f64) -> f64 {
    if scores.is_empty() {
        return 0.0;
    }
    let mut sorted = scores.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_scores_returns_one() {
        assert_eq!(percentile_normalize(&[], 95.0, 10), 1.0);
    }

    #[test]
    fn small_set_uses_max() {
        let scores = vec![0.5, 1.0, 0.8];
        let d = percentile_normalize(&scores, 95.0, 10);
        assert!((d - 1.0).abs() < 1e-9);
    }

    #[test]
    fn normalize_caps_at_one() {
        assert_eq!(normalize_score(2.0, 1.0), 1.0);
    }

    #[test]
    fn normalize_scales_down() {
        assert!((normalize_score(0.5, 1.0) - 0.5).abs() < 1e-9);
    }
}
