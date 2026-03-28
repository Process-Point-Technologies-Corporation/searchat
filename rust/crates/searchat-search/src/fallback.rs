/// Progressive fallback for palace searches with insufficient scoped results.
///
/// Three-tier strategy (mirrors Python `progressive_fallback.py`):
///   Tier 1 — scoped search (resolved projects only)
///   Tier 2 — related projects expansion (if results < min_results)
///   Tier 3 — unscoped search (if still insufficient)
use log::{info, warn};
use std::collections::HashMap;

use crate::error::SearchError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackTier {
    Scoped,
    Related,
    Unscoped,
}

impl FallbackTier {
    pub fn as_str(&self) -> &'static str {
        match self {
            FallbackTier::Scoped => "scoped",
            FallbackTier::Related => "related",
            FallbackTier::Unscoped => "unscoped",
        }
    }
}

#[derive(Debug, Default)]
pub struct FallbackStats {
    pub scoped: u64,
    pub related: u64,
    pub unscoped: u64,
}

impl FallbackStats {
    pub fn total(&self) -> u64 {
        self.scoped + self.related + self.unscoped
    }

    pub fn to_map(&self) -> HashMap<String, serde_json::Value> {
        let total = self.total();
        let pct = |n: u64| {
            if total == 0 {
                0.0
            } else {
                100.0 * n as f64 / total as f64
            }
        };
        let mut m = HashMap::new();
        m.insert(
            "total_searches".to_string(),
            serde_json::json!(total),
        );
        m.insert("scoped_count".to_string(), serde_json::json!(self.scoped));
        m.insert("scoped_pct".to_string(), serde_json::json!(pct(self.scoped)));
        m.insert("related_count".to_string(), serde_json::json!(self.related));
        m.insert("related_pct".to_string(), serde_json::json!(pct(self.related)));
        m.insert("unscoped_count".to_string(), serde_json::json!(self.unscoped));
        m.insert("unscoped_pct".to_string(), serde_json::json!(pct(self.unscoped)));
        m
    }

    pub fn increment(&mut self, tier: FallbackTier) {
        match tier {
            FallbackTier::Scoped => self.scoped += 1,
            FallbackTier::Related => self.related += 1,
            FallbackTier::Unscoped => self.unscoped += 1,
        }
    }
}

/// A row returned from a palace semantic search.
pub type PalaceRow = std::collections::HashMap<String, serde_json::Value>;

/// Execute palace search with progressive fallback using a storage backend.
///
/// The `search_fn` closure accepts `(query_embedding, limit, project_ids)` and
/// returns a `Vec<PalaceRow>`.  This keeps `ProgressiveFallback` decoupled from
/// the concrete storage implementation.
pub fn search_with_fallback<F>(
    query_embedding: &[f32],
    project_ids: Option<&[String]>,
    limit: usize,
    min_results: usize,
    mut search_fn: F,
    find_related_fn: Option<&dyn Fn(&[String]) -> Result<Vec<String>, SearchError>>,
    stats: &mut FallbackStats,
) -> Result<(Vec<PalaceRow>, FallbackTier), SearchError>
where
    F: FnMut(&[f32], usize, Option<&[String]>) -> Result<Vec<PalaceRow>, SearchError>,
{
    // Tier 1: scoped search.
    if let Some(pids) = project_ids {
        if !pids.is_empty() {
            let results = search_fn(query_embedding, limit, Some(pids))?;

            if results.len() >= min_results {
                info!(
                    "Scoped search successful: {} results for projects {:?}",
                    results.len(),
                    pids
                );
                stats.increment(FallbackTier::Scoped);
                return Ok((results, FallbackTier::Scoped));
            }

            info!(
                "Scoped search insufficient: {} results (min {}) for projects {:?}",
                results.len(),
                min_results,
                pids
            );

            // Tier 2: related projects.
            if let Some(related_fn) = find_related_fn {
                let related = related_fn(pids)?;
                if !related.is_empty() {
                    let mut expanded = pids.to_vec();
                    expanded.extend_from_slice(&related);
                    let expanded_results =
                        search_fn(query_embedding, limit, Some(&expanded))?;
                    if expanded_results.len() >= min_results {
                        info!(
                            "Related projects expansion successful: {} results (added {:?})",
                            expanded_results.len(),
                            related
                        );
                        stats.increment(FallbackTier::Related);
                        return Ok((expanded_results, FallbackTier::Related));
                    }
                    info!(
                        "Related projects expansion insufficient: {} results",
                        expanded_results.len()
                    );
                }
            }
        }
    }

    // Tier 3: unscoped.
    let unscoped_results = search_fn(query_embedding, limit, None)?;
    if project_ids.map_or(false, |p| !p.is_empty()) {
        warn!(
            "Fell back to unscoped search: {} results (facet resolution was: {:?})",
            unscoped_results.len(),
            project_ids
        );
    } else {
        info!(
            "Unscoped search (no facet resolution): {} results",
            unscoped_results.len()
        );
    }
    stats.increment(FallbackTier::Unscoped);
    Ok((unscoped_results, FallbackTier::Unscoped))
}

/// Stateful wrapper over `search_with_fallback` that accumulates stats across calls.
pub struct ProgressiveFallback {
    pub min_results: usize,
    pub stats: FallbackStats,
}

impl ProgressiveFallback {
    pub fn new(min_results: usize) -> Self {
        Self {
            min_results,
            stats: FallbackStats::default(),
        }
    }

    pub fn get_stats(&self) -> HashMap<String, serde_json::Value> {
        self.stats.to_map()
    }

    pub fn reset_stats(&mut self) {
        self.stats = FallbackStats::default();
    }
}
