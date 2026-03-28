/// Merge palace (Layer 2) and verbatim (Layer 1) search results by conversation_id
/// using intersection boost and weighted score combination.
///
/// This mirrors the Python `result_merger.merge_results` function.
use searchat_config::settings::RankingConfig;
use searchat_models::{PalaceSearchResult, SearchResult, UnifiedSearchResult};

use crate::normalize::{normalize_score, percentile_normalize};

const PERCENTILE: f64 = 95.0;
const MIN_SAMPLES: usize = 10;

pub fn merge_results(
    palace_results: &[PalaceSearchResult],
    verbatim_results: &[SearchResult],
    ranking: &RankingConfig,
) -> Vec<UnifiedSearchResult> {
    use std::collections::HashMap;

    let palace_weight = ranking.scaled_palace_weight();
    let verbatim_weight = ranking.scaled_verbatim_weight();
    let boost_multiplier = ranking.boost_multiplier();

    // Normalize palace scores.
    let palace_raw: Vec<f64> = palace_results.iter().map(|r| r.score).collect();
    let palace_divisor = percentile_normalize(&palace_raw, PERCENTILE, MIN_SAMPLES);

    let mut by_conv: HashMap<String, UnifiedSearchResult> = HashMap::new();

    for p in palace_results {
        let norm_score = normalize_score(p.score, palace_divisor);
        by_conv.entry(p.conversation_id.clone()).or_insert_with(|| {
            UnifiedSearchResult {
                conversation_id: p.conversation_id.clone(),
                project_id: p.project_id.clone(),
                // Title / metadata filled by caller via enrichment; use exchange_core as fallback.
                title: truncate(&p.exchange_core, 50),
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
                message_count: 0,
                file_path: String::new(),
                combined_score: norm_score * palace_weight,
                palace_score: Some(p.score),
                palace_summary: Some(p.exchange_core.clone()),
                palace_context: Some(p.specific_context.clone()),
                rooms: p.rooms.clone(),
                files_touched: p.files_touched.clone(),
                ply_start: Some(p.ply_start),
                ply_end: Some(p.ply_end),
                object_id: Some(p.object_id.clone()),
                palace_bm25_score: Some(p.keyword_score),
                palace_semantic_score: Some(p.semantic_score),
                // Verbatim fields empty until merged below.
                verbatim_score: None,
                verbatim_snippet: None,
                message_start_index: None,
                message_end_index: None,
                verbatim_bm25_score: None,
                verbatim_semantic_score: None,
                fallback_tier: None,
            }
        });
    }

    // Normalize verbatim scores.
    let verbatim_raw: Vec<f64> = verbatim_results.iter().map(|r| r.score).collect();
    let verbatim_divisor = percentile_normalize(&verbatim_raw, PERCENTILE, MIN_SAMPLES);

    for v in verbatim_results {
        let norm_score = normalize_score(v.score, verbatim_divisor);

        if let Some(existing) = by_conv.get_mut(&v.conversation_id) {
            // Intersection: combine scores with boost.
            existing.verbatim_score = Some(v.score);
            existing.verbatim_snippet = Some(v.snippet.clone());
            existing.message_start_index = v.message_start_index;
            existing.message_end_index = v.message_end_index;
            existing.verbatim_bm25_score = v.bm25_score;
            existing.verbatim_semantic_score = v.semantic_score;

            let palace_norm = existing
                .palace_score
                .map(|s| normalize_score(s, palace_divisor))
                .unwrap_or(0.0);
            existing.combined_score =
                (palace_weight * palace_norm + verbatim_weight * norm_score) * boost_multiplier;
        } else {
            // Verbatim-only result.
            by_conv.insert(
                v.conversation_id.clone(),
                UnifiedSearchResult {
                    conversation_id: v.conversation_id.clone(),
                    project_id: v.project_id.clone(),
                    title: v.title.clone(),
                    created_at: v.created_at,
                    updated_at: v.updated_at,
                    message_count: v.message_count,
                    file_path: v.file_path.clone(),
                    combined_score: norm_score * verbatim_weight,
                    palace_score: None,
                    palace_summary: None,
                    palace_context: None,
                    rooms: Vec::new(),
                    files_touched: Vec::new(),
                    ply_start: None,
                    ply_end: None,
                    object_id: None,
                    palace_bm25_score: None,
                    palace_semantic_score: None,
                    verbatim_score: Some(v.score),
                    verbatim_snippet: Some(v.snippet.clone()),
                    message_start_index: v.message_start_index,
                    message_end_index: v.message_end_index,
                    verbatim_bm25_score: v.bm25_score,
                    verbatim_semantic_score: v.semantic_score,
                    fallback_tier: None,
                },
            );
        }
    }

    let mut results: Vec<UnifiedSearchResult> = by_conv.into_values().collect();
    results.sort_by(|a, b| {
        b.combined_score
            .partial_cmp(&a.combined_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results
}

/// Merge with soft scoping: verbatim results in resolved_project_ids receive a boost
/// multiplier, but out-of-scope results are not excluded.
pub fn merge_results_with_scoping(
    palace_results: &[PalaceSearchResult],
    verbatim_results: &[SearchResult],
    ranking: &RankingConfig,
    resolved_project_ids: Option<&[String]>,
    verbatim_boost: f64,
) -> (Vec<UnifiedSearchResult>, ScopingStats) {
    use std::collections::{HashMap, HashSet};

    let palace_weight = ranking.scaled_palace_weight();
    let verbatim_weight = ranking.scaled_verbatim_weight();
    let boost_multiplier = ranking.boost_multiplier();

    let palace_raw: Vec<f64> = palace_results.iter().map(|r| r.score).collect();
    let palace_divisor = percentile_normalize(&palace_raw, PERCENTILE, MIN_SAMPLES);

    let mut by_conv: HashMap<String, UnifiedSearchResult> = HashMap::new();

    for p in palace_results {
        let norm_score = normalize_score(p.score, palace_divisor);
        by_conv.entry(p.conversation_id.clone()).or_insert_with(|| {
            UnifiedSearchResult {
                conversation_id: p.conversation_id.clone(),
                project_id: p.project_id.clone(),
                title: truncate(&p.exchange_core, 50),
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
                message_count: 0,
                file_path: String::new(),
                combined_score: norm_score * palace_weight,
                palace_score: Some(p.score),
                palace_summary: Some(p.exchange_core.clone()),
                palace_context: Some(p.specific_context.clone()),
                rooms: p.rooms.clone(),
                files_touched: p.files_touched.clone(),
                ply_start: Some(p.ply_start),
                ply_end: Some(p.ply_end),
                object_id: Some(p.object_id.clone()),
                palace_bm25_score: Some(p.keyword_score),
                palace_semantic_score: Some(p.semantic_score),
                verbatim_score: None,
                verbatim_snippet: None,
                message_start_index: None,
                message_end_index: None,
                verbatim_bm25_score: None,
                verbatim_semantic_score: None,
                fallback_tier: None,
            }
        });
    }

    let verbatim_raw: Vec<f64> = verbatim_results.iter().map(|r| r.score).collect();
    let verbatim_divisor = percentile_normalize(&verbatim_raw, PERCENTILE, MIN_SAMPLES);

    let resolved_set: HashSet<&str> = resolved_project_ids
        .unwrap_or(&[])
        .iter()
        .map(|s| s.as_str())
        .collect();

    let mut stats = ScopingStats::default();

    for v in verbatim_results {
        let norm_score = normalize_score(v.score, verbatim_divisor);
        let is_scoped = !resolved_set.is_empty() && resolved_set.contains(v.project_id.as_str());
        let scoping_multiplier = if is_scoped { verbatim_boost } else { 1.0 };

        if is_scoped {
            stats.boosted += 1;
        } else {
            stats.not_boosted += 1;
        }

        if let Some(existing) = by_conv.get_mut(&v.conversation_id) {
            existing.verbatim_score = Some(v.score);
            existing.verbatim_snippet = Some(v.snippet.clone());
            existing.message_start_index = v.message_start_index;
            existing.message_end_index = v.message_end_index;
            existing.verbatim_bm25_score = v.bm25_score;
            existing.verbatim_semantic_score = v.semantic_score;

            let palace_norm = existing
                .palace_score
                .map(|s| normalize_score(s, palace_divisor))
                .unwrap_or(0.0);
            existing.combined_score = (palace_weight * palace_norm
                + verbatim_weight * norm_score * scoping_multiplier)
                * boost_multiplier;
        } else {
            by_conv.insert(
                v.conversation_id.clone(),
                UnifiedSearchResult {
                    conversation_id: v.conversation_id.clone(),
                    project_id: v.project_id.clone(),
                    title: v.title.clone(),
                    created_at: v.created_at,
                    updated_at: v.updated_at,
                    message_count: v.message_count,
                    file_path: v.file_path.clone(),
                    combined_score: norm_score * scoping_multiplier * verbatim_weight,
                    palace_score: None,
                    palace_summary: None,
                    palace_context: None,
                    rooms: Vec::new(),
                    files_touched: Vec::new(),
                    ply_start: None,
                    ply_end: None,
                    object_id: None,
                    palace_bm25_score: None,
                    palace_semantic_score: None,
                    verbatim_score: Some(v.score),
                    verbatim_snippet: Some(v.snippet.clone()),
                    message_start_index: v.message_start_index,
                    message_end_index: v.message_end_index,
                    verbatim_bm25_score: v.bm25_score,
                    verbatim_semantic_score: v.semantic_score,
                    fallback_tier: None,
                },
            );
        }
    }

    let mut results: Vec<UnifiedSearchResult> = by_conv.into_values().collect();
    results.sort_by(|a, b| {
        b.combined_score
            .partial_cmp(&a.combined_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    (results, stats)
}

#[derive(Debug, Default)]
pub struct ScopingStats {
    pub boosted: usize,
    pub not_boosted: usize,
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() > max {
        format!("{}...", &s[..max])
    } else {
        s.to_string()
    }
}
