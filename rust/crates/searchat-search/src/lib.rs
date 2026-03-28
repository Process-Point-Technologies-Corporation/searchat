//! Unified search engine for searchat.
//!
//! Ports the Python `unified_search.py` implementation to Rust.
//! The engine is generic over storage and embedder backends via traits
//! defined in `storage.rs`, allowing the concrete DuckDB / ONNX
//! implementations (Phase 2) to be injected without changing this crate.

pub mod bridge;
pub mod cache;
pub mod error;
pub mod fallback;
pub mod merger;
pub mod normalize;
pub mod query_classifier;
pub mod query_parser;
pub mod storage;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::Utc;
use log::info;
use searchat_config::settings::{Config, RankingConfig};
use searchat_models::{AlgorithmType, ParsedQuery, SearchFilters, SearchResult, SearchResults};

use crate::cache::SearchCache;
use crate::error::SearchError;
use crate::normalize::{normalize_score, percentile_normalize};
use crate::query_classifier::QueryClassifier;
use crate::query_parser::QueryParser;
use crate::storage::{EmbedderBackend, PalaceRow, StorageBackend, VerbatimRow};

const PERCENTILE: f64 = 95.0;
const MIN_SAMPLES: usize = 10;

/// Unified search engine.
///
/// Generic over `S: StorageBackend` and `E: EmbedderBackend` so the
/// concrete DuckDB / ONNX types can be injected via `Arc<T>`.
pub struct UnifiedSearchEngine<S, E> {
    storage: Arc<S>,
    embedder: Arc<E>,
    config: Config,
    query_parser: QueryParser,
    query_classifier: QueryClassifier,
    cache: SearchCache,
}

impl<S: StorageBackend, E: EmbedderBackend> UnifiedSearchEngine<S, E> {
    /// Create a new engine from shared storage and embedder instances.
    pub fn new(storage: Arc<S>, embedder: Arc<E>, config: Config) -> Self {
        let cache_size = config.performance.query_cache_size;
        let cache_ttl = Duration::from_secs(300); // 5 minutes
        Self {
            storage,
            embedder,
            config,
            query_parser: QueryParser::new(),
            query_classifier: QueryClassifier::new(),
            cache: SearchCache::new(cache_size, cache_ttl),
        }
    }

    // -------------------------------------------------------------------------
    // Public API
    // -------------------------------------------------------------------------

    /// Search conversations with exchange-level granularity.
    pub fn search(
        &self,
        query: &str,
        algorithm: AlgorithmType,
        filters: Option<&SearchFilters>,
        limit: usize,
    ) -> Result<SearchResults, SearchError> {
        let start = Instant::now();
        let cache_key = self.cache_key(query, algorithm, filters, limit);

        if let Some(mut cached) = self.cache.get(&cache_key) {
            cached.search_time_ms = start.elapsed().as_secs_f64() * 1000.0;
            return Ok(cached);
        }

        let results = match algorithm {
            AlgorithmType::Keyword => self.keyword_search(query, filters, limit)?,
            AlgorithmType::Semantic => self.semantic_search(query, filters, limit)?,
            AlgorithmType::Hybrid => self.hybrid_search(query, filters, limit)?,
            AlgorithmType::Adaptive => self.adaptive_hybrid_search(query, filters, limit)?,
            AlgorithmType::CrossLayer => self.cross_layer_search(query, filters, limit)?,
            AlgorithmType::Distill => self.distill_search(query, filters, limit)?,
        };

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        let total = results.len() as i64;
        let search_results = SearchResults {
            results,
            total_count: total,
            search_time_ms: elapsed_ms,
            mode_used: format!("{:?}", algorithm).to_lowercase(),
            error: None,
        };

        self.cache.insert(cache_key, search_results.clone());
        Ok(search_results)
    }

    /// Return storage statistics (pass-through).
    pub fn get_stats(&self) -> Result<HashMap<String, serde_json::Value>, SearchError> {
        self.storage.get_stats()
    }

    // -------------------------------------------------------------------------
    // Search modes
    // -------------------------------------------------------------------------

    /// BM25 keyword search over exchanges.
    fn keyword_search(
        &self,
        query: &str,
        filters: Option<&SearchFilters>,
        limit: usize,
    ) -> Result<Vec<SearchResult>, SearchError> {
        let parsed = self.query_parser.parse(query);
        let project_ids = project_ids(filters);

        let rows = self
            .storage
            .search_verbatim_bm25(query, limit * 2, project_ids.as_deref())?;
        if rows.is_empty() {
            return Ok(vec![]);
        }

        let rows = apply_verbatim_filters(rows, filters, &parsed);
        let deduped = dedup_verbatim_by_conv(rows);
        let results = deduped
            .into_iter()
            .take(limit)
            .map(|r| verbatim_to_search_result(r, query, &parsed, &self.config.search.snippet_length))
            .collect();
        Ok(results)
    }

    /// Semantic (HNSW) search over exchange embeddings.
    fn semantic_search(
        &self,
        query: &str,
        filters: Option<&SearchFilters>,
        limit: usize,
    ) -> Result<Vec<SearchResult>, SearchError> {
        let embedding = self.embedder.encode(query)?;
        let project_ids = project_ids(filters);
        let parsed = self.query_parser.parse(query);

        let rows = self
            .storage
            .search_verbatim_semantic(&embedding, limit * 2, project_ids.as_deref())?;
        if rows.is_empty() {
            return Ok(vec![]);
        }

        let rows = apply_verbatim_filters(rows, filters, &parsed);
        let deduped = dedup_verbatim_by_conv(rows);
        let results = deduped
            .into_iter()
            .take(limit)
            .map(|r| verbatim_to_search_result(r, query, &parsed, &self.config.search.snippet_length))
            .collect();
        Ok(results)
    }

    /// Hybrid search combining BM25 and semantic with configured weights.
    fn hybrid_search(
        &self,
        query: &str,
        filters: Option<&SearchFilters>,
        limit: usize,
    ) -> Result<Vec<SearchResult>, SearchError> {
        let ranking = &self.config.search.ranking;
        self.hybrid_search_with_weights(
            query,
            filters,
            limit,
            ranking.keyword_weight,
            ranking.semantic_weight,
        )
    }

    /// Adaptive hybrid: classify query first, then pick weights dynamically.
    fn adaptive_hybrid_search(
        &self,
        query: &str,
        filters: Option<&SearchFilters>,
        limit: usize,
    ) -> Result<Vec<SearchResult>, SearchError> {
        let classification = self.query_classifier.classify(query);
        info!(
            "Adaptive search: query='{}' classified as {} (confidence={:.2}, reason: {}) → weights=(bm25={:.1}, sem={:.1})",
            &query[..query.len().min(100)],
            classification.query_type.as_str(),
            classification.confidence,
            classification.reasoning,
            classification.bm25_weight,
            classification.semantic_weight,
        );
        self.hybrid_search_with_weights(
            query,
            filters,
            limit,
            classification.bm25_weight,
            classification.semantic_weight,
        )
    }

    /// CombMNZ fusion: BM25 on verbatim exchanges + HNSW on distilled palace objects.
    ///
    /// `score = sum(norm_scores) * count(nonzero_signals)`.
    fn cross_layer_search(
        &self,
        query: &str,
        filters: Option<&SearchFilters>,
        limit: usize,
    ) -> Result<Vec<SearchResult>, SearchError> {
        let project_ids = project_ids(filters);
        let parsed = self.query_parser.parse(query);

        let verbatim_rows = self
            .storage
            .search_verbatim_bm25(query, limit * 2, project_ids.as_deref())?;

        let embedding = self.embedder.encode(query)?;
        let palace_rows = self
            .storage
            .search_palace_semantic(&embedding, limit * 2, project_ids.as_deref())?;

        if verbatim_rows.is_empty() && palace_rows.is_empty() {
            return Ok(vec![]);
        }

        // Normalize verbatim scores.
        let v_raw: Vec<f64> = verbatim_rows.iter().map(|r| r.score).collect();
        let v_divisor = percentile_normalize(&v_raw, PERCENTILE, MIN_SAMPLES);
        let mut v_scores: HashMap<String, f64> = HashMap::new();
        let mut v_data: HashMap<String, VerbatimRow> = HashMap::new();
        for r in &verbatim_rows {
            v_scores.insert(r.exchange_id.clone(), normalize_score(r.score, v_divisor));
            v_data.insert(r.exchange_id.clone(), r.clone());
        }

        // Normalize palace scores.
        let p_raw: Vec<f64> = palace_rows.iter().map(|r| r.score).collect();
        let p_divisor = percentile_normalize(&p_raw, PERCENTILE, MIN_SAMPLES);
        let mut p_scores: HashMap<String, f64> = HashMap::new();
        let mut p_data: HashMap<String, PalaceRow> = HashMap::new();
        for r in &palace_rows {
            p_scores.insert(r.exchange_id.clone(), normalize_score(r.score, p_divisor));
            p_data.insert(r.exchange_id.clone(), r.clone());
        }

        // Enrich palace-only rows with conversation metadata.
        let palace_only: Vec<String> = p_data
            .keys()
            .filter(|eid| !v_data.contains_key(*eid))
            .cloned()
            .collect();
        if !palace_only.is_empty() {
            let conv_ids: Vec<String> = palace_only
                .iter()
                .filter_map(|eid| p_data.get(eid))
                .map(|r| r.conversation_id.clone())
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect();
            let meta = self.storage.get_conversations_batch(&conv_ids)?;
            for eid in &palace_only {
                if let Some(row) = p_data.get_mut(eid) {
                    if let Some(m) = meta.get(&row.conversation_id) {
                        row.title.get_or_insert_with(|| m.title.clone());
                        row.file_path.get_or_insert_with(|| m.file_path.clone());
                        row.message_count.get_or_insert(m.message_count);
                        if row.updated_at.is_none() {
                            row.updated_at = m.updated_at;
                        }
                        if row.created_at.is_none() {
                            row.created_at = m.created_at;
                        }
                    }
                }
            }
        }

        // CombMNZ fusion.
        let all_eids: std::collections::HashSet<String> = v_scores
            .keys()
            .chain(p_scores.keys())
            .cloned()
            .collect();

        let mut combined: Vec<(String, f64)> = all_eids
            .iter()
            .map(|eid| {
                let v = v_scores.get(eid).copied().unwrap_or(0.0);
                let p = p_scores.get(eid).copied().unwrap_or(0.0);
                let nonzero = (if v > 0.0 { 1 } else { 0 }) + (if p > 0.0 { 1 } else { 0 });
                (eid.clone(), (v + p) * nonzero as f64)
            })
            .collect();
        combined.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Deduplicate by conversation_id.
        let mut seen_convs: HashMap<String, (String, f64)> = HashMap::new();
        for (eid, score) in &combined {
            // Prefer verbatim data for metadata.
            let conv_id = v_data
                .get(eid)
                .map(|r| r.conversation_id.clone())
                .or_else(|| p_data.get(eid).map(|r| r.conversation_id.clone()));
            let conv_id = match conv_id {
                Some(id) => id,
                None => continue,
            };

            // Apply date filters.
            if let Some(f) = filters {
                let updated = v_data
                    .get(eid)
                    .and_then(|r| r.updated_at)
                    .or_else(|| p_data.get(eid).and_then(|r| r.updated_at));
                if let Some(dt) = updated {
                    if let Some(from) = f.date_from {
                        if dt < from {
                            continue;
                        }
                    }
                    if let Some(to) = f.date_to {
                        if dt > to {
                            continue;
                        }
                    }
                }
            }

            seen_convs.entry(conv_id).or_insert((eid.clone(), *score));
        }

        let mut sorted_convs: Vec<(String, f64)> = seen_convs
            .into_values()
            .collect();
        sorted_convs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let snippet_len = self.config.search.snippet_length;
        let results = sorted_convs
            .into_iter()
            .take(limit)
            .filter_map(|(eid, score)| {
                let v_row = v_data.get(&eid);
                let p_row = p_data.get(&eid);

                // Build SearchResult from whichever layer has data.
                let mut result = match (v_row, p_row) {
                    (Some(v), _) => {
                        verbatim_to_search_result(v.clone(), query, &parsed, &snippet_len)
                    }
                    (None, Some(p)) => {
                        palace_row_to_search_result(p.clone(), query, &parsed, &snippet_len)
                    }
                    _ => return None,
                };

                result.score = score;
                result.bm25_score = v_scores.get(&eid).copied();
                result.semantic_score = p_scores.get(&eid).copied();

                if let Some(p) = p_row {
                    result.palace_summary = p.exchange_core.clone();
                    result.palace_context = p.specific_context.clone();
                    result.object_id = p.object_id.clone();
                }

                Some(result)
            })
            .collect();

        Ok(results)
    }

    /// Pure semantic search on distilled palace objects.
    fn distill_search(
        &self,
        query: &str,
        filters: Option<&SearchFilters>,
        limit: usize,
    ) -> Result<Vec<SearchResult>, SearchError> {
        let project_ids = project_ids(filters);
        let embedding = self.embedder.encode(query)?;
        let parsed = self.query_parser.parse(query);

        let palace_rows = self
            .storage
            .search_palace_semantic(&embedding, limit * 2, project_ids.as_deref())?;

        if palace_rows.is_empty() {
            return Ok(vec![]);
        }

        // Deduplicate by conversation_id before enriching.
        let mut seen: HashMap<String, usize> = HashMap::new(); // conv_id → index
        let mut deduped: Vec<PalaceRow> = Vec::new();
        for row in palace_rows {
            if let Some(&idx) = seen.get(&row.conversation_id) {
                if row.score > deduped[idx].score {
                    deduped[idx] = row;
                }
            } else {
                seen.insert(row.conversation_id.clone(), deduped.len());
                deduped.push(row);
            }
        }
        deduped.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        deduped.truncate(limit);

        // Batch-enrich with conversation metadata.
        let conv_ids: Vec<String> = deduped
            .iter()
            .map(|r| r.conversation_id.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        let meta = self.storage.get_conversations_batch(&conv_ids)?;
        for row in &mut deduped {
            if let Some(m) = meta.get(&row.conversation_id) {
                row.title.get_or_insert_with(|| m.title.clone());
                row.file_path.get_or_insert_with(|| m.file_path.clone());
                row.message_count.get_or_insert(m.message_count);
                if row.updated_at.is_none() {
                    row.updated_at = m.updated_at;
                }
                if row.created_at.is_none() {
                    row.created_at = m.created_at;
                }
            }
        }

        // Apply date filters after enrichment.
        let filtered: Vec<PalaceRow> = deduped
            .into_iter()
            .filter(|r| {
                if let Some(f) = filters {
                    if let Some(from) = f.date_from {
                        if let Some(dt) = r.updated_at {
                            if dt < from {
                                return false;
                            }
                        }
                    }
                    if let Some(to) = f.date_to {
                        if let Some(dt) = r.updated_at {
                            if dt > to {
                                return false;
                            }
                        }
                    }
                }
                true
            })
            .collect();

        let snippet_len = self.config.search.snippet_length;
        let results = filtered
            .into_iter()
            .map(|r| {
                let score = r.score;
                let exchange_core = r.exchange_core.clone();
                let specific_context = r.specific_context.clone();
                let object_id = r.object_id.clone();
                let mut result =
                    palace_row_to_search_result(r, query, &parsed, &snippet_len);
                result.semantic_score = Some(score);
                result.palace_summary = exchange_core;
                result.palace_context = specific_context;
                result.object_id = object_id;
                result
            })
            .collect();

        Ok(results)
    }

    // -------------------------------------------------------------------------
    // Shared hybrid merge helper
    // -------------------------------------------------------------------------

    fn hybrid_search_with_weights(
        &self,
        query: &str,
        filters: Option<&SearchFilters>,
        limit: usize,
        keyword_weight: f64,
        semantic_weight: f64,
    ) -> Result<Vec<SearchResult>, SearchError> {
        let parsed = self.query_parser.parse(query);
        let project_ids = project_ids(filters);

        let keyword_rows = self
            .storage
            .search_verbatim_bm25(query, limit * 2, project_ids.as_deref())?;
        let embedding = self.embedder.encode(query)?;
        let semantic_rows = self
            .storage
            .search_verbatim_semantic(&embedding, limit * 2, project_ids.as_deref())?;

        let keyword_rows = apply_verbatim_filters(keyword_rows, filters, &parsed);
        let semantic_rows = apply_verbatim_filters(semantic_rows, filters, &parsed);

        let ranking = &self.config.search.ranking;
        Ok(merge_verbatim_results(
            keyword_rows,
            semantic_rows,
            query,
            limit,
            keyword_weight,
            semantic_weight,
            ranking,
            self.config.search.snippet_length,
        ))
    }

    // -------------------------------------------------------------------------
    // Cache key generation
    // -------------------------------------------------------------------------

    fn cache_key(
        &self,
        query: &str,
        algorithm: AlgorithmType,
        filters: Option<&SearchFilters>,
        limit: usize,
    ) -> String {
        use std::fmt::Write;
        let mut key = format!("{}|{:?}|limit:{}", query, algorithm, limit);
        if let Some(f) = filters {
            if let Some(ref pids) = f.project_ids {
                let _ = write!(key, "|projects:{}", pids.join(","));
            }
            if let Some(from) = f.date_from {
                let _ = write!(key, "|from:{}", from.timestamp());
            }
            if let Some(to) = f.date_to {
                let _ = write!(key, "|to:{}", to.timestamp());
            }
            if f.min_messages > 0 {
                let _ = write!(key, "|min_msgs:{}", f.min_messages);
            }
        }
        // MD5-like fingerprint via sha2 would be better, but using a plain string
        // key is fine for LRU purposes — just keep it bounded.
        key
    }
}

// =============================================================================
// Free functions
// =============================================================================

/// Extract optional project_ids from filters.
fn project_ids(filters: Option<&SearchFilters>) -> Option<Vec<String>> {
    filters.and_then(|f| f.project_ids.clone())
}

/// Filter verbatim rows by date, exact phrases, and must-exclude terms.
fn apply_verbatim_filters(
    rows: Vec<VerbatimRow>,
    filters: Option<&SearchFilters>,
    parsed: &ParsedQuery,
) -> Vec<VerbatimRow> {
    if filters.is_none() && parsed.exact_phrases.is_empty() && parsed.must_exclude.is_empty() {
        return rows;
    }

    rows.into_iter()
        .filter(|r| {
            if let Some(f) = filters {
                if let Some(from) = f.date_from {
                    if let Some(dt) = r.updated_at {
                        if dt < from {
                            return false;
                        }
                    }
                }
                if let Some(to) = f.date_to {
                    if let Some(dt) = r.updated_at {
                        if dt > to {
                            return false;
                        }
                    }
                }
                if f.min_messages > 0 {
                    if r.message_count.unwrap_or(0) < f.min_messages {
                        return false;
                    }
                }
            }

            let text_lower = r.exchange_text.as_deref().unwrap_or("").to_lowercase();

            if !parsed.exact_phrases.is_empty() {
                if !parsed
                    .exact_phrases
                    .iter()
                    .all(|ph| text_lower.contains(&ph.to_lowercase()))
                {
                    return false;
                }
            }

            if !parsed.must_exclude.is_empty() {
                if parsed
                    .must_exclude
                    .iter()
                    .any(|t| text_lower.contains(&t.to_lowercase()))
                {
                    return false;
                }
            }

            true
        })
        .collect()
}

/// Keep best exchange per conversation, sorted descending by score.
fn dedup_verbatim_by_conv(rows: Vec<VerbatimRow>) -> Vec<VerbatimRow> {
    let mut best: HashMap<String, VerbatimRow> = HashMap::new();
    for r in rows {
        let entry = best.entry(r.conversation_id.clone()).or_insert_with(|| r.clone());
        if r.score > entry.score {
            *entry = r;
        }
    }
    let mut v: Vec<VerbatimRow> = best.into_values().collect();
    v.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    v
}

/// Merge BM25 and semantic verbatim results with weighted score fusion.
///
/// Mirrors `_merge_results_with_weights` in the Python implementation.
#[allow(clippy::too_many_arguments)]
fn merge_verbatim_results(
    keyword_rows: Vec<VerbatimRow>,
    semantic_rows: Vec<VerbatimRow>,
    query: &str,
    limit: usize,
    keyword_weight: f64,
    semantic_weight: f64,
    ranking: &RankingConfig,
    snippet_length: usize,
) -> Vec<SearchResult> {
    let intersection_boost = ranking.boost_multiplier();
    let rank_decay = ranking.rank_decay;
    let title_boost = ranking.title_boost;

    // Normalize and score keyword rows.
    let mut kw_scores: HashMap<String, f64> = HashMap::new();
    let mut exchange_data: HashMap<String, VerbatimRow> = HashMap::new();

    if !keyword_rows.is_empty() {
        let raw: Vec<f64> = keyword_rows.iter().map(|r| r.score).collect();
        let divisor = percentile_normalize(&raw, PERCENTILE, MIN_SAMPLES);
        for (rank, r) in keyword_rows.iter().enumerate() {
            let norm = normalize_score(r.score, divisor);
            let rank_w = 1.0 / (1.0 + rank_decay * (rank + 1) as f64);
            kw_scores.insert(r.exchange_id.clone(), norm * rank_w);
            exchange_data.insert(r.exchange_id.clone(), r.clone());
        }
    }

    let mut sem_scores: HashMap<String, f64> = HashMap::new();

    if !semantic_rows.is_empty() {
        let raw: Vec<f64> = semantic_rows.iter().map(|r| r.score).collect();
        let divisor = percentile_normalize(&raw, PERCENTILE, MIN_SAMPLES);
        for (rank, r) in semantic_rows.iter().enumerate() {
            let norm = normalize_score(r.score, divisor);
            let rank_w = 1.0 / (1.0 + rank_decay * (rank + 1) as f64);
            sem_scores.insert(r.exchange_id.clone(), norm * rank_w);
            exchange_data.entry(r.exchange_id.clone()).or_insert_with(|| r.clone());
        }
    }

    // Combine scores for every exchange ID seen.
    let all_eids: std::collections::HashSet<String> = kw_scores
        .keys()
        .chain(sem_scores.keys())
        .cloned()
        .collect();

    let query_terms: Vec<String> = query
        .to_lowercase()
        .split_whitespace()
        .map(|t| t.to_string())
        .collect();

    // Combined: (exchange_id, combined_score, kw_score, sem_score)
    let combined: Vec<(String, f64, f64, f64)> = all_eids
        .iter()
        .map(|eid| {
            let kw = kw_scores.get(eid).copied().unwrap_or(0.0);
            let sem = sem_scores.get(eid).copied().unwrap_or(0.0);
            let mut score = keyword_weight * kw + semantic_weight * sem;

            // Intersection boost.
            if kw > 0.0 && sem > 0.0 {
                score *= intersection_boost;
            }

            // Title boost.
            if let Some(row) = exchange_data.get(eid) {
                let title = row.title.as_deref().unwrap_or("").to_lowercase();
                if query_terms.iter().any(|t| title.contains(t.as_str())) {
                    score *= title_boost;
                }
            }

            (eid.clone(), score, kw, sem)
        })
        .collect();

    // Deduplicate by conversation, keeping best exchange.
    let mut conv_best: HashMap<String, (String, f64, f64, f64)> = HashMap::new();
    for (eid, score, kw, sem) in combined {
        if let Some(row) = exchange_data.get(&eid) {
            let conv_id = row.conversation_id.clone();
            let entry = conv_best.entry(conv_id).or_insert((eid.clone(), score, kw, sem));
            if score > entry.1 {
                *entry = (eid, score, kw, sem);
            }
        }
    }

    let mut sorted: Vec<(String, f64, f64, f64)> = conv_best.into_values().collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let parsed_dummy = ParsedQuery {
        original: query.to_string(),
        must_include: vec![],
        should_include: query.split_whitespace().map(|t| t.to_string()).collect(),
        must_exclude: vec![],
        exact_phrases: vec![],
        date_filter: None,
    };

    sorted
        .into_iter()
        .take(limit)
        .filter_map(|(eid, score, kw, sem)| {
            let row = exchange_data.remove(&eid)?;
            let mut result = verbatim_to_search_result(row, query, &parsed_dummy, &snippet_length);
            result.score = score;
            result.bm25_score = if kw > 0.0 { Some(kw) } else { None };
            result.semantic_score = if sem > 0.0 { Some(sem) } else { None };
            Some(result)
        })
        .collect()
}

// =============================================================================
// Row → domain model converters
// =============================================================================

fn verbatim_to_search_result(
    row: VerbatimRow,
    _query: &str,
    parsed: &ParsedQuery,
    snippet_length: &usize,
) -> SearchResult {
    let text_owned = row.exchange_text.unwrap_or_default();
    let snippet = create_snippet(&text_owned, parsed, *snippet_length);

    SearchResult {
        conversation_id: row.conversation_id,
        project_id: row.project_id,
        title: row.title.unwrap_or_else(|| "Untitled".to_string()),
        created_at: row.created_at.unwrap_or_else(Utc::now),
        updated_at: row.updated_at.unwrap_or_else(Utc::now),
        message_count: row.message_count.unwrap_or(0),
        file_path: row.file_path.unwrap_or_default(),
        score: row.score,
        snippet,
        message_start_index: row.ply_start,
        message_end_index: row.ply_end,
        bm25_score: None,
        semantic_score: None,
        exchange_id: Some(row.exchange_id),
        exchange_text: Some(text_owned),
        match_source: Some("unified".to_string()),
        palace_summary: None,
        palace_context: None,
        files_touched_raw: None,
        object_id: None,
        search_metadata: None,
    }
}

fn palace_row_to_search_result(
    row: PalaceRow,
    _query: &str,
    parsed: &ParsedQuery,
    snippet_length: &usize,
) -> SearchResult {
    // Extract fields before consuming row.
    let text_owned = row.exchange_core.clone().unwrap_or_default();
    let snippet = create_snippet(&text_owned, parsed, *snippet_length);

    SearchResult {
        conversation_id: row.conversation_id,
        project_id: row.project_id,
        title: row.title.unwrap_or_else(|| "Untitled".to_string()),
        created_at: row.created_at.unwrap_or_else(Utc::now),
        updated_at: row.updated_at.unwrap_or_else(Utc::now),
        message_count: row.message_count.unwrap_or(0),
        file_path: row.file_path.unwrap_or_default(),
        score: row.score,
        snippet,
        message_start_index: row.ply_start,
        message_end_index: row.ply_end,
        bm25_score: None,
        semantic_score: None,
        exchange_id: Some(row.exchange_id),
        exchange_text: Some(text_owned.clone()),
        match_source: Some("palace".to_string()),
        palace_summary: Some(text_owned),
        palace_context: row.specific_context,
        files_touched_raw: None,
        object_id: row.object_id,
        search_metadata: None,
    }
}

/// Create a query-centred snippet from text, length `snippet_length` chars.
fn create_snippet(text: &str, parsed: &ParsedQuery, length: usize) -> String {
    if text.is_empty() {
        return String::new();
    }

    let text_lower = text.to_lowercase();
    let mut best_pos: Option<usize> = None;

    // Try exact phrases first.
    for phrase in &parsed.exact_phrases {
        if let Some(pos) = text_lower.find(&phrase.to_lowercase()) {
            best_pos = Some(pos);
            break;
        }
    }

    // Try must_include and should_include terms.
    if best_pos.is_none() {
        for term in parsed.must_include.iter().chain(parsed.should_include.iter()) {
            if let Some(pos) = text_lower.find(&term.to_lowercase()) {
                best_pos = Some(pos);
                break;
            }
        }
    }

    match best_pos {
        None => {
            let end = length.min(text.len());
            let mut s = text[..end].to_string();
            if text.len() > length {
                s.push_str("...");
            }
            s
        }
        Some(pos) => {
            let start = pos.saturating_sub(length / 2);
            let end = (pos + length).min(text.len());
            let mut s = text[start..end].to_string();
            if start > 0 {
                s = format!("...{}", s);
            }
            if end < text.len() {
                s.push_str("...");
            }
            s
        }
    }
}
