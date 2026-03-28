use regex::Regex;

/// Result of classifying a query into keyword/semantic/balanced categories.
#[derive(Debug, Clone)]
pub struct QueryClassification {
    pub query_type: QueryType,
    pub bm25_weight: f64,
    pub semantic_weight: f64,
    /// Confidence in classification (0–1).
    pub confidence: f64,
    pub reasoning: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryType {
    Keyword,
    Semantic,
    Balanced,
}

impl QueryType {
    pub fn as_str(&self) -> &'static str {
        match self {
            QueryType::Keyword => "keyword",
            QueryType::Semantic => "semantic",
            QueryType::Balanced => "balanced",
        }
    }
}

/// Patterns that indicate keyword-heavy queries (file paths, identifiers, code constructs).
const KEYWORD_PATTERN_STRS: &[&str] = &[
    r"\.\w+:\d+",                                // file.py:142
    r"\b\w+\.(py|ts|js|tsx|jsx|java|cpp|c|h|go|rs|rb|php|md|txt|json|toml|yaml|yml|sql|sh|bat|ps1|css|html)\b",
    r"\bline\s+\d+",                             // "line 42"
    r"error\s+code\s+\d+",                       // "error code 500"
    r"\b[A-Z_]{5,}\b",                           // CONSTANT_NAMES
    r"\bfunction\s+\w+",                         // "function foo"
    r"\bclass\s+\w+",                            // "class Bar"
    r"\bdef\s+\w+",                              // "def baz"
    r"\bimport\s+[\w.]+",                        // "import xyz"
    r"\bfrom\s+[\w.]+\s+import",                 // "from abc import"
    r"\bTODO\b|\bFIXME\b|\bNOTE\b|\bBUG\b",
    r"[/_]\w+[/_]",                              // path separators
];

/// Terms that indicate semantic/conceptual queries.
const SEMANTIC_TERMS: &[&str] = &[
    "architecture",
    "pattern",
    "design",
    "approach",
    "strategy",
    "concept",
    "principle",
    "methodology",
    "framework",
    "paradigm",
    "philosophy",
    "rationale",
    "tradeoff",
    "pros and cons",
    "advantages",
    "disadvantages",
    "best practice",
    "anti-pattern",
    "recommendation",
    "guideline",
    "overview",
    "introduction",
    "explanation",
    "understanding",
    "why",
    "how does",
    "what is",
    "when to",
    "should i",
    "comparison",
    "difference between",
    "versus",
    "vs",
    "alternative",
    "option",
    "choice",
    "decide",
];

/// Question-word prefixes that are a strong semantic signal.
const QUESTION_START_PREFIXES: &[&str] =
    &["how ", "why ", "what ", "when ", "where ", "should ", "can "];

/// Question words anywhere in query (weaker signal than start).
const QUESTION_WORDS: &[&str] = &["how ", "why ", "what ", "when ", "where "];

pub struct QueryClassifier {
    keyword_regex: Regex,
}

impl QueryClassifier {
    pub fn new() -> Self {
        let combined = KEYWORD_PATTERN_STRS
            .iter()
            .map(|p| format!("({})", p))
            .collect::<Vec<_>>()
            .join("|");
        let keyword_regex = Regex::new(&combined).expect("keyword regex");
        Self { keyword_regex }
    }

    pub fn classify(&self, query: &str) -> QueryClassification {
        let query_lower = query.to_lowercase();

        // Count keyword pattern matches (each capture group is one match).
        let keyword_matches = self.keyword_regex.find_iter(query).count();

        // Count semantic term matches.
        let semantic_matches = SEMANTIC_TERMS
            .iter()
            .filter(|t| query_lower.contains(**t))
            .count();

        // File path indicators.
        let has_quotes = query.contains('"') || query.contains('\'');
        let has_path =
            query.contains('/') || query.contains('\\') || query.contains('.');

        // Build scores.
        let mut keyword_score = keyword_matches as f64 * 2.5;
        if has_quotes {
            keyword_score += 2.0;
        }
        let architectural = ["architecture", "design", "pattern"];
        if has_path && !architectural.iter().any(|t| query_lower.contains(t)) {
            keyword_score += 1.5;
        }

        let mut semantic_score = semantic_matches as f64 * 2.0;

        // Short queries with file indicators lean keyword.
        if query.len() < 15 && (has_path || keyword_matches > 0) {
            keyword_score += 1.0;
        }

        // Long queries with question words lean semantic.
        if query.len() > 80
            && QUESTION_WORDS.iter().any(|q| query_lower.contains(q))
        {
            semantic_score += 2.0;
        }

        let has_question_start = QUESTION_START_PREFIXES
            .iter()
            .any(|p| query_lower.starts_with(p));
        let has_question_word = QUESTION_WORDS.iter().any(|q| query_lower.contains(q));

        if has_question_start {
            semantic_score += 4.0;
        } else if has_question_word {
            semantic_score += 2.0;
        }

        let total_score = keyword_score + semantic_score;

        if total_score == 0.0 {
            QueryClassification {
                query_type: QueryType::Balanced,
                bm25_weight: 0.5,
                semantic_weight: 0.5,
                confidence: 0.3,
                reasoning: "No clear keyword or semantic signals detected".to_string(),
            }
        } else if keyword_score > semantic_score * 2.0 && !has_question_start {
            let confidence = (keyword_score / (total_score + 1.0)).min(0.9);
            QueryClassification {
                query_type: QueryType::Keyword,
                bm25_weight: 0.8,
                semantic_weight: 0.2,
                confidence,
                reasoning: format!(
                    "File paths/identifiers/code patterns detected ({} matches)",
                    keyword_matches
                ),
            }
        } else if semantic_score > keyword_score * 1.5 || has_question_start {
            let confidence = (semantic_score / (total_score + 1.0)).min(0.9);
            let reasoning = if semantic_matches > 0 {
                format!(
                    "Conceptual/architectural terms detected ({} matches)",
                    semantic_matches
                )
            } else {
                "Question query detected".to_string()
            };
            QueryClassification {
                query_type: QueryType::Semantic,
                bm25_weight: 0.2,
                semantic_weight: 0.8,
                confidence,
                reasoning,
            }
        } else {
            QueryClassification {
                query_type: QueryType::Balanced,
                bm25_weight: 0.5,
                semantic_weight: 0.5,
                confidence: 0.5,
                reasoning: format!(
                    "Mixed signals: {} keyword patterns, {} semantic terms",
                    keyword_matches, semantic_matches
                ),
            }
        }
    }
}

impl Default for QueryClassifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn question_start_is_semantic() {
        let c = QueryClassifier::new();
        let cl = c.classify("how does the indexer work");
        assert_eq!(cl.query_type, QueryType::Semantic);
        assert!(cl.semantic_weight > cl.bm25_weight);
    }

    #[test]
    fn file_extension_is_keyword() {
        let c = QueryClassifier::new();
        let cl = c.classify("unified_storage.py:300 error");
        assert_eq!(cl.query_type, QueryType::Keyword);
        assert!(cl.bm25_weight > cl.semantic_weight);
    }

    #[test]
    fn no_signals_is_balanced() {
        let c = QueryClassifier::new();
        let cl = c.classify("foo bar");
        // "foo bar" has no clear signals — could be balanced
        // (it won't have keyword patterns or semantic terms)
        assert_eq!(cl.query_type, QueryType::Balanced);
    }
}
