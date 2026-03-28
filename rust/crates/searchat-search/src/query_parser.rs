use chrono::{Duration, Utc};
use regex::Regex;
use searchat_models::{DateFilter, ParsedQuery};

/// Parses a raw query string into structured components:
/// exact phrases, must-include, must-exclude, should-include, and optional date filter.
pub struct QueryParser {
    quoted_double: Regex,
    quoted_single: Regex,
    must_include_re: Regex,
    must_exclude_re: Regex,
}

impl QueryParser {
    pub fn new() -> Self {
        Self {
            quoted_double: Regex::new(r#""([^"]+)""#).expect("double-quote regex"),
            quoted_single: Regex::new(r#"'([^']+)'"#).expect("single-quote regex"),
            must_include_re: Regex::new(r"\+(\w+)").expect("must-include regex"),
            must_exclude_re: Regex::new(r"-(\w+)").expect("must-exclude regex"),
        }
    }

    pub fn parse(&self, query: &str) -> ParsedQuery {
        let mut result = ParsedQuery {
            original: query.to_string(),
            must_include: Vec::new(),
            should_include: Vec::new(),
            must_exclude: Vec::new(),
            exact_phrases: Vec::new(),
            date_filter: None,
        };

        // Extract quoted exact phrases (both single and double quotes).
        for cap in self.quoted_double.captures_iter(query) {
            if let Some(m) = cap.get(1) {
                result.exact_phrases.push(m.as_str().to_string());
            }
        }
        for cap in self.quoted_single.captures_iter(query) {
            if let Some(m) = cap.get(1) {
                result.exact_phrases.push(m.as_str().to_string());
            }
        }

        // Strip quotes from the working copy.
        let q = self.quoted_double.replace_all(query, "");
        let q = self.quoted_single.replace_all(&q, "");
        let q = q.as_ref();

        // Extract +term (must include).
        for cap in self.must_include_re.captures_iter(q) {
            if let Some(m) = cap.get(1) {
                result.must_include.push(m.as_str().to_string());
            }
        }
        let q = self.must_include_re.replace_all(q, "");
        let q = q.as_ref();

        // Extract -term (must exclude).
        for cap in self.must_exclude_re.captures_iter(q) {
            if let Some(m) = cap.get(1) {
                result.must_exclude.push(m.as_str().to_string());
            }
        }
        let q = self.must_exclude_re.replace_all(q, "");
        let q = q.as_ref();

        // Extract date filter and strip date terms.
        result.date_filter = extract_date_filter(q);
        let q = remove_date_terms(q);
        let q = q.trim().to_string();

        // Boolean operators.
        let q_upper = q.to_uppercase();
        if q_upper.contains(" AND ") {
            let terms = split_bool(&q, " AND ");
            result.must_include.extend(terms);
        } else if q_upper.contains(" OR ") {
            let terms = split_bool(&q, " OR ");
            result.should_include = terms;
        } else {
            result.should_include = q
                .split_whitespace()
                .filter(|t| !t.is_empty())
                .map(|t| t.to_string())
                .collect();
        }

        result
    }
}

impl Default for QueryParser {
    fn default() -> Self {
        Self::new()
    }
}

fn split_bool(q: &str, sep: &str) -> Vec<String> {
    // Case-insensitive split on the separator.
    let sep_lower = sep.to_lowercase();
    let q_lower = q.to_lowercase();
    let mut parts = Vec::new();
    let mut last = 0;
    let sep_len = sep.len();
    while let Some(pos) = q_lower[last..].find(&sep_lower) {
        let abs = last + pos;
        let part = q[last..abs].trim().to_string();
        if !part.is_empty() {
            parts.push(part);
        }
        last = abs + sep_len;
    }
    let tail = q[last..].trim().to_string();
    if !tail.is_empty() {
        parts.push(tail);
    }
    parts
}

fn extract_date_filter(query: &str) -> Option<DateFilter> {
    let q = query.to_lowercase();
    let now = Utc::now();

    if q.contains("today") {
        let start = now
            .date_naive()
            .and_hms_opt(0, 0, 0)
            .map(|dt| dt.and_utc())?;
        Some(DateFilter {
            from_date: Some(start),
            to_date: Some(now),
        })
    } else if q.contains("last week") || q.contains("last 7 days") {
        Some(DateFilter {
            from_date: Some(now - Duration::days(7)),
            to_date: Some(now),
        })
    } else if q.contains("last 30 days") || q.contains("last month") {
        Some(DateFilter {
            from_date: Some(now - Duration::days(30)),
            to_date: Some(now),
        })
    } else if q.contains("last 3 months") {
        Some(DateFilter {
            from_date: Some(now - Duration::days(90)),
            to_date: Some(now),
        })
    } else {
        None
    }
}

fn remove_date_terms(query: &str) -> String {
    let date_terms = [
        "today",
        "last week",
        "last 7 days",
        "last 30 days",
        "last month",
        "last 3 months",
    ];

    let mut result = query.to_string();
    for term in &date_terms {
        // Simple case-insensitive replacement — build a pattern like \bterm\b.
        let pat = format!(r"(?i)\b{}\b", regex::escape(term));
        if let Ok(re) = Regex::new(&pat) {
            result = re.replace_all(&result, "").to_string();
        }
    }
    result.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_quoted_phrase() {
        let p = QueryParser::new();
        let pq = p.parse(r#""exact match" foo"#);
        assert!(pq.exact_phrases.contains(&"exact match".to_string()));
        assert!(pq.should_include.contains(&"foo".to_string()));
    }

    #[test]
    fn parses_must_exclude() {
        let p = QueryParser::new();
        let pq = p.parse("foo -bar");
        assert!(pq.must_exclude.contains(&"bar".to_string()));
    }

    #[test]
    fn parses_must_include() {
        let p = QueryParser::new();
        let pq = p.parse("+baz hello");
        assert!(pq.must_include.contains(&"baz".to_string()));
    }

    #[test]
    fn parses_and() {
        let p = QueryParser::new();
        let pq = p.parse("foo AND bar");
        assert!(pq.must_include.contains(&"foo".to_string()));
        assert!(pq.must_include.contains(&"bar".to_string()));
    }

    #[test]
    fn parses_or() {
        let p = QueryParser::new();
        let pq = p.parse("foo OR bar");
        assert!(pq.should_include.contains(&"foo".to_string()));
        assert!(pq.should_include.contains(&"bar".to_string()));
    }
}
