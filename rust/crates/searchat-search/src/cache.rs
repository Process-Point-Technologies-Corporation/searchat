use lru::LruCache;
use parking_lot::Mutex;
use searchat_models::SearchResults;
use std::num::NonZeroUsize;
use std::time::{Duration, Instant};

/// LRU cache with TTL-based eviction for search results.
pub struct SearchCache {
    inner: Mutex<LruCache<String, (SearchResults, Instant)>>,
    ttl: Duration,
}

impl SearchCache {
    pub fn new(capacity: usize, ttl: Duration) -> Self {
        let cap = NonZeroUsize::new(capacity.max(1)).expect("cache capacity > 0");
        Self {
            inner: Mutex::new(LruCache::new(cap)),
            ttl,
        }
    }

    /// Return a cached entry if it exists and has not expired.
    pub fn get(&self, key: &str) -> Option<SearchResults> {
        let mut guard = self.inner.lock();
        if let Some((results, ts)) = guard.get(key) {
            if ts.elapsed() < self.ttl {
                return Some(results.clone());
            }
            // Expired — remove it.
            guard.pop(key);
        }
        None
    }

    /// Insert a new entry, evicting the LRU entry when at capacity.
    pub fn insert(&self, key: String, results: SearchResults) {
        self.inner.lock().put(key, (results, Instant::now()));
    }
}
