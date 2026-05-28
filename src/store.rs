use std::{collections::HashMap, sync::Arc, time::Instant};

use serde_json::Value;
use tokio::sync::RwLock;
use tokio::time::{Duration, interval};
use tracing::{debug, info};

use crate::error::ProxyError;

#[derive(Clone, Default)]
pub struct ResponseStore {
    inner: Arc<RwLock<HashMap<String, StoredResponse>>>,
    ttl_seconds: Arc<u64>,
}

#[derive(Debug, Clone)]
pub struct StoredResponse {
    pub response: Value,
    pub input_items: Vec<Value>,
    pub request_messages: Vec<Value>,
    pub inserted_at: Instant,
}

impl ResponseStore {
    pub fn with_ttl(ttl_seconds: u64) -> Self {
        Self {
            inner: Arc::new(RwLock::new(HashMap::new())),
            ttl_seconds: Arc::new(ttl_seconds),
        }
    }

    pub async fn put(&self, response_id: String, stored: StoredResponse) -> Result<(), ProxyError> {
        let mut guard = self.inner.write().await;
        guard.insert(response_id, stored);
        // Opportunistic cleanup: every 32 inserts, remove expired entries
        if guard.len() % 32 == 0 {
            let ttl = *self.ttl_seconds;
            guard.retain(|id, entry| {
                let alive = entry.inserted_at.elapsed().as_secs() < ttl;
                if !alive {
                    debug!(response_id = %id, "expired stored response");
                }
                alive
            });
        }
        Ok(())
    }

    pub async fn get(&self, response_id: &str) -> Result<Option<StoredResponse>, ProxyError> {
        Ok(self.inner.read().await.get(response_id).cloned())
    }

    pub async fn delete(&self, response_id: &str) -> Result<Option<StoredResponse>, ProxyError> {
        Ok(self.inner.write().await.remove(response_id))
    }

    pub fn start_cleanup_task(&self) {
        let store = self.clone();
        let ttl = *self.ttl_seconds;
        if ttl == 0 {
            info!("TTL is 0, skipping cleanup task");
            return;
        }
        let interval_secs = if ttl < 60 { ttl.max(10) } else { 60 };
        tokio::spawn(async move {
            let mut ticker = interval(Duration::from_secs(interval_secs));
            ticker.tick().await; // skip first immediate tick
            loop {
                ticker.tick().await;
                let before = store.inner.read().await.len();
                let mut guard = store.inner.write().await;
                guard.retain(|id, entry| {
                    let alive = entry.inserted_at.elapsed().as_secs() < ttl;
                    if !alive {
                        debug!(response_id = %id, "cleanup: removed expired response");
                    }
                    alive
                });
                let after = guard.len();
                let removed = before.saturating_sub(after);
                if removed > 0 {
                    info!(removed, remaining = after, "periodic cleanup completed");
                }
                drop(guard);
            }
        });
    }

    pub async fn update_response(
        &self,
        response_id: &str,
        response: Value,
    ) -> Result<(), ProxyError> {
        let mut guard = self.inner.write().await;
        if let Some(stored) = guard.get_mut(response_id) {
            stored.response = response;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn stored_response(text: &str) -> StoredResponse {
        StoredResponse {
            response: json!({"id":"resp_1","output_text":text}),
            input_items: vec![json!({"type":"message"})],
            request_messages: vec![json!({"role":"user","content":"hi"})],
            inserted_at: std::time::Instant::now(),
        }
    }

    #[tokio::test]
    async fn put_get_update_delete_round_trip() {
        let store = ResponseStore::default();
        store
            .put("resp_1".to_owned(), stored_response("first"))
            .await
            .expect("put");

        let loaded = store.get("resp_1").await.expect("get").expect("stored");
        assert_eq!(loaded.response["output_text"], "first");
        assert_eq!(loaded.input_items.len(), 1);
        assert_eq!(loaded.request_messages.len(), 1);

        store
            .update_response("resp_1", json!({"id":"resp_1","output_text":"updated"}))
            .await
            .expect("update");
        let updated = store.get("resp_1").await.expect("get").expect("stored");
        assert_eq!(updated.response["output_text"], "updated");

        let deleted = store.delete("resp_1").await.expect("delete");
        assert!(deleted.is_some());
        assert!(store.get("resp_1").await.expect("get").is_none());
    }

    #[tokio::test]
    async fn update_and_delete_missing_entries_are_noops() {
        let store = ResponseStore::default();
        store
            .update_response("missing", json!({"id":"missing"}))
            .await
            .expect("update");
        assert!(store.delete("missing").await.expect("delete").is_none());
    }

    #[tokio::test]
    async fn cleanup_removes_expired_entries() {
        use std::time::Duration;

        let store = ResponseStore::with_ttl(1); // 1 second TTL

        // Insert a response with an old timestamp
        let old = StoredResponse {
            response: json!({"id":"old"}),
            input_items: vec![],
            request_messages: vec![],
            inserted_at: Instant::now() - Duration::from_secs(10),
        };
        store.put("old".to_owned(), old).await.expect("put");

        // Insert a fresh response
        store
            .put("fresh".to_owned(), stored_response("fresh"))
            .await
            .expect("put");

        assert_eq!(store.inner.read().await.len(), 2);

        // Run cleanup
        store.start_cleanup_task();

        // Wait for cleanup to run (interval is max(ttl, 10) = 10s for ttl=1, but we can test manually)
        let mut guard = store.inner.write().await;
        let ttl = *store.ttl_seconds;
        guard.retain(|_, entry| entry.inserted_at.elapsed().as_secs() < ttl);
        drop(guard);

        assert_eq!(store.inner.read().await.len(), 1);
        assert!(store.get("old").await.expect("get").is_none());
        assert!(store.get("fresh").await.expect("get").is_some());
    }
}
