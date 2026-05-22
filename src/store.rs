use std::{collections::HashMap, sync::Arc};

use serde_json::Value;
use tokio::sync::RwLock;

use crate::error::ProxyError;

#[derive(Clone, Default)]
pub struct ResponseStore {
    inner: Arc<RwLock<HashMap<String, StoredResponse>>>,
}

#[derive(Debug, Clone)]
pub struct StoredResponse {
    pub response: Value,
    pub input_items: Vec<Value>,
    pub request_messages: Vec<Value>,
}

impl ResponseStore {
    pub async fn put(&self, response_id: String, stored: StoredResponse) -> Result<(), ProxyError> {
        self.inner.write().await.insert(response_id, stored);
        Ok(())
    }

    pub async fn get(&self, response_id: &str) -> Result<Option<StoredResponse>, ProxyError> {
        Ok(self.inner.read().await.get(response_id).cloned())
    }

    pub async fn delete(&self, response_id: &str) -> Result<Option<StoredResponse>, ProxyError> {
        Ok(self.inner.write().await.remove(response_id))
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
}
