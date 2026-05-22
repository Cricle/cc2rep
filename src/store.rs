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
