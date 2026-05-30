mod exec;
mod helpers;
mod restore;
mod store_ext;
mod streaming;

#[cfg(test)]
mod tests;

use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use axum::{
    Router,
    routing::{get, post},
};
use serde_json::Value;
use tokio::sync::RwLock;

use crate::{
    config::Settings,
    metrics::RequestMetrics,
    probe::Capabilities,
    store::ResponseStore,
    tools::ToolExecutor,
    translate::AssistantTurn,
    upstream::UpstreamClient,
};

pub(crate) use exec::{execute_non_stream_turn, should_auto_execute_tools};
pub(crate) use helpers::{extract_request_messages, translated_with_request_messages};
pub(crate) use restore::{assistant_turn_from_output, context_from_response};
pub(crate) use store_ext::{load_previous_messages, store_final_response};
pub(crate) use streaming::{stream_response, stream_response_with_auto_tools};

#[derive(Clone)]
pub(crate) struct AppState {
    pub settings: Arc<Settings>,
    pub capabilities: Capabilities,
    pub upstream: UpstreamClient,
    pub http_client: reqwest::Client,
    pub tool_executor: ToolExecutor,
    pub store: ResponseStore,
    pub inflight: InflightRegistry,
    pub metrics: Arc<RequestMetrics>,
}

#[derive(Clone, Default)]
pub(crate) struct InflightRegistry {
    inner: Arc<RwLock<HashMap<String, Arc<AtomicBool>>>>,
}

#[derive(Clone)]
pub(crate) struct NonStreamExecution {
    pub turn: AssistantTurn,
    pub usage: Value,
    pub request_messages: Vec<Value>,
}

impl InflightRegistry {
    pub(crate) async fn start(&self, response_id: String) -> Arc<AtomicBool> {
        let token = Arc::new(AtomicBool::new(false));
        self.inner.write().await.insert(response_id, token.clone());
        token
    }

    pub(crate) async fn cancel(&self, response_id: &str) -> bool {
        let guard = self.inner.read().await;
        let Some(flag) = guard.get(response_id) else {
            return false;
        };
        flag.store(true, Ordering::SeqCst);
        true
    }

    pub(crate) async fn finish(&self, response_id: &str) {
        self.inner.write().await.remove(response_id);
    }

    pub(crate) async fn count(&self) -> usize {
        self.inner.read().await.len()
    }
}


pub fn build_router(settings: Settings, capabilities: Capabilities) -> Router {
    let settings = Arc::new(settings);
    let upstream = UpstreamClient::new(settings.clone()).expect("invalid settings");
    let store = ResponseStore::with_ttl(settings.response_ttl_seconds);
    store.start_cleanup_task();
    let http_client = reqwest::Client::new();
    let state = AppState {
        tool_executor: ToolExecutor::new(settings.clone()),
        store,
        capabilities,
        settings,
        upstream,
        http_client,
        inflight: InflightRegistry::default(),
        metrics: Arc::new(RequestMetrics::default()),
    };

    Router::new()
        .route("/healthz", get(crate::handlers::healthz))
        .route("/stats", get(crate::handlers::stats))
        .route("/v1/responses", post(crate::handlers::create_response))
        .route(
            "/v1/responses/{response_id}",
            get(crate::handlers::get_response).delete(crate::handlers::delete_response),
        )
        .route(
            "/v1/responses/{response_id}/input_items",
            get(crate::handlers::list_input_items),
        )
        .route(
            "/v1/responses/{response_id}/cancel",
            post(crate::handlers::cancel_response),
        )
        .with_state(state)
}

