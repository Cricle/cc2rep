#![allow(dead_code)]

use axum::{Router, body::Body, http::Request};
use http::StatusCode;
use http_body_util::BodyExt;
use serde_json::{Value, json};
use std::sync::LazyLock;
use tokio::sync::Mutex;
use tower::ServiceExt;

use cc2rep::{Settings, build_router, probe::probe_upstream};

pub const PROXY_KEY: &str = "test-proxy-key";
pub const MIMO_BASE_URL: &str = "https://token-plan-cn.xiaomimimo.com/v1";
pub const MIMO_MODEL: &str = "mimo-v2.5-pro";

pub fn mimo_api_key() -> Option<String> {
    std::env::var("MIMO_API_KEY").ok()
}

/// Skip the test if MIMO_API_KEY is not set.
#[macro_export]
macro_rules! require_api_key {
    () => {
        let Some(_) = $crate::common::mimo_api_key() else {
            eprintln!("skipping: MIMO_API_KEY not set");
            return;
        };
    };
}

/// Rate limiter: ensures minimum delay between API calls.
pub static RATE_LIMIT: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

pub async fn rate_limit() {
    let _guard = RATE_LIMIT.lock().await;
    tokio::time::sleep(std::time::Duration::from_millis(2000)).await;
}

pub fn mimo_settings() -> Option<Settings> {
    Some(Settings {
        proxy_host: "127.0.0.1".to_owned(),
        proxy_port: 8800,
        proxy_api_key: PROXY_KEY.to_owned(),
        upstream_base_url: MIMO_BASE_URL.to_owned(),
        upstream_chat_path: "/chat/completions".to_owned(),
        upstream_model: MIMO_MODEL.to_owned(),
        upstream_api_key: mimo_api_key()?,
        upstream_headers: Default::default(),
        upstream_api_key_header_name: "Authorization".to_owned(),
        upstream_api_key_prefix: "Bearer ".to_owned(),
        request_timeout_seconds: 120.0,
        strict_protocol: false,
        upstream_supports_image_input: false,
        upstream_supports_reasoning_content: None,
        upstream_supports_tool_choice_required: None,
        upstream_supports_named_tool_choice: None,
        response_ttl_seconds: 3600,
        drop_input_reasoning: false,
        drop_tools: false,
        upstream_body: Default::default(),
        model_aliases: Default::default(),
        local_tools: Default::default(),
        max_auto_tool_rounds: 8,
        upstream_max_retries: 3,
        upstream_retry_base_delay_ms: 1000,
        web_search_url: None,
        web_search_max_results: 5,
        file_search_paths: Vec::new(),
        file_search_max_results: 5,
    })
}

pub fn weather_tool() -> Value {
    json!({
        "type": "function",
        "name": "get_weather",
        "description": "Get the current weather for a given city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The city name"}
            },
            "required": ["city"]
        }
    })
}

pub fn calculator_tool() -> Value {
    json!({
        "type": "function",
        "name": "calculate",
        "description": "Calculate a math expression",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression to evaluate"}
            },
            "required": ["expression"]
        }
    })
}

/// Shared router: probe once, reuse for all tests.
static SHARED_ROUTER: LazyLock<tokio::sync::OnceCell<Option<Router>>> =
    LazyLock::new(tokio::sync::OnceCell::new);

pub async fn shared_router() -> Option<&'static Router> {
    SHARED_ROUTER
        .get_or_init(|| async {
            let settings = mimo_settings()?;
            let caps = probe_upstream(&settings).await;
            Some(build_router(settings, caps))
        })
        .await
        .as_ref()
}

pub fn responses_request(body: &Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/responses")
        .header("authorization", format!("Bearer {PROXY_KEY}"))
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .expect("request")
}

pub async fn send(router: &Router, request: Request<Body>) -> axum::response::Response {
    router.clone().oneshot(request).await.expect("response")
}

pub async fn send_json(router: &Router, body: Value) -> Value {
    for attempt in 0..4 {
        rate_limit().await;
        let response = send(router, responses_request(&body)).await;
        if response.status() == StatusCode::TOO_MANY_REQUESTS && attempt < 3 {
            eprintln!(
                "429 rate limited, retrying in 10s (attempt {})",
                attempt + 1
            );
            tokio::time::sleep(std::time::Duration::from_secs(10)).await;
            continue;
        }
        assert_eq!(response.status(), StatusCode::OK, "request failed");
        let bytes = response
            .into_body()
            .collect()
            .await
            .expect("collect")
            .to_bytes();
        return serde_json::from_slice(&bytes).expect("json");
    }
    panic!("too many 429 retries");
}

pub async fn send_json_expect_error(router: &Router, body: Value) -> (StatusCode, Value) {
    rate_limit().await;
    let response = send(router, responses_request(&body)).await;
    let status = response.status();
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("collect")
        .to_bytes();
    let body: Value = serde_json::from_slice(&bytes).unwrap_or(json!(null));
    (status, body)
}

pub async fn send_stream(router: &Router, body: Value) -> Vec<Value> {
    for attempt in 0..4 {
        rate_limit().await;
        let response = send(router, responses_request(&body)).await;
        if response.status() == StatusCode::TOO_MANY_REQUESTS && attempt < 3 {
            eprintln!(
                "429 rate limited, retrying in 10s (attempt {})",
                attempt + 1
            );
            tokio::time::sleep(std::time::Duration::from_secs(10)).await;
            continue;
        }
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = response
            .into_body()
            .collect()
            .await
            .expect("collect")
            .to_bytes();
        let text = String::from_utf8_lossy(&bytes);
        return text
            .lines()
            .filter_map(|line| {
                let stripped = line.strip_prefix("data: ")?;
                if stripped == "[DONE]" {
                    return None;
                }
                serde_json::from_str(stripped).ok()
            })
            .collect();
    }
    panic!("too many 429 retries");
}

/// Helper: get the last message text from a non-streaming response body.
pub fn last_message_text(body: &Value) -> &str {
    body["output"].as_array().unwrap().last().unwrap()["content"][0]["text"]
        .as_str()
        .unwrap()
}
