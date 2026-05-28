use axum::{Router, body::Body, http::Request};
use http::StatusCode;
use http_body_util::BodyExt;
use serde_json::{Value, json};
use std::sync::LazyLock;
use tokio::sync::Mutex;
use tower::ServiceExt;

use cc2rep::{Settings, build_router, probe::probe_upstream};

const PROXY_KEY: &str = "test-proxy-key";
const MIMO_BASE_URL: &str = "https://token-plan-cn.xiaomimimo.com/v1";
const MIMO_MODEL: &str = "mimo-v2.5-pro";

fn mimo_api_key() -> Option<String> {
    std::env::var("MIMO_API_KEY").ok()
}

/// Skip the test if MIMO_API_KEY is not set.
macro_rules! require_api_key {
    () => {
        let Some(_) = mimo_api_key() else {
            eprintln!("skipping: MIMO_API_KEY not set");
            return;
        };
    };
}

/// Rate limiter: ensures minimum delay between API calls.
/// MiMo free tier has strict rate limits.
static RATE_LIMIT: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

async fn rate_limit() {
    let _guard = RATE_LIMIT.lock().await;
    tokio::time::sleep(std::time::Duration::from_millis(2000)).await;
}

/// Build a request to the responses endpoint.
fn responses_request(body: &Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/responses")
        .header("authorization", format!("Bearer {PROXY_KEY}"))
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .expect("request")
}

fn mimo_settings() -> Option<Settings> {
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
    })
}

fn weather_tool() -> Value {
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

fn calculator_tool() -> Value {
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

async fn shared_router() -> Option<&'static Router> {
    SHARED_ROUTER
        .get_or_init(|| async {
            let settings = mimo_settings()?;
            let caps = probe_upstream(&settings).await;
            Some(build_router(settings, caps))
        })
        .await
        .as_ref()
}

async fn send(router: &Router, request: Request<Body>) -> axum::response::Response {
    router.clone().oneshot(request).await.expect("response")
}

async fn send_json(router: &Router, body: Value) -> Value {
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

async fn send_json_expect_error(router: &Router, body: Value) -> (StatusCode, Value) {
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

async fn send_stream(router: &Router, body: Value) -> Vec<Value> {
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
fn last_message_text(body: &Value) -> &str {
    body["output"].as_array().unwrap().last().unwrap()["content"][0]["text"]
        .as_str()
        .unwrap()
}

// ============================================================
// Health check
// ============================================================

#[tokio::test]
async fn healthz_returns_ok() {
    require_api_key!();
    let Some(router) = shared_router().await else {
        return;
    };
    let response = send(
        router,
        Request::builder()
            .uri("/healthz")
            .body(Body::empty())
            .expect("request"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("collect")
        .to_bytes();
    let body: Value = serde_json::from_slice(&bytes).expect("json");
    assert_eq!(body["ok"], true);
}

// ============================================================
// Auth
// ============================================================

#[tokio::test]
async fn missing_auth_returns_401() {
    require_api_key!();
    let Some(router) = shared_router().await else {
        return;
    };
    let response = send(
        router,
        Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header("content-type", "application/json")
            .body(Body::from(json!({"input": "hi"}).to_string()))
            .expect("request"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn wrong_auth_returns_401() {
    require_api_key!();
    let Some(router) = shared_router().await else {
        return;
    };
    let response = send(
        router,
        Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header("authorization", "Bearer wrong-key")
            .header("content-type", "application/json")
            .body(Body::from(json!({"input": "hi"}).to_string()))
            .expect("request"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}

// ============================================================
// Non-streaming basic
// ============================================================

#[tokio::test]
async fn non_stream_basic_completion() {
    require_api_key!();
    let Some(router) = shared_router().await else {
        return;
    };
    let body = send_json(
        router,
        json!({
            "model": MIMO_MODEL,
            "input": "Say hello in one word",
            "stream": false
        }),
    )
    .await;

    assert_eq!(body["status"], "completed");
    assert_eq!(body["object"], "response");
    assert_eq!(body["model"], MIMO_MODEL);
    assert!(body["id"].as_str().unwrap().starts_with("resp_"));

    let output = body["output"].as_array().unwrap();
    assert!(!output.is_empty());
    let last = output.last().unwrap();
    assert_eq!(last["type"], "message");
    assert_eq!(last["role"], "assistant");
    assert_eq!(last["status"], "completed");

    let text = last["content"][0]["text"].as_str().unwrap();
    assert!(!text.is_empty());

    assert!(body["usage"]["input_tokens"].as_u64().unwrap() > 0);
    assert!(body["usage"]["output_tokens"].as_u64().unwrap() > 0);
}

// ============================================================
// Streaming basic
// ============================================================

#[tokio::test]
async fn stream_basic_completion() {
    require_api_key!();
    let Some(router) = shared_router().await else {
        return;
    };
    let events = send_stream(
        router,
        json!({
            "model": MIMO_MODEL,
            "input": "Count from 1 to 3",
            "stream": true
        }),
    )
    .await;

    assert!(!events.is_empty());

    // First event should be response.created
    assert_eq!(events[0]["type"], "response.created");
    assert_eq!(events[0]["response"]["status"], "in_progress");

    // Last event should be response.completed
    let last = events.last().unwrap();
    assert_eq!(last["type"], "response.completed");
    assert_eq!(last["response"]["status"], "completed");

    // Should have text deltas
    let has_text_delta = events
        .iter()
        .any(|e| e["type"] == "response.output_text.delta");
    assert!(has_text_delta, "should have text deltas");

    // Should have output_item.added for message
    let has_message = events
        .iter()
        .any(|e| e["type"] == "response.output_item.added" && e["item"]["type"] == "message");
    assert!(has_message, "should have message item");
}

// ============================================================
// Reasoning content
// ============================================================

#[tokio::test]
async fn non_stream_reasoning_content() {
    require_api_key!();
    let Some(router) = shared_router().await else {
        return;
    };
    let body = send_json(
        router,
        json!({
            "model": MIMO_MODEL,
            "input": "What is 2+2? Think step by step.",
            "stream": false
        }),
    )
    .await;

    let output = body["output"].as_array().unwrap();

    // Should have reasoning item
    let reasoning = output.iter().find(|o| o["type"] == "reasoning");
    assert!(reasoning.is_some(), "should have reasoning output item");
    let reasoning = reasoning.unwrap();
    assert_eq!(reasoning["status"], "completed");
    let summary = reasoning["summary"][0]["text"].as_str().unwrap();
    assert!(!summary.is_empty(), "reasoning summary should not be empty");

    // Should have message item
    let message = output.iter().find(|o| o["type"] == "message");
    assert!(message.is_some(), "should have message output item");

    // Usage should have reasoning_tokens
    let reasoning_tokens = body["usage"]["output_tokens_details"]["reasoning_tokens"]
        .as_u64()
        .unwrap();
    assert!(reasoning_tokens > 0, "should have reasoning tokens");
}

#[tokio::test]
async fn stream_reasoning_content() {
    require_api_key!();
    let Some(router) = shared_router().await else {
        return;
    };
    let events = send_stream(
        router,
        json!({
            "model": MIMO_MODEL,
            "input": "What is 1+1? Think step by step.",
            "stream": true
        }),
    )
    .await;

    // Should have reasoning item added
    let has_reasoning = events
        .iter()
        .any(|e| e["type"] == "response.output_item.added" && e["item"]["type"] == "reasoning");
    assert!(has_reasoning, "should have reasoning item in stream");

    // Should have reasoning summary text deltas
    let has_reasoning_delta = events
        .iter()
        .any(|e| e["type"] == "response.reasoning_summary_text.delta");
    assert!(has_reasoning_delta, "should have reasoning summary deltas");
}

// ============================================================
// Tool calls
// ============================================================

#[tokio::test]
async fn non_stream_tool_choice_auto() {
    require_api_key!();
    let Some(router) = shared_router().await else {
        return;
    };
    let body = send_json(
        router,
        json!({
            "model": MIMO_MODEL,
            "input": "What is the weather in Beijing?",
            "tools": [weather_tool()],
            "tool_choice": "auto",
            "stream": false
        }),
    )
    .await;

    assert_eq!(body["status"], "completed");
    let output = body["output"].as_array().unwrap();
    assert!(!output.is_empty());

    // Should have either function_call or message (model decides)
    let has_function_call = output.iter().any(|o| o["type"] == "function_call");
    let has_message = output.iter().any(|o| o["type"] == "message");
    assert!(
        has_function_call || has_message,
        "should have function_call or message"
    );
}

#[tokio::test]
async fn non_stream_tool_choice_required() {
    require_api_key!();
    let Some(router) = shared_router().await else {
        return;
    };
    let body = send_json(
        router,
        json!({
            "model": MIMO_MODEL,
            "input": "What is the weather in Shanghai?",
            "tools": [weather_tool()],
            "tool_choice": "required",
            "stream": false
        }),
    )
    .await;

    assert_eq!(body["status"], "completed");
    let output = body["output"].as_array().unwrap();

    // Should have function_call
    let function_call = output.iter().find(|o| o["type"] == "function_call");
    assert!(
        function_call.is_some(),
        "should have function_call with required"
    );
    let fc = function_call.unwrap();
    assert_eq!(fc["name"], "get_weather");
    assert!(fc["call_id"].as_str().unwrap().starts_with("call_"));

    let args: Value = serde_json::from_str(fc["arguments"].as_str().unwrap()).expect("args json");
    assert!(args["city"].as_str().is_some(), "should have city argument");
}

#[tokio::test]
async fn non_stream_named_tool_choice() {
    require_api_key!();
    let Some(router) = shared_router().await else {
        return;
    };
    let body = send_json(
        router,
        json!({
            "model": MIMO_MODEL,
            "input": "Check weather in Tokyo",
            "tools": [weather_tool()],
            "tool_choice": {"type": "function", "name": "get_weather"},
            "stream": false
        }),
    )
    .await;

    assert_eq!(body["status"], "completed");
    let output = body["output"].as_array().unwrap();

    let function_call = output.iter().find(|o| o["type"] == "function_call");
    assert!(
        function_call.is_some(),
        "should have function_call with named tool_choice"
    );
    assert_eq!(function_call.unwrap()["name"], "get_weather");
}

#[tokio::test]
async fn non_stream_multiple_tools() {
    require_api_key!();
    let Some(router) = shared_router().await else {
        return;
    };
    let body = send_json(
        router,
        json!({
            "model": MIMO_MODEL,
            "input": "What is the weather in Beijing and what is 100+200?",
            "tools": [weather_tool(), calculator_tool()],
            "tool_choice": "required",
            "stream": false
        }),
    )
    .await;

    assert_eq!(body["status"], "completed");
    let output = body["output"].as_array().unwrap();

    let function_calls: Vec<_> = output
        .iter()
        .filter(|o| o["type"] == "function_call")
        .collect();
    assert!(
        !function_calls.is_empty(),
        "should have at least one function_call"
    );

    // Check that tool names are valid
    for fc in &function_calls {
        let name = fc["name"].as_str().unwrap();
        assert!(
            name == "get_weather" || name == "calculate",
            "unexpected tool name: {name}"
        );
    }
}

#[tokio::test]
async fn stream_tool_calls() {
    require_api_key!();
    let Some(router) = shared_router().await else {
        return;
    };
    let events = send_stream(
        router,
        json!({
            "model": MIMO_MODEL,
            "input": "What is the weather in Beijing?",
            "tools": [weather_tool()],
            "tool_choice": "required",
            "stream": true
        }),
    )
    .await;

    let last = events.last().unwrap();
    assert_eq!(last["type"], "response.completed");

    let output = last["response"]["output"].as_array().unwrap();
    let function_call = output.iter().find(|o| o["type"] == "function_call");
    assert!(
        function_call.is_some(),
        "should have function_call in stream result"
    );
}

// ============================================================
// Local tool execution
// ============================================================

#[tokio::test]
async fn local_tool_auto_execution() {
    require_api_key!();
    let _guard = rate_limit().await;
    let Some(mut settings) = mimo_settings() else {
        return;
    };
    settings.local_tools.insert(
        "get_weather".to_owned(),
        cc2rep::config::LocalToolSettings {
            command: "sh".to_owned(),
            args: vec![
                "-c".to_owned(),
                r#"read JSON; CITY=$(echo "$JSON" | jq -r '.arguments.city // "unknown"'); echo "{\"temperature\": \"22°C\", \"condition\": \"sunny\", \"city\": \"$CITY\"}""#.to_owned(),
            ],
            env: Default::default(),
            workdir: None,
            timeout_seconds: 10.0,
            stdin_json: true,
            output_json: true,
        },
    );

    let caps = probe_upstream(&settings).await;
    let router = build_router(settings, caps);

    let body = send_json(
        &router,
        json!({
            "model": MIMO_MODEL,
            "input": "What is the weather in Beijing?",
            "tools": [weather_tool()],
            "tool_choice": "required",
            "stream": false
        }),
    )
    .await;

    assert_eq!(body["status"], "completed");
    let output = body["output"].as_array().unwrap();

    // Auto-tool execution: tools are executed internally, final response only has message
    let message = output.iter().find(|o| o["type"] == "message");
    assert!(
        message.is_some(),
        "should have final message after tool execution"
    );

    // Final message should reference the weather data from the local tool
    let text = message.unwrap()["content"][0]["text"].as_str().unwrap();
    assert!(
        text.contains("22") || text.contains("Beijing") || text.contains("sunny"),
        "final answer should reference tool output: {text}"
    );
}

// ============================================================
// Multi-turn conversation
// ============================================================

#[tokio::test]
async fn non_stream_multi_turn() {
    require_api_key!();
    let Some(router) = shared_router().await else {
        return;
    };
    let body = send_json(
        router,
        json!({
            "model": MIMO_MODEL,
            "input": [
                {"role": "user", "content": "My name is Alice"},
                {"role": "assistant", "content": "Hello Alice! Nice to meet you."},
                {"role": "user", "content": "What is my name?"}
            ],
            "stream": false
        }),
    )
    .await;

    assert_eq!(body["status"], "completed");
    let text = last_message_text(&body);
    assert!(
        text.contains("Alice"),
        "should remember name from multi-turn: {text}"
    );
}

// ============================================================
// System instructions
// ============================================================

#[tokio::test]
async fn non_stream_instructions() {
    require_api_key!();
    let Some(router) = shared_router().await else {
        return;
    };
    let body = send_json(
        router,
        json!({
            "model": MIMO_MODEL,
            "input": "What language should you respond in?",
            "instructions": "You must respond only in French.",
            "stream": false
        }),
    )
    .await;

    assert_eq!(body["status"], "completed");
    let text = last_message_text(&body);
    // French response should contain common French words
    let lower = text.to_lowercase();
    assert!(
        lower.contains("français")
            || lower.contains("french")
            || lower.contains("je")
            || lower.contains("dois")
            || lower.contains("répondre"),
        "should respond in French: {text}"
    );
}

// ============================================================
// Previous response ID
// ============================================================

#[tokio::test]
async fn previous_response_id() {
    require_api_key!();
    let Some(router) = shared_router().await else {
        return;
    };

    // First request
    let body1 = send_json(
        router,
        json!({
            "model": MIMO_MODEL,
            "input": "My favorite color is blue. Remember this.",
            "stream": false
        }),
    )
    .await;
    let resp_id = body1["id"].as_str().unwrap().to_owned();

    // Second request with previous_response_id
    let body2 = send_json(
        router,
        json!({
            "model": MIMO_MODEL,
            "input": "What is my favorite color?",
            "previous_response_id": resp_id,
            "stream": false
        }),
    )
    .await;

    assert_eq!(body2["status"], "completed");
    let text = last_message_text(&body2);
    assert!(
        text.contains("blue"),
        "should remember from previous response: {text}"
    );
}

// ============================================================
// Model alias
// ============================================================

#[tokio::test]
async fn model_alias() {
    require_api_key!();
    let _guard = rate_limit().await;
    let Some(mut settings) = mimo_settings() else {
        return;
    };
    settings
        .model_aliases
        .insert("my-alias".to_owned(), MIMO_MODEL.to_owned());
    let caps = probe_upstream(&settings).await;
    let router = build_router(settings, caps);

    let body = send_json(
        &router,
        json!({
            "model": "my-alias",
            "input": "Say hi",
            "stream": false
        }),
    )
    .await;

    assert_eq!(body["status"], "completed");
    assert_eq!(body["model"], "my-alias");
}

// ============================================================
// Response CRUD
// ============================================================

#[tokio::test]
async fn response_get_and_delete() {
    require_api_key!();
    let Some(router) = shared_router().await else {
        return;
    };

    // Create response
    let body = send_json(
        router,
        json!({
            "model": MIMO_MODEL,
            "input": "Say hello",
            "stream": false
        }),
    )
    .await;
    let resp_id = body["id"].as_str().unwrap().to_owned();

    // Get response
    let get_response = send(
        router,
        Request::builder()
            .method("GET")
            .uri(format!("/v1/responses/{resp_id}"))
            .header("authorization", format!("Bearer {PROXY_KEY}"))
            .body(Body::empty())
            .expect("request"),
    )
    .await;
    assert_eq!(get_response.status(), StatusCode::OK);
    let get_body: Value = serde_json::from_slice(
        &get_response
            .into_body()
            .collect()
            .await
            .expect("collect")
            .to_bytes(),
    )
    .expect("json");
    assert_eq!(get_body["id"], resp_id);
    assert_eq!(get_body["status"], "completed");

    // Delete response
    let del_response = send(
        router,
        Request::builder()
            .method("DELETE")
            .uri(format!("/v1/responses/{resp_id}"))
            .header("authorization", format!("Bearer {PROXY_KEY}"))
            .body(Body::empty())
            .expect("request"),
    )
    .await;
    assert_eq!(del_response.status(), StatusCode::OK);

    // Get after delete should fail (proxy returns 400 for not found)
    let get_after_del = send(
        router,
        Request::builder()
            .method("GET")
            .uri(format!("/v1/responses/{resp_id}"))
            .header("authorization", format!("Bearer {PROXY_KEY}"))
            .body(Body::empty())
            .expect("request"),
    )
    .await;
    assert_eq!(get_after_del.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn response_list_input_items() {
    require_api_key!();
    let Some(router) = shared_router().await else {
        return;
    };

    let body = send_json(
        router,
        json!({
            "model": MIMO_MODEL,
            "input": "Say hello",
            "stream": false
        }),
    )
    .await;
    let resp_id = body["id"].as_str().unwrap().to_owned();

    let response = send(
        router,
        Request::builder()
            .method("GET")
            .uri(format!("/v1/responses/{resp_id}/input_items"))
            .header("authorization", format!("Bearer {PROXY_KEY}"))
            .body(Body::empty())
            .expect("request"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let items: Value = serde_json::from_slice(
        &response
            .into_body()
            .collect()
            .await
            .expect("collect")
            .to_bytes(),
    )
    .expect("json");
    assert!(!items["data"].as_array().unwrap().is_empty());
}

// ============================================================
// Cancel
// ============================================================

#[tokio::test]
async fn cancel_nonexistent_returns_400() {
    require_api_key!();
    let Some(router) = shared_router().await else {
        return;
    };
    let response = send(
        router,
        Request::builder()
            .method("POST")
            .uri("/v1/responses/resp_nonexistent/cancel")
            .header("authorization", format!("Bearer {PROXY_KEY}"))
            .body(Body::empty())
            .expect("request"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

// ============================================================
// Error handling
// ============================================================

#[tokio::test]
async fn non_object_body_returns_400() {
    require_api_key!();
    let Some(router) = shared_router().await else {
        return;
    };
    let response = send(
        router,
        Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header("authorization", format!("Bearer {PROXY_KEY}"))
            .header("content-type", "application/json")
            .body(Body::from("[]"))
            .expect("request"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn missing_input_returns_400() {
    require_api_key!();
    let Some(router) = shared_router().await else {
        return;
    };
    let (status, _) = send_json_expect_error(
        router,
        json!({
            "model": MIMO_MODEL
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn delete_missing_returns_400() {
    require_api_key!();
    let Some(router) = shared_router().await else {
        return;
    };
    let response = send(
        router,
        Request::builder()
            .method("DELETE")
            .uri("/v1/responses/resp_nonexistent")
            .header("authorization", format!("Bearer {PROXY_KEY}"))
            .body(Body::empty())
            .expect("request"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn get_missing_returns_400() {
    require_api_key!();
    let Some(router) = shared_router().await else {
        return;
    };
    let response = send(
        router,
        Request::builder()
            .method("GET")
            .uri("/v1/responses/resp_nonexistent")
            .header("authorization", format!("Bearer {PROXY_KEY}"))
            .body(Body::empty())
            .expect("request"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

// ============================================================
// Drop tools
// ============================================================

#[tokio::test]
async fn drop_tools_ignores_tool_definitions() {
    require_api_key!();
    let _guard = rate_limit().await;
    let Some(mut settings) = mimo_settings() else {
        return;
    };
    settings.drop_tools = true;
    let caps = probe_upstream(&settings).await;
    let router = build_router(settings, caps);

    let body = send_json(
        &router,
        json!({
            "model": MIMO_MODEL,
            "input": "What is the weather?",
            "tools": [weather_tool()],
            "tool_choice": "required",
            "stream": false
        }),
    )
    .await;

    assert_eq!(body["status"], "completed");
    // Should NOT have function_call since tools were dropped
    let output = body["output"].as_array().unwrap();
    let has_function_call = output.iter().any(|o| o["type"] == "function_call");
    assert!(
        !has_function_call,
        "should not have function_call when drop_tools is true"
    );
}

// ============================================================
// Store: false
// ============================================================

#[tokio::test]
async fn store_false_not_persisted() {
    require_api_key!();
    let Some(router) = shared_router().await else {
        return;
    };
    let body = send_json(
        router,
        json!({
            "model": MIMO_MODEL,
            "input": "Say hi",
            "store": false,
            "stream": false
        }),
    )
    .await;

    let resp_id = body["id"].as_str().unwrap().to_owned();

    // Should not be retrievable (proxy returns 400 for not found)
    let response = send(
        router,
        Request::builder()
            .method("GET")
            .uri(format!("/v1/responses/{resp_id}"))
            .header("authorization", format!("Bearer {PROXY_KEY}"))
            .body(Body::empty())
            .expect("request"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

// ============================================================
// Parallel tool calls
// ============================================================

#[tokio::test]
async fn parallel_tool_calls_setting() {
    require_api_key!();
    let Some(router) = shared_router().await else {
        return;
    };
    let body = send_json(
        router,
        json!({
            "model": MIMO_MODEL,
            "input": "Say hi",
            "parallel_tool_calls": true,
            "stream": false
        }),
    )
    .await;

    assert_eq!(body["parallel_tool_calls"], true);
}

// ============================================================
// Text format
// ============================================================

#[tokio::test]
async fn response_has_correct_format() {
    require_api_key!();
    let Some(router) = shared_router().await else {
        return;
    };
    let body = send_json(
        router,
        json!({
            "model": MIMO_MODEL,
            "input": "Say hi",
            "stream": false
        }),
    )
    .await;

    assert_eq!(body["text"]["format"]["type"], "text");
}

// ============================================================
// Usage
// ============================================================

#[tokio::test]
async fn usage_fields_present() {
    require_api_key!();
    let Some(router) = shared_router().await else {
        return;
    };
    let body = send_json(
        router,
        json!({
            "model": MIMO_MODEL,
            "input": "What is 2+2? Think step by step.",
            "stream": false
        }),
    )
    .await;

    let usage = &body["usage"];
    assert!(usage["input_tokens"].as_u64().unwrap() > 0);
    assert!(usage["output_tokens"].as_u64().unwrap() > 0);
    assert!(usage["total_tokens"].as_u64().unwrap() > 0);
    assert!(
        usage["input_tokens_details"]["cached_tokens"]
            .as_u64()
            .is_some()
    );
    assert!(
        usage["output_tokens_details"]["reasoning_tokens"]
            .as_u64()
            .is_some()
    );
}

// ============================================================
// Stream: reasoning + message sequence
// ============================================================

#[tokio::test]
async fn stream_output_item_order() {
    require_api_key!();
    let Some(router) = shared_router().await else {
        return;
    };
    let events = send_stream(
        router,
        json!({
            "model": MIMO_MODEL,
            "input": "What is 1+1? Think step by step.",
            "stream": true
        }),
    )
    .await;

    // Find indices of key events
    let created_idx = events.iter().position(|e| e["type"] == "response.created");
    let completed_idx = events
        .iter()
        .position(|e| e["type"] == "response.completed");

    assert!(created_idx.is_some(), "should have response.created");
    assert!(completed_idx.is_some(), "should have response.completed");
    assert!(
        created_idx.unwrap() < completed_idx.unwrap(),
        "created before completed"
    );

    // reasoning should come before message
    let reasoning_idx = events.iter().position(|e| {
        e["type"] == "response.output_item.added" && e["item"]["type"] == "reasoning"
    });
    let message_idx = events
        .iter()
        .position(|e| e["type"] == "response.output_item.added" && e["item"]["type"] == "message");

    if let (Some(ri), Some(mi)) = (reasoning_idx, message_idx) {
        assert!(ri < mi, "reasoning should come before message");
    }
}

// ============================================================
// Input formats
// ============================================================

#[tokio::test]
async fn string_input_works() {
    require_api_key!();
    let Some(router) = shared_router().await else {
        return;
    };
    let body = send_json(
        router,
        json!({
            "model": MIMO_MODEL,
            "input": "Hello",
            "stream": false
        }),
    )
    .await;
    assert_eq!(body["status"], "completed");
}

#[tokio::test]
async fn array_input_works() {
    require_api_key!();
    let Some(router) = shared_router().await else {
        return;
    };
    let body = send_json(
        router,
        json!({
            "model": MIMO_MODEL,
            "input": [{"role": "user", "content": "Hello"}],
            "stream": false
        }),
    )
    .await;
    assert_eq!(body["status"], "completed");
}

// ============================================================
// Edge cases
// ============================================================

#[tokio::test]
async fn max_output_tokens_zero_is_ignored() {
    require_api_key!();
    let Some(router) = shared_router().await else {
        return;
    };
    let body = send_json(
        router,
        json!({
            "model": MIMO_MODEL,
            "input": "Say hello",
            "max_output_tokens": 0,
            "stream": false
        }),
    )
    .await;
    assert_eq!(body["status"], "completed");
}

#[tokio::test]
async fn temperature_and_top_p_passed_through() {
    require_api_key!();
    let Some(router) = shared_router().await else {
        return;
    };
    let body = send_json(
        router,
        json!({
            "model": MIMO_MODEL,
            "input": "Say hello",
            "temperature": 0.5,
            "top_p": 0.9,
            "stream": false
        }),
    )
    .await;
    assert_eq!(body["status"], "completed");
}

#[tokio::test]
async fn stop_sequences_passed_through() {
    require_api_key!();
    let Some(router) = shared_router().await else {
        return;
    };
    let body = send_json(
        router,
        json!({
            "model": MIMO_MODEL,
            "input": "Count: 1 2 3 4 5",
            "stop": ["3"],
            "stream": false
        }),
    )
    .await;
    assert_eq!(body["status"], "completed");
}
