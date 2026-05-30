use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use async_stream::stream;
use axum::{
    Json, Router,
    body::Body,
    extract::{Path, Request, State},
    http::{HeaderMap, StatusCode},
    response::{
        IntoResponse,
        sse::{Event, Sse},
    },
    routing::post,
};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tower::ServiceExt;

use super::*;
use crate::config::Settings;
use crate::handlers::{authorize, cancel_response};
use crate::metrics::RequestMetrics;
use crate::probe::Capabilities;
use crate::store::{ResponseStore, StoredResponse};
use crate::stream::{
    SseParser, StreamState, apply_stream_delta, finalize_stream_items, json_event, parse_sse_event,
};
use crate::tools::ToolExecutor;
use crate::translate::{AssistantTurn, RequestContext, ToolCall};
use crate::upstream::UpstreamClient;

fn settings(upstream_base_url: String, image_input: bool) -> Settings {
    Settings {
        proxy_host: "127.0.0.1".to_owned(),
        proxy_port: 8800,
        proxy_api_key: "proxy-secret".to_owned(),
        upstream_base_url,
        upstream_chat_path: "/v1/chat/completions".to_owned(),
        upstream_model: "deepseek-chat".to_owned(),
        upstream_api_key: "upstream-secret".to_owned(),
        upstream_headers: Default::default(),
        upstream_api_key_header_name: "Authorization".to_owned(),
        upstream_api_key_prefix: "Bearer ".to_owned(),
        request_timeout_seconds: 10.0,
        strict_protocol: false,
        upstream_supports_image_input: image_input,
        upstream_supports_reasoning_content: None,
        upstream_supports_tool_choice_required: None,
        upstream_supports_named_tool_choice: None,
        upstream_supports_reasoning_effort: None,
        response_ttl_seconds: 3600,
        drop_input_reasoning: false,
        drop_tools: false,
        upstream_body: Default::default(),
        model_aliases: [("gpt-5-codex".to_owned(), "deepseek-chat".to_owned())]
            .into_iter()
            .collect(),
        local_tools: Default::default(),
        max_auto_tool_rounds: 8,
        upstream_max_retries: 0,
        upstream_retry_base_delay_ms: 0,
        upstream_reasoning_effort_field: "reasoning_effort".to_owned(),
        web_search_url: None,
        web_search_max_results: 5,
        file_search_paths: Vec::new(),
        file_search_max_results: 5,
    }
}

fn caps(image_input: bool) -> Capabilities {
    Capabilities {
        supports_named_tool_choice: false,
        supports_tool_choice_required: false,
        supports_reasoning_content: false,
        supports_image_input: image_input,
        supports_reasoning_effort: false,
    }
}

async fn spawn_upstream() -> String {
    let app = Router::new().route(
            "/v1/chat/completions",
            post(|Json(payload): Json<Value>| async move {
                if payload.get("stream").and_then(Value::as_bool) == Some(true) {
                    let stream = stream! {
                        yield Ok::<Event, Infallible>(Event::default().data(r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1714444444,"model":"deepseek-chat","choices":[{"index":0,"delta":{"reasoning_content":"Think "},"finish_reason":null}]}"#));
                        yield Ok::<Event, Infallible>(Event::default().data(r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1714444444,"model":"deepseek-chat","choices":[{"index":0,"delta":{"reasoning_content":"carefully.","content":"Final"},"finish_reason":null}]}"#));
                        tokio::time::sleep(Duration::from_millis(50)).await;
                        yield Ok::<Event, Infallible>(Event::default().data(r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1714444444,"model":"deepseek-chat","choices":[{"index":0,"delta":{"content":" answer"},"finish_reason":"stop"}],"usage":{"prompt_tokens":11,"completion_tokens":4,"total_tokens":15}}"#));
                        yield Ok::<Event, Infallible>(Event::default().data("[DONE]"));
                    };
                    Sse::new(stream).into_response()
                } else {
                    let has_image = payload
                        .get("messages")
                        .and_then(Value::as_array)
                        .and_then(|messages| messages.first())
                        .and_then(|message| message.get("content"))
                        .map(|content| content.is_array())
                        .unwrap_or(false);
                    Json(json!({
                        "id": "chatcmpl-1",
                        "object": "chat.completion",
                        "created": 1714444444,
                        "model": "deepseek-chat",
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": if has_image { "Vision answer" } else { "Hello from upstream" },
                                "reasoning_content": "I checked the constraints."
                            },
                            "finish_reason": "stop"
                        }],
                        "usage": {
                            "prompt_tokens": 11,
                            "completion_tokens": 4,
                            "total_tokens": 15
                        }
                    }))
                    .into_response()
                }
            }),
        );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind test upstream");
    let addr = listener.local_addr().expect("local addr");
    drop(tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    }));
    format!("http://{addr}")
}

async fn spawn_upstream_with_router(app: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind test upstream");
    let addr = listener.local_addr().expect("local addr");
    drop(tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    }));
    format!("http://{addr}")
}

async fn send(router: &Router, request: Request<Body>) -> axum::response::Response {
    router.clone().oneshot(request).await.expect("response")
}

#[tokio::test]
async fn non_stream_reasoning_is_mapped_to_output_item() {
    let upstream_base_url = spawn_upstream().await;
    let router = build_router(settings(upstream_base_url, false), caps(false));

    let response = send(
        &router,
        Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header("authorization", "Bearer proxy-secret")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({"model":"gpt-5-codex","input":"Say hello"}).to_string(),
            ))
            .expect("request"),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = response
        .into_body()
        .collect()
        .await
        .expect("collect")
        .to_bytes();
    let payload: Value = serde_json::from_slice(&body).expect("json");
    assert_eq!(payload["output"][0]["type"], "reasoning");
    assert_eq!(payload["output"][1]["type"], "message");
}

#[tokio::test]
async fn stream_reasoning_events_are_emitted() {
    let upstream_base_url = spawn_upstream().await;
    let router = build_router(settings(upstream_base_url, false), caps(false));

    let response = send(
        &router,
        Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header("authorization", "Bearer proxy-secret")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({"model":"gpt-5-codex","input":"Think","stream":true}).to_string(),
            ))
            .expect("request"),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = String::from_utf8(
        response
            .into_body()
            .collect()
            .await
            .expect("collect")
            .to_bytes()
            .to_vec(),
    )
    .expect("utf8");
    assert!(body.contains("event: response.reasoning_summary_text.delta"));
    assert!(body.contains("event: response.reasoning_summary_text.done"));
    assert!(body.contains("event: response.completed"));
}

#[tokio::test]
async fn image_input_requires_flag() {
    let upstream_base_url = spawn_upstream().await;
    let router = build_router(settings(upstream_base_url, false), caps(false));

    let response = send(
            &router,
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header("authorization", "Bearer proxy-secret")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model":"gpt-5-codex",
                        "input":[{
                            "type":"message",
                            "role":"user",
                            "content":[{"type":"input_image","image_url":"https://example.test/cat.png"}]
                        }]
                    })
                    .to_string(),
                ))
                .expect("request"),
        )
        .await;

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn image_input_is_forwarded_when_enabled() {
    let upstream_base_url = spawn_upstream().await;
    let router = build_router(settings(upstream_base_url, true), caps(true));

    let response = send(
            &router,
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header("authorization", "Bearer proxy-secret")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model":"gpt-5-codex",
                        "input":[{
                            "type":"message",
                            "role":"user",
                            "content":[{"type":"input_image","image_url":"https://example.test/cat.png"}]
                        }]
                    })
                    .to_string(),
                ))
                .expect("request"),
        )
        .await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = response
        .into_body()
        .collect()
        .await
        .expect("collect")
        .to_bytes();
    let payload: Value = serde_json::from_slice(&body).expect("json");
    assert_eq!(payload["output_text"], "Vision answer");
}

#[tokio::test]
async fn cancel_marks_inflight_stream_as_cancelled() {
    let upstream_base_url = spawn_upstream().await;
    let router = build_router(settings(upstream_base_url, false), caps(false));

    let create = send(
        &router,
        Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header("authorization", "Bearer proxy-secret")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({"model":"gpt-5-codex","input":"Think","stream":true,"store":true})
                    .to_string(),
            ))
            .expect("request"),
    );

    tokio::time::sleep(Duration::from_millis(10)).await;

    let stored_create = send(
        &router,
        Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header("authorization", "Bearer proxy-secret")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({"model":"gpt-5-codex","input":"seed"}).to_string(),
            ))
            .expect("request"),
    )
    .await;
    let stored_body = stored_create
        .into_body()
        .collect()
        .await
        .expect("collect")
        .to_bytes();
    let stored_payload: Value = serde_json::from_slice(&stored_body).expect("json");
    let response_id = stored_payload["id"]
        .as_str()
        .expect("response id")
        .to_owned();

    let _stream_response = create.await;

    let cancelled = send(
        &router,
        Request::builder()
            .method("POST")
            .uri(format!("/v1/responses/{response_id}/cancel"))
            .header("authorization", "Bearer proxy-secret")
            .body(Body::empty())
            .expect("request"),
    )
    .await;

    assert!(matches!(
        cancelled.status(),
        StatusCode::OK | StatusCode::BAD_REQUEST
    ));
}

#[tokio::test]
async fn healthz_and_auth_and_response_crud_work() {
    let upstream_base_url = spawn_upstream().await;
    let router = build_router(settings(upstream_base_url, false), caps(false));

    let health = send(
        &router,
        Request::builder()
            .method("GET")
            .uri("/healthz")
            .body(Body::empty())
            .expect("request"),
    )
    .await;
    assert_eq!(health.status(), StatusCode::OK);

    let unauthorized = send(
        &router,
        Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header("content-type", "application/json")
            .body(Body::from(json!({"input":"hi"}).to_string()))
            .expect("request"),
    )
    .await;
    assert_eq!(unauthorized.status(), StatusCode::UNAUTHORIZED);

    let created = send(
        &router,
        Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header("authorization", "Bearer proxy-secret")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({"input":"hello","store":true}).to_string(),
            ))
            .expect("request"),
    )
    .await;
    let created_body = created
        .into_body()
        .collect()
        .await
        .expect("collect")
        .to_bytes();
    let created_payload: Value = serde_json::from_slice(&created_body).expect("json");
    let response_id = created_payload["id"].as_str().expect("id").to_owned();

    let get_response = send(
        &router,
        Request::builder()
            .method("GET")
            .uri(format!("/v1/responses/{response_id}"))
            .header("authorization", "Bearer proxy-secret")
            .body(Body::empty())
            .expect("request"),
    )
    .await;
    assert_eq!(get_response.status(), StatusCode::OK);

    let items = send(
        &router,
        Request::builder()
            .method("GET")
            .uri(format!("/v1/responses/{response_id}/input_items"))
            .header("authorization", "Bearer proxy-secret")
            .body(Body::empty())
            .expect("request"),
    )
    .await;
    let items_body = items
        .into_body()
        .collect()
        .await
        .expect("collect")
        .to_bytes();
    let items_payload: Value = serde_json::from_slice(&items_body).expect("json");
    assert_eq!(items_payload["object"], "list");
    assert_eq!(items_payload["data"][0]["role"], "user");

    let deleted = send(
        &router,
        Request::builder()
            .method("DELETE")
            .uri(format!("/v1/responses/{response_id}"))
            .header("authorization", "Bearer proxy-secret")
            .body(Body::empty())
            .expect("request"),
    )
    .await;
    assert_eq!(deleted.status(), StatusCode::OK);

    for uri in [
        format!("/v1/responses/{response_id}"),
        format!("/v1/responses/{response_id}/input_items"),
        format!("/v1/responses/{response_id}/cancel"),
    ] {
        let method = if uri.ends_with("/cancel") {
            "POST"
        } else {
            "GET"
        };
        let response = send(
            &router,
            Request::builder()
                .method(method)
                .uri(uri)
                .header("authorization", "Bearer proxy-secret")
                .body(Body::empty())
                .expect("request"),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }
}

#[tokio::test]
async fn previous_response_id_and_strict_protocol_behave() {
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let calls_for_handler = calls.clone();
    let upstream = Router::new().route(
        "/v1/chat/completions",
        post(move |Json(payload): Json<Value>| {
            let calls = calls_for_handler.clone();
            async move {
                let call = calls.fetch_add(1, Ordering::SeqCst);
                let messages = payload["messages"].as_array().expect("messages");
                if call == 0 {
                    assert_eq!(messages.len(), 1);
                    assert_eq!(messages[0]["content"], "seed");
                } else {
                    assert_eq!(messages.len(), 3);
                    assert_eq!(messages[0]["content"], "seed");
                    assert_eq!(messages[1]["role"], "assistant");
                    assert_eq!(messages[2]["content"], "follow up");
                }
                Json(json!({
                    "choices": [{
                        "message": {
                            "content": "next",
                            "tool_calls": [{
                                "id":"call_1",
                                "function":{"name":"lookup","arguments":"{}"}
                            }]
                        }
                    }]
                }))
            }
        }),
    );
    let upstream_base_url = spawn_upstream_with_router(upstream).await;
    let router = build_router(settings(upstream_base_url.clone(), false), caps(false));

    let first = send(
        &router,
        Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header("authorization", "Bearer proxy-secret")
            .header("content-type", "application/json")
            .body(Body::from(json!({"input":"seed"}).to_string()))
            .expect("request"),
    )
    .await;
    let first_body = first
        .into_body()
        .collect()
        .await
        .expect("collect")
        .to_bytes();
    let first_payload: Value = serde_json::from_slice(&first_body).expect("json");
    let response_id = first_payload["id"].as_str().expect("id");

    let second = send(
        &router,
        Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header("authorization", "Bearer proxy-secret")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({"input":"follow up","previous_response_id":response_id}).to_string(),
            ))
            .expect("request"),
    )
    .await;
    assert_eq!(second.status(), StatusCode::OK);

    let missing_previous = send(
        &router,
        Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header("authorization", "Bearer proxy-secret")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({"input":"x","previous_response_id":"missing"}).to_string(),
            ))
            .expect("request"),
    )
    .await;
    assert_eq!(missing_previous.status(), StatusCode::BAD_REQUEST);

    let mut strict = settings(upstream_base_url, false);
    strict.strict_protocol = true;
    let strict_router = build_router(strict, caps(false));
    let strict_response = send(
        &strict_router,
        Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header("authorization", "Bearer proxy-secret")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({"input":"x","background":true}).to_string(),
            ))
            .expect("request"),
    )
    .await;
    assert_eq!(strict_response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn stream_previous_response_id_chains_messages() {
    let upstream = Router::new().route(
            "/v1/chat/completions",
            post(move |Json(payload): Json<Value>| {
                async move {
                    let is_stream = payload.get("stream").and_then(Value::as_bool) == Some(true);
                    if is_stream {
                        // First request: streaming, returns "seed answer"
                        let stream = stream! {
                            yield Ok::<Event, Infallible>(Event::default().data(
                                r#"{"choices":[{"delta":{"content":"seed answer"}}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}"#
                            ));
                            yield Ok::<Event, Infallible>(Event::default().data("[DONE]"));
                        };
                        Sse::new(stream).into_response()
                    } else {
                        // Second request: non-stream, verify chained messages
                        let messages = payload["messages"].as_array().expect("messages");
                        assert_eq!(messages.len(), 3, "expected 3 messages (seed user + assistant + follow up), got {}", messages.len());
                        assert_eq!(messages[0]["role"], "user");
                        assert_eq!(messages[0]["content"], "seed");
                        assert_eq!(messages[1]["role"], "assistant");
                        assert_eq!(messages[2]["role"], "user");
                        assert_eq!(messages[2]["content"], "follow up");
                        Json(json!({
                            "choices": [{
                                "message": {
                                    "role": "assistant",
                                    "content": "chained reply"
                                },
                                "finish_reason": "stop"
                            }],
                            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}
                        }))
                        .into_response()
                    }
                }
            }),
        );
    let upstream_base_url = spawn_upstream_with_router(upstream).await;
    let router = build_router(settings(upstream_base_url, false), caps(false));

    // Step 1: streaming request with store: true
    let stream_response = send(
        &router,
        Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header("authorization", "Bearer proxy-secret")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({"input":"seed","stream":true,"store":true}).to_string(),
            ))
            .expect("request"),
    )
    .await;
    assert_eq!(stream_response.status(), StatusCode::OK);

    // Parse SSE to extract response_id from response.completed event
    let body = String::from_utf8(
        stream_response
            .into_body()
            .collect()
            .await
            .expect("collect")
            .to_bytes()
            .to_vec(),
    )
    .expect("utf8");
    assert!(
        body.contains("event: response.completed"),
        "stream should complete"
    );

    // Extract response_id from the response.completed data line
    let response_id = body
        .lines()
        .filter(|line| line.starts_with("data: "))
        .filter_map(|line| {
            let data = &line[6..];
            let v: Value = serde_json::from_str(data).ok()?;
            let resp = v.get("response")?;
            if v.get("type").and_then(Value::as_str) == Some("response.completed") {
                resp.get("id").and_then(Value::as_str).map(str::to_owned)
            } else {
                None
            }
        })
        .next()
        .expect("should find response_id in stream");
    assert!(
        response_id.starts_with("resp_"),
        "response_id should be valid"
    );

    // Step 2: follow-up non-stream request using previous_response_id
    let follow_up = send(
        &router,
        Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header("authorization", "Bearer proxy-secret")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({"input":"follow up","previous_response_id":response_id}).to_string(),
            ))
            .expect("request"),
    )
    .await;
    assert_eq!(follow_up.status(), StatusCode::OK);
    let follow_body = follow_up
        .into_body()
        .collect()
        .await
        .expect("collect")
        .to_bytes();
    let follow_payload: Value = serde_json::from_slice(&follow_body).expect("json");
    assert_eq!(follow_payload["output_text"], "chained reply");
}

#[tokio::test]
async fn tool_execution_round_trip_is_preserved_across_requests() {
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let calls_for_handler = calls.clone();
    let upstream = Router::new().route(
            "/v1/chat/completions",
            post(move |Json(payload): Json<Value>| {
                let calls = calls_for_handler.clone();
                async move {
                    let call = calls.fetch_add(1, Ordering::SeqCst);
                    let messages = payload["messages"].as_array().expect("messages");
                    if call == 0 {
                        assert_eq!(messages.len(), 1);
                        assert_eq!(messages[0]["role"], "user");
                        assert_eq!(messages[0]["content"], "weather?");
                        Json(json!({
                            "choices": [{
                                "message": {
                                    "role": "assistant",
                                    "reasoning_content": "Check weather.",
                                    "content": null,
                                    "tool_calls": [{
                                        "id":"call_weather_1",
                                        "type":"function",
                                        "function":{"name":"get_weather","arguments":"{\"city\":\"Shanghai\"}"}
                                    }]
                                },
                                "finish_reason": "tool_calls"
                            }]
                        }))
                    } else {
                        assert_eq!(messages.len(), 3);
                        assert_eq!(messages[1]["role"], "assistant");
                        assert_eq!(messages[1]["tool_calls"][0]["id"], "call_weather_1");
                        assert_eq!(messages[1]["reasoning_content"], "Check weather.");
                        assert_eq!(messages[2]["role"], "tool");
                        assert_eq!(messages[2]["tool_call_id"], "call_weather_1");
                        assert_eq!(messages[2]["content"], "{\"temp\":26}");
                        Json(json!({
                            "choices": [{
                                "message": {
                                    "role": "assistant",
                                    "content": "26C"
                                },
                                "finish_reason": "stop"
                            }]
                        }))
                    }
                }
            }),
        );
    let upstream_base_url = spawn_upstream_with_router(upstream).await;
    let router = build_router(settings(upstream_base_url, false), caps(false));

    let first = send(
        &router,
        Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header("authorization", "Bearer proxy-secret")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({
                    "model":"gpt-5-codex",
                    "input":"weather?",
                    "tools":[{
                        "type":"function",
                        "name":"get_weather",
                        "parameters":{"type":"object","properties":{"city":{"type":"string"}}}
                    }]
                })
                .to_string(),
            ))
            .expect("request"),
    )
    .await;
    assert_eq!(first.status(), StatusCode::OK);
    let first_body = first
        .into_body()
        .collect()
        .await
        .expect("collect")
        .to_bytes();
    let first_payload: Value = serde_json::from_slice(&first_body).expect("json");
    let response_id = first_payload["id"].as_str().expect("id").to_owned();
    let function_call = first_payload["output"]
        .as_array()
        .expect("output")
        .iter()
        .find(|item| item["type"] == "function_call")
        .expect("function call item");
    assert_eq!(function_call["call_id"], "call_weather_1");

    let second = send(
        &router,
        Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header("authorization", "Bearer proxy-secret")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({
                    "model":"gpt-5-codex",
                    "previous_response_id": response_id,
                    "input":[{
                        "type":"function_call_output",
                        "call_id":"call_weather_1",
                        "output":{"temp":26}
                    }]
                })
                .to_string(),
            ))
            .expect("request"),
    )
    .await;
    assert_eq!(second.status(), StatusCode::OK);
    let second_body = second
        .into_body()
        .collect()
        .await
        .expect("collect")
        .to_bytes();
    let second_payload: Value = serde_json::from_slice(&second_body).expect("json");
    assert_eq!(second_payload["output_text"], "26C");
}

#[tokio::test]
async fn local_tool_execution_can_complete_non_stream_request() {
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let calls_for_handler = calls.clone();
    let upstream = Router::new().route(
            "/v1/chat/completions",
            post(move |Json(payload): Json<Value>| {
                let calls = calls_for_handler.clone();
                async move {
                    let call = calls.fetch_add(1, Ordering::SeqCst);
                    let messages = payload["messages"].as_array().expect("messages");
                    if call == 0 {
                        assert_eq!(messages.len(), 1);
                        Json(json!({
                            "choices": [{
                                "message": {
                                    "role": "assistant",
                                    "reasoning_content": "Call local tool.",
                                    "content": null,
                                    "tool_calls": [{
                                        "id":"call_echo_1",
                                        "type":"function",
                                        "function":{"name":"echo_json","arguments":"{\"city\":\"Shanghai\"}"}
                                    }]
                                },
                                "finish_reason": "tool_calls"
                            }]
                        }))
                    } else {
                        assert_eq!(messages.len(), 3);
                        assert_eq!(messages[1]["reasoning_content"], "Call local tool.");
                        assert_eq!(messages[2]["role"], "tool");
                        let tool_output: Value =
                            serde_json::from_str(messages[2]["content"].as_str().expect("content"))
                                .expect("tool output");
                        assert_eq!(tool_output["arguments"]["city"], "Shanghai");
                        Json(json!({
                            "choices": [{
                                "message": {
                                    "role": "assistant",
                                    "content": "tool-finished"
                                },
                                "finish_reason": "stop"
                            }],
                            "usage": {
                                "prompt_tokens": 4,
                                "completion_tokens": 2,
                                "total_tokens": 6
                            }
                        }))
                    }
                }
            }),
        );
    let upstream_base_url = spawn_upstream_with_router(upstream).await;
    let mut settings = settings(upstream_base_url, false);
    settings.local_tools.insert(
        "echo_json".to_owned(),
        crate::config::LocalToolSettings {
            command: "sh".to_owned(),
            args: vec!["-lc".to_owned(), "cat".to_owned()],
            env: Default::default(),
            workdir: None,
            timeout_seconds: 5.0,
            stdin_json: true,
            output_json: true,
        },
    );
    let router = build_router(settings, caps(false));

    let response = send(
        &router,
        Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header("authorization", "Bearer proxy-secret")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({
                    "model":"gpt-5-codex",
                    "input":"run local tool",
                    "tools":[{
                        "type":"function",
                        "name":"echo_json",
                        "parameters":{"type":"object","properties":{"city":{"type":"string"}}}
                    }]
                })
                .to_string(),
            ))
            .expect("request"),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = response
        .into_body()
        .collect()
        .await
        .expect("collect")
        .to_bytes();
    let payload: Value = serde_json::from_slice(&body).expect("json");
    assert_eq!(payload["output_text"], "tool-finished");
    assert_eq!(payload["usage"]["total_tokens"], 6);
}

#[tokio::test]
async fn local_tool_execution_can_complete_stream_request() {
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let calls_for_handler = calls.clone();
    let upstream = Router::new().route(
            "/v1/chat/completions",
            post(move |Json(_payload): Json<Value>| {
                let calls = calls_for_handler.clone();
                async move {
                    let call = calls.fetch_add(1, Ordering::SeqCst);
                    let stream = stream! {
                        if call == 0 {
                            yield Ok::<Event, Infallible>(Event::default().data(r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_echo_stream_1","function":{"name":"echo_json","arguments":"{\"city\":\"Shanghai\"}"}}]}}]}"#));
                            yield Ok::<Event, Infallible>(Event::default().data("[DONE]"));
                        } else {
                            yield Ok::<Event, Infallible>(Event::default().data(r#"{"choices":[{"delta":{"content":"stream-"}}]}"#));
                            yield Ok::<Event, Infallible>(Event::default().data(r#"{"choices":[{"delta":{"content":"done"}}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}"#));
                            yield Ok::<Event, Infallible>(Event::default().data("[DONE]"));
                        }
                    };
                    Sse::new(stream).into_response()
                }
            }),
        );
    let upstream_base_url = spawn_upstream_with_router(upstream).await;
    let mut settings = settings(upstream_base_url, false);
    settings.local_tools.insert(
        "echo_json".to_owned(),
        crate::config::LocalToolSettings {
            command: "sh".to_owned(),
            args: vec!["-lc".to_owned(), "cat".to_owned()],
            env: Default::default(),
            workdir: None,
            timeout_seconds: 5.0,
            stdin_json: true,
            output_json: true,
        },
    );
    let router = build_router(settings, caps(false));

    let response = send(
        &router,
        Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header("authorization", "Bearer proxy-secret")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({
                    "model":"gpt-5-codex",
                    "input":"run local tool stream",
                    "stream": true,
                    "tools":[{
                        "type":"function",
                        "name":"echo_json",
                        "parameters":{"type":"object","properties":{"city":{"type":"string"}}}
                    }]
                })
                .to_string(),
            ))
            .expect("request"),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = String::from_utf8(
        response
            .into_body()
            .collect()
            .await
            .expect("collect")
            .to_bytes()
            .to_vec(),
    )
    .expect("utf8");
    assert!(body.contains("event: response.created"));
    assert!(body.contains("event: response.output_text.done"));
    assert!(body.contains("stream-done"));
    assert!(body.contains("event: response.completed"));
    assert!(!body.contains("call_echo_stream_1"));
}

#[tokio::test]
async fn streaming_failure_paths_are_reported() {
    let parse_router = Router::new().route(
        "/v1/chat/completions",
        post(|| async { "data: {not-json}\n\n" }),
    );
    let parse_router = build_router(
        settings(spawn_upstream_with_router(parse_router).await, false),
        caps(false),
    );
    let parse_response = send(
        &parse_router,
        Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header("authorization", "Bearer proxy-secret")
            .header("content-type", "application/json")
            .body(Body::from(json!({"input":"x","stream":true}).to_string()))
            .expect("request"),
    )
    .await;
    let parse_body = String::from_utf8(
        parse_response
            .into_body()
            .collect()
            .await
            .expect("collect")
            .to_bytes()
            .to_vec(),
    )
    .expect("utf8");
    assert!(parse_body.contains("event: response.failed"));

    let bad_json_router = Router::new().route(
        "/v1/chat/completions",
        post(|| async { Json(json!({"not_choices":true})) }),
    );
    let bad_json_router = build_router(
        settings(spawn_upstream_with_router(bad_json_router).await, false),
        caps(false),
    );
    let bad_json_response = send(
        &bad_json_router,
        Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header("authorization", "Bearer proxy-secret")
            .header("content-type", "application/json")
            .body(Body::from(json!({"input":"x"}).to_string()))
            .expect("request"),
    )
    .await;
    assert_eq!(bad_json_response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn internal_helpers_cover_remaining_branches() {
    let inflight = InflightRegistry::default();
    assert!(!inflight.cancel("missing").await);
    let token = inflight.start("resp_1".to_owned()).await;
    assert!(!token.load(Ordering::SeqCst));
    assert!(inflight.cancel("resp_1").await);
    assert!(token.load(Ordering::SeqCst));
    inflight.finish("resp_1").await;
    assert!(!inflight.cancel("resp_1").await);

    let mut sequence = 0;
    let _event = json_event("x", &mut sequence, json!({"type":"x"}));
    assert_eq!(sequence, 1);

    let turn = AssistantTurn {
        reasoning: "r".to_owned(),
        text: "t".to_owned(),
        tool_calls: vec![ToolCall {
            call_id: "call_1".to_owned(),
            name: "lookup".to_owned(),
            arguments: "{}".to_owned(),
        }],
    };
    let context = RequestContext {
        response_id: "resp_1".to_owned(),
        reasoning_id: "rs_1".to_owned(),
        message_id: "msg_1".to_owned(),
        created_at: 1,
        client_model: "m".to_owned(),
        upstream_model: "u".to_owned(),
        stream: true,
        store: true,
        parallel_tool_calls: true,
        instructions: None,
        metadata: json!({}),
        tool_choice: json!("auto"),
        tools: vec![],
        max_output_tokens: None,
        max_tool_calls: None,
        hosted_output_items: Vec::new(),
        previous_response_id: None,
        reasoning_effort: None,
        reasoning_summary: None,
        truncation: "auto".to_owned(),
        include: Vec::new(),
        temperature: None,
        top_p: None,
        skip_reasoning_output: false,
    };
    let events = finalize_stream_items(&turn, &context, &mut 0);
    assert_eq!(events.len(), 7);

    let mut stream_state = StreamState::new();
    assert!(stream_state.has_message());
    let delta_events = apply_stream_delta(
        &mut stream_state,
        &json!({
            "choices": [{
                "delta": {
                    "reasoning_summary": [{"text":"a"}],
                    "content": [{"text":"b"}],
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_1",
                        "function": {"name":"lookup","arguments":"{"}
                    },{
                        "index": 0,
                        "function": {"arguments":"}"}
                    }]
                }
            }]
        }),
        &context,
        &mut 0,
    )
    .expect("delta");
    assert!(!delta_events.is_empty());
    let built_turn = stream_state.take_turn();
    assert_eq!(built_turn.reasoning, "a");
    assert_eq!(built_turn.text, "b");
    assert_eq!(built_turn.tool_calls[0].arguments, "{}");

    let no_choice =
        apply_stream_delta(&mut StreamState::new(), &json!({}), &context, &mut 0).expect("empty");
    assert!(no_choice.is_empty());
    let no_delta = apply_stream_delta(
        &mut StreamState::new(),
        &json!({"choices":[{}]}),
        &context,
        &mut 0,
    )
    .expect("no delta");
    assert!(no_delta.is_empty());

    let assistant = assistant_turn_from_output(&[
        json!({"type":"reasoning","summary":[{"text":"r"}]}),
        json!({"type":"message","content":"hello"}),
        json!({"type":"function_call","call_id":"call_1","name":"lookup","arguments":"{}"}),
        json!({"type":"ignored"}),
    ])
    .expect("assistant");
    assert_eq!(assistant.reasoning, "r");
    assert_eq!(assistant.text, "hello");
    assert_eq!(assistant.tool_calls.len(), 1);
    assert!(
        assistant_turn_from_output(&[json!({"type":"function_call","name":"lookup"})]).is_err()
    );
    assert!(
        assistant_turn_from_output(&[json!({"type":"function_call","call_id":"call_1"})]).is_err()
    );

    let response = json!({
        "id":"resp_1",
        "created_at":1,
        "model":"m",
        "store":false,
        "parallel_tool_calls":false,
        "instructions":"i",
        "metadata":{"a":1},
        "tool_choice":"none",
        "tools":[{"type":"function"}],
        "max_output_tokens":12,
        "output":[
            {"id":"rs_1","type":"reasoning"},
            {"id":"msg_1","type":"message"}
        ]
    });
    let restored = context_from_response(&response).expect("context");
    assert_eq!(restored.reasoning_id, "rs_1");
    assert_eq!(restored.message_id, "msg_1");
    assert!(!restored.store);
    assert!(!restored.parallel_tool_calls);
    assert_eq!(restored.instructions.as_deref(), Some("i"));
    assert_eq!(restored.max_output_tokens, Some(12));

    let restored_defaults = context_from_response(&json!({
        "id":"resp_2",
        "created_at":2,
        "output":[]
    }))
    .expect("context");
    assert_eq!(restored_defaults.reasoning_id, "rs_cancelled");
    assert_eq!(restored_defaults.message_id, "msg_cancelled");
    assert_eq!(restored_defaults.tool_choice, json!("auto"));

    assert!(context_from_response(&json!({"created_at":1})).is_err());
    assert!(context_from_response(&json!({"id":"x"})).is_err());

    let mut parser = SseParser::default();
    assert_eq!(parser.push("data: first\n\n").len(), 1);
    assert!(parser.push("data: second").is_empty());
    assert_eq!(parser.push("\n\n").len(), 1);
    assert_eq!(parse_sse_event("event: x\n\n"), None);
    assert_eq!(
        parse_sse_event("data: a\ndata: b\n\n").as_deref(),
        Some("a\nb")
    );

    let store = ResponseStore::default();
    let translated = crate::translate::TranslatedRequest {
        upstream_payload: json!({}),
        context: RequestContext {
            store: false,
            ..context.clone()
        },
        input_items: vec![json!({"type":"message"})],
        request_messages: vec![json!({"role":"user","content":"hi"})],
    };
    store_final_response(&store, &translated, json!({"id":"resp_1"}))
        .await
        .expect("store");
    assert!(store.get("resp_1").await.expect("get").is_none());

    let translated = crate::translate::TranslatedRequest {
        upstream_payload: json!({}),
        context,
        input_items: vec![json!({"type":"message"})],
        request_messages: vec![json!({"role":"user","content":"hi"})],
    };
    store_final_response(&store, &translated, json!({"id":"resp_1"}))
        .await
        .expect("store");
    assert!(store.get("resp_1").await.expect("get").is_some());

    assert!(
        authorize(
            &settings("http://127.0.0.1".to_owned(), false),
            &HeaderMap::new()
        )
        .is_err()
    );
    let mut wrong = HeaderMap::new();
    wrong.insert("authorization", "Token nope".parse().expect("header"));
    assert!(authorize(&settings("http://127.0.0.1".to_owned(), false), &wrong).is_err());
}

#[tokio::test]
async fn direct_handler_and_helper_edge_cases_are_covered() {
    let upstream_base_url = spawn_upstream().await;
    let mut strict = settings(upstream_base_url.clone(), false);
    strict.strict_protocol = true;
    let strict_router = build_router(strict, caps(false));

    let supported_in_strict = send(
        &strict_router,
        Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header("authorization", "Bearer proxy-secret")
            .header("content-type", "application/json")
            .body(Body::from(json!({"input":"ok","store":false}).to_string()))
            .expect("request"),
    )
    .await;
    assert_eq!(supported_in_strict.status(), StatusCode::OK);

    let non_object = send(
        &strict_router,
        Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header("authorization", "Bearer proxy-secret")
            .header("content-type", "application/json")
            .body(Body::from("[]"))
            .expect("request"),
    )
    .await;
    assert_eq!(non_object.status(), StatusCode::BAD_REQUEST);

    let router = build_router(settings(upstream_base_url, false), caps(false));
    let missing_delete = send(
        &router,
        Request::builder()
            .method("DELETE")
            .uri("/v1/responses/missing")
            .header("authorization", "Bearer proxy-secret")
            .body(Body::empty())
            .expect("request"),
    )
    .await;
    assert_eq!(missing_delete.status(), StatusCode::BAD_REQUEST);

    let app_state = AppState {
        settings: Arc::new(settings("http://127.0.0.1:1".to_owned(), false)),
        capabilities: caps(false),
        upstream: UpstreamClient::new(Arc::new(settings("http://127.0.0.1:1".to_owned(), false)))
            .expect("client"),
        http_client: reqwest::Client::new(),
        tool_executor: ToolExecutor::new(Arc::new(settings(
            "http://127.0.0.1:1".to_owned(),
            false,
        ))),
        store: ResponseStore::default(),
        inflight: InflightRegistry::default(),
        metrics: Arc::new(RequestMetrics::default()),
    };
    app_state
        .store
        .put(
            "resp_in_progress".to_owned(),
            StoredResponse {
                response: json!({
                    "id":"resp_in_progress",
                    "created_at":1,
                    "status":"in_progress",
                    "model":"m",
                    "store":true,
                    "parallel_tool_calls":true,
                    "metadata":{},
                    "tool_choice":"auto",
                    "tools":[],
                    "output":[
                        {"id":"rs_1","type":"reasoning","summary":[{"text":"r"}]},
                        {"id":"msg_1","type":"message","content":"t"},
                        {"type":"function_call","call_id":"call_1","name":"lookup","arguments":"{}"}
                    ]
                }),
                input_items: vec![],
                request_messages: vec![json!({"role":"user","content":"hi"})],
                inserted_at: std::time::Instant::now(),
            },
        )
        .await
        .expect("put");
    let mut headers = HeaderMap::new();
    headers.insert(
        "authorization",
        "Bearer proxy-secret".parse().expect("header"),
    );
    let cancelled = cancel_response(
        State(app_state.clone()),
        Path("resp_in_progress".to_owned()),
        headers.clone(),
    )
    .await
    .expect("cancel");
    assert_eq!(cancelled.status(), StatusCode::OK);
    let stored = app_state
        .store
        .get("resp_in_progress")
        .await
        .expect("get")
        .expect("stored");
    assert_eq!(stored.response["status"], "cancelled");

    app_state
        .store
        .put(
            "resp_in_progress_empty".to_owned(),
            StoredResponse {
                response: json!({
                    "id":"resp_in_progress_empty",
                    "created_at":1,
                    "status":"in_progress",
                    "output":[]
                }),
                input_items: vec![],
                request_messages: vec![],
                inserted_at: std::time::Instant::now(),
            },
        )
        .await
        .expect("put");
    let empty_cancelled = cancel_response(
        State(app_state.clone()),
        Path("resp_in_progress_empty".to_owned()),
        headers.clone(),
    )
    .await
    .expect("cancel");
    assert_eq!(empty_cancelled.status(), StatusCode::OK);

    let no_output_store = ResponseStore::default();
    no_output_store
        .put(
            "resp_prev".to_owned(),
            StoredResponse {
                response: json!({"id":"resp_prev"}),
                input_items: vec![],
                request_messages: vec![json!({"role":"user","content":"hi"})],
                inserted_at: std::time::Instant::now(),
            },
        )
        .await
        .expect("put");
    let previous = load_previous_messages(
        &no_output_store,
        json!({"previous_response_id":"resp_prev"})
            .as_object()
            .expect("object"),
    )
    .await
    .expect("previous");
    assert_eq!(previous.len(), 1);

    let mut wrong_headers = HeaderMap::new();
    wrong_headers.insert(
        "authorization",
        "Bearer definitely-wrong".parse().expect("header"),
    );
    assert!(
        authorize(
            &settings("http://127.0.0.1".to_owned(), false),
            &wrong_headers
        )
        .is_err()
    );

    let mut state = StreamState::new();
    let context = RequestContext {
        response_id: "resp_1".to_owned(),
        reasoning_id: "rs_1".to_owned(),
        message_id: "msg_1".to_owned(),
        created_at: 1,
        client_model: "m".to_owned(),
        upstream_model: "u".to_owned(),
        stream: true,
        store: true,
        parallel_tool_calls: true,
        instructions: None,
        metadata: json!({}),
        tool_choice: json!("auto"),
        tools: vec![],
        max_output_tokens: None,
        max_tool_calls: None,
        hosted_output_items: Vec::new(),
        previous_response_id: None,
        reasoning_effort: None,
        reasoning_summary: None,
        truncation: "auto".to_owned(),
        include: Vec::new(),
        temperature: None,
        top_p: None,
        skip_reasoning_output: false,
    };
    let events = apply_stream_delta(
        &mut state,
        &json!({
            "choices": [{
                "delta": {
                    "reasoning": "",
                    "content": 1,
                    "tool_calls": [{"id":"call_2"}]
                }
            }]
        }),
        &context,
        &mut 0,
    )
    .expect("delta");
    assert!(events.is_empty());

    let events = apply_stream_delta(
        &mut state,
        &json!({
            "choices": [{
                "delta": {
                    "reasoning": "x",
                    "tool_calls": [{
                        "id":"call_3",
                        "function":{"name":"lookup","arguments":{"x":1}}
                    },{
                        "id":"call_4",
                        "function":{"name":"noop"}
                    }]
                }
            }]
        }),
        &context,
        &mut 0,
    )
    .expect("delta");
    assert!(!events.is_empty());

    let parsed = assistant_turn_from_output(&[json!({"type":"message"})]).expect("turn");
    assert_eq!(parsed.text, "");

    let restored = context_from_response(&json!({
        "id":"resp_3",
        "created_at":3,
        "output":[{"type":"message","id":"msg_3"}]
    }))
    .expect("context");
    assert_eq!(restored.reasoning_id, "rs_cancelled");
}
