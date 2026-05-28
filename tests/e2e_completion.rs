mod common;

use serde_json::json;

use common::{MIMO_MODEL, last_message_text, send_json, send_stream, shared_router};

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
    assert_eq!(events[0]["type"], "response.created");
    assert_eq!(events[0]["response"]["status"], "in_progress");

    let last = events.last().unwrap();
    assert_eq!(last["type"], "response.completed");
    assert_eq!(last["response"]["status"], "completed");

    let has_text_delta = events
        .iter()
        .any(|e| e["type"] == "response.output_text.delta");
    assert!(has_text_delta, "should have text deltas");

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
    let reasoning = output.iter().find(|o| o["type"] == "reasoning");
    assert!(reasoning.is_some(), "should have reasoning output item");
    let reasoning = reasoning.unwrap();
    assert_eq!(reasoning["status"], "completed");
    let summary = reasoning["summary"][0]["text"].as_str().unwrap();
    assert!(!summary.is_empty(), "reasoning summary should not be empty");

    let message = output.iter().find(|o| o["type"] == "message");
    assert!(message.is_some(), "should have message output item");

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

    let has_reasoning = events
        .iter()
        .any(|e| e["type"] == "response.output_item.added" && e["item"]["type"] == "reasoning");
    assert!(has_reasoning, "should have reasoning item in stream");

    let has_reasoning_delta = events
        .iter()
        .any(|e| e["type"] == "response.reasoning_summary_text.delta");
    assert!(has_reasoning_delta, "should have reasoning summary deltas");
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
