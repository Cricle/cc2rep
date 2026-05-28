mod common;

use serde_json::json;

use common::{MIMO_MODEL, send_json, send_stream, shared_router};

#[tokio::test]
async fn model_alias() {
    require_api_key!();
    let _guard = common::rate_limit().await;
    let Some(mut settings) = common::mimo_settings() else {
        return;
    };
    settings
        .model_aliases
        .insert("my-alias".to_owned(), MIMO_MODEL.to_owned());
    let caps = cc2rep::probe_upstream(&settings).await;
    let router = cc2rep::build_router(settings, caps);

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

#[tokio::test]
async fn local_tool_auto_execution() {
    require_api_key!();
    let _guard = common::rate_limit().await;
    let Some(mut settings) = common::mimo_settings() else {
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

    let caps = cc2rep::probe_upstream(&settings).await;
    let router = cc2rep::build_router(settings, caps);

    let body = send_json(
        &router,
        json!({
            "model": MIMO_MODEL,
            "input": "What is the weather in Beijing?",
            "tools": [common::weather_tool()],
            "tool_choice": "required",
            "stream": false
        }),
    )
    .await;

    assert_eq!(body["status"], "completed");
    let output = body["output"].as_array().unwrap();

    let message = output.iter().find(|o| o["type"] == "message");
    assert!(
        message.is_some(),
        "should have final message after tool execution"
    );

    let text = message.unwrap()["content"][0]["text"].as_str().unwrap();
    assert!(
        text.contains("22") || text.contains("Beijing") || text.contains("sunny"),
        "final answer should reference tool output: {text}"
    );
}
