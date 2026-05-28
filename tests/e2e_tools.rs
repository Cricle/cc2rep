mod common;

use serde_json::{Value, json};

use common::{
    MIMO_MODEL, calculator_tool, last_message_text, send_json, send_stream, shared_router,
    weather_tool,
};

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

#[tokio::test]
async fn drop_tools_ignores_tool_definitions() {
    require_api_key!();
    let _guard = common::rate_limit().await;
    let Some(mut settings) = common::mimo_settings() else {
        return;
    };
    settings.drop_tools = true;
    let caps = cc2rep::probe_upstream(&settings).await;
    let router = cc2rep::build_router(settings, caps);

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
    let output = body["output"].as_array().unwrap();
    let has_function_call = output.iter().any(|o| o["type"] == "function_call");
    assert!(
        !has_function_call,
        "should not have function_call when drop_tools is true"
    );
}
