mod common;

use axum::{body::Body, http::Request};
use http::StatusCode;
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tower::ServiceExt;

use common::{PROXY_KEY, send, shared_router};

#[tokio::test]
async fn stats_endpoint_returns_json() {
    require_api_key!();
    let Some(router) = shared_router().await else {
        return;
    };

    let response = send(
        router,
        Request::builder()
            .uri("/stats")
            .header("authorization", format!("Bearer {PROXY_KEY}"))
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

    // Check top-level fields exist
    assert!(
        body["uptime_seconds"].as_u64().is_some(),
        "should have uptime_seconds"
    );
    assert!(
        body["stored_responses"].as_u64().is_some(),
        "should have stored_responses"
    );

    // Check requests section
    let req = &body["requests"];
    assert!(
        req["total"].as_u64().is_some(),
        "should have requests.total"
    );
    assert!(
        req["stream"].as_u64().is_some(),
        "should have requests.stream"
    );
    assert!(
        req["non_stream"].as_u64().is_some(),
        "should have requests.non_stream"
    );
    assert!(
        req["completed"].as_u64().is_some(),
        "should have requests.completed"
    );
    assert!(
        req["failed"].as_u64().is_some(),
        "should have requests.failed"
    );
    assert!(
        req["cancelled"].as_u64().is_some(),
        "should have requests.cancelled"
    );
    assert!(
        req["inflight"].as_u64().is_some(),
        "should have requests.inflight"
    );

    // Check tokens section
    let tok = &body["tokens"];
    assert!(tok["input"].as_u64().is_some(), "should have tokens.input");
    assert!(
        tok["output"].as_u64().is_some(),
        "should have tokens.output"
    );
    assert!(
        tok["cached"].as_u64().is_some(),
        "should have tokens.cached"
    );
    assert!(
        tok["reasoning"].as_u64().is_some(),
        "should have tokens.reasoning"
    );
}

#[tokio::test]
async fn stats_reflects_request_count() {
    require_api_key!();
    let Some(router) = shared_router().await else {
        return;
    };

    // Get initial stats
    let before = get_stats(router).await;
    let before_total = before["requests"]["total"].as_u64().unwrap();

    // Make a request
    common::send_json(
        router,
        json!({
            "model": common::MIMO_MODEL,
            "input": "Say hi",
            "stream": false,
            "store": false,
        }),
    )
    .await;

    // Get stats after
    let after = get_stats(router).await;
    let after_total = after["requests"]["total"].as_u64().unwrap();

    assert!(
        after_total > before_total,
        "total requests should increase: before={before_total}, after={after_total}"
    );
    assert!(
        after["requests"]["completed"].as_u64().unwrap()
            > before["requests"]["completed"].as_u64().unwrap(),
        "completed count should increase"
    );
}

async fn get_stats(router: &axum::Router) -> Value {
    let response = send(
        router,
        Request::builder()
            .uri("/stats")
            .header("authorization", format!("Bearer {PROXY_KEY}"))
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
    serde_json::from_slice(&bytes).expect("json")
}
