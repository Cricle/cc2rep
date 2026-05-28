mod common;

use axum::{body::Body, http::Request};
use http::StatusCode;
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tower::ServiceExt;

use common::{PROXY_KEY, send, shared_router};

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
