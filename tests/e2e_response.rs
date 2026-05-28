mod common;

use axum::{body::Body, http::Request};
use common::{MIMO_MODEL, PROXY_KEY, send, send_json, send_json_expect_error, shared_router};
use http::StatusCode;
use http_body_util::BodyExt;
use serde_json::{Value, json};

#[tokio::test]
async fn response_get_and_delete() {
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

    // Get
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

    // Delete
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

    // Get after delete
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
