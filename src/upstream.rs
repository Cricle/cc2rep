use std::{sync::Arc, time::Duration};

use reqwest::{
    Client, Response,
    header::{HeaderMap, HeaderName, HeaderValue},
};
use serde_json::Value;
use tracing::{info, warn};

use crate::{config::Settings, error::ProxyError};

#[derive(Clone)]
pub struct UpstreamClient {
    upstream_url: String,
    headers: HeaderMap,
    http: Client,
    max_retries: u32,
    retry_base_delay: Duration,
}

impl UpstreamClient {
    pub fn new(settings: Arc<Settings>) -> Result<Self, ProxyError> {
        let upstream_url = settings.upstream_url();
        let headers = build_headers(&settings)?;
        let http = Client::builder()
            .timeout(Duration::from_secs_f64(settings.request_timeout_seconds))
            .build()
            .map_err(|err| {
                ProxyError::invalid_config(format!("failed to build HTTP client: {err}"))
            })?;
        Ok(Self {
            upstream_url,
            headers,
            http,
            max_retries: settings.upstream_max_retries,
            retry_base_delay: Duration::from_millis(settings.upstream_retry_base_delay_ms),
        })
    }

    pub async fn chat_json(&self, payload: &Value) -> Result<Value, ProxyError> {
        let response = self.send_with_tool_choice_retry(payload).await?;
        response
            .json::<Value>()
            .await
            .map_err(|err| ProxyError::Transport(err.to_string()))
    }

    pub async fn chat_stream(&self, payload: &Value) -> Result<Response, ProxyError> {
        self.send_with_tool_choice_retry(payload).await
    }

    async fn send_with_tool_choice_retry(&self, payload: &Value) -> Result<Response, ProxyError> {
        match self.send(payload).await {
            Ok(response) => Ok(response),
            Err(ProxyError::Upstream { status, message })
                if status.as_u16() == 400
                    && (message.contains("tool_choice") || message.contains("tool choice")) =>
            {
                warn!("tool_choice not supported, retrying without it");
                let mut retry_payload = payload.clone();
                if let Some(obj) = retry_payload.as_object_mut() {
                    obj.remove("tool_choice");
                }
                self.send(&retry_payload).await
            }
            Err(err) => Err(err),
        }
    }

    async fn send(&self, payload: &Value) -> Result<Response, ProxyError> {
        let model = payload.get("model").and_then(Value::as_str).unwrap_or("");
        let mut last_err = None;

        for attempt in 0..=self.max_retries {
            if attempt > 0 {
                let delay = self.retry_base_delay * 2u32.pow(attempt - 1);
                warn!(
                    attempt = attempt,
                    max_retries = self.max_retries,
                    delay_ms = delay.as_millis() as u64,
                    "retrying upstream request after 429"
                );
                tokio::time::sleep(delay).await;
            }

            info!(
                upstream_url = %self.upstream_url,
                model = %model,
                attempt = attempt,
                "sending request to upstream"
            );
            let request = self
                .http
                .post(&self.upstream_url)
                .headers(self.headers.clone())
                .json(payload);

            let response = request.send().await.map_err(map_reqwest_error)?;
            let status = response.status();

            if status.as_u16() == 429 && attempt < self.max_retries {
                let body = response
                    .text()
                    .await
                    .unwrap_or_else(|_| "rate limited".to_owned());
                warn!(
                    upstream_status = %status,
                    attempt = attempt,
                    "upstream rate limited, will retry"
                );
                last_err = Some(ProxyError::from_upstream_body(status, body));
                continue;
            }

            if !status.is_success() {
                let body = response
                    .text()
                    .await
                    .unwrap_or_else(|_| "failed to read upstream error body".to_owned());
                warn!(
                    upstream_status = %status,
                    upstream_error_body = %body,
                    "upstream request failed"
                );
                return Err(ProxyError::from_upstream_body(status, body));
            }

            info!(upstream_status = %status, "upstream request succeeded");
            return Ok(response);
        }

        Err(last_err.unwrap_or_else(|| ProxyError::Transport("max retries exceeded".to_owned())))
    }
}

fn build_headers(settings: &Settings) -> Result<HeaderMap, ProxyError> {
    let mut headers = HeaderMap::new();
    let auth_name: HeaderName = settings
        .upstream_api_key_header_name
        .parse()
        .map_err(|err| {
            ProxyError::invalid_config(format!("invalid upstream_api_key_header_name: {err}"))
        })?;
    let auth_value = format!(
        "{}{}",
        settings.upstream_api_key_prefix, settings.upstream_api_key
    );
    headers.insert(
        auth_name,
        HeaderValue::from_str(&auth_value).map_err(|err| {
            ProxyError::invalid_config(format!("invalid upstream auth header value: {err}"))
        })?,
    );

    for (name, value) in &settings.upstream_headers {
        let header_name: HeaderName = name.parse().map_err(|err| {
            ProxyError::invalid_config(format!("invalid upstream header `{name}`: {err}"))
        })?;
        let header_value = HeaderValue::from_str(value).map_err(|err| {
            ProxyError::invalid_config(format!("invalid upstream header value for `{name}`: {err}"))
        })?;
        headers.insert(header_name, header_value);
    }

    Ok(headers)
}

fn map_reqwest_error(error: reqwest::Error) -> ProxyError {
    if error.is_timeout() {
        ProxyError::Timeout(error.to_string())
    } else {
        ProxyError::Transport(error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use axum::{Json, Router, routing::post};
    use serde_json::json;

    use super::*;

    fn settings(base_url: String) -> Settings {
        Settings {
            proxy_host: "127.0.0.1".to_owned(),
            proxy_port: 8080,
            proxy_api_key: "proxy-key".to_owned(),
            upstream_base_url: base_url,
            upstream_chat_path: "/v1/chat/completions".to_owned(),
            upstream_model: "upstream-model".to_owned(),
            upstream_api_key: "upstream-key".to_owned(),
            upstream_headers: [("x-extra".to_owned(), "present".to_owned())]
                .into_iter()
                .collect(),
            upstream_api_key_header_name: "Authorization".to_owned(),
            upstream_api_key_prefix: "Bearer ".to_owned(),
            request_timeout_seconds: 0.1,
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
            upstream_retry_base_delay_ms: 100,
            upstream_reasoning_effort_field: "reasoning_effort".to_owned(),
            web_search_url: None,
            web_search_max_results: 5,
            file_search_paths: Vec::new(),
            file_search_max_results: 5,
        }
    }

    async fn spawn_server(router: Router) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("addr");
        drop(tokio::spawn(async move {
            let _ = axum::serve(listener, router).await;
        }));
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn build_headers_and_send_json_work() {
        let base_url = spawn_server(Router::new().route(
            "/v1/chat/completions",
            post(
                |headers: HeaderMap, Json(payload): Json<Value>| async move {
                    assert_eq!(headers["authorization"], "Bearer upstream-key");
                    assert_eq!(headers["x-extra"], "present");
                    assert_eq!(payload["hello"], "world");
                    Json(json!({"ok":true}))
                },
            ),
        ))
        .await;

        let client = UpstreamClient::new(Arc::new(settings(base_url))).expect("client");
        let payload = client
            .chat_json(&json!({"hello":"world"}))
            .await
            .expect("json");
        assert_eq!(payload["ok"], true);
    }

    #[tokio::test]
    async fn chat_stream_returns_success_response() {
        let base_url = spawn_server(Router::new().route(
            "/v1/chat/completions",
            post(|| async { "data: [DONE]\n\n" }),
        ))
        .await;

        let client = UpstreamClient::new(Arc::new(settings(base_url))).expect("client");
        let response = client
            .chat_stream(&json!({"model":"m","stream":true}))
            .await
            .expect("stream");
        assert_eq!(response.status(), http::StatusCode::OK);
    }

    #[tokio::test]
    async fn upstream_error_and_transport_and_invalid_headers_are_mapped() {
        let base_url = spawn_server(Router::new().route(
            "/v1/chat/completions",
            post(|| async {
                (
                    http::StatusCode::UNAUTHORIZED,
                    Json(json!({"error":{"message":"invalid token"}})),
                )
            }),
        ))
        .await;

        let client = UpstreamClient::new(Arc::new(settings(base_url))).expect("client");
        let error = client.chat_json(&json!({"x":1})).await.unwrap_err();
        let ProxyError::Upstream { status, message } = error else {
            panic!("expected upstream error");
        };
        assert_eq!(status, http::StatusCode::UNAUTHORIZED);
        assert_eq!(message, "invalid token");

        let transport = UpstreamClient::new(Arc::new(settings("http://127.0.0.1:1".to_owned())))
            .expect("client")
            .chat_json(&json!({"x":1}))
            .await
            .unwrap_err();
        assert!(matches!(transport, ProxyError::Transport(_)));

        let mut invalid_name = settings("http://127.0.0.1:1".to_owned());
        invalid_name.upstream_api_key_header_name = "bad header".to_owned();
        assert!(matches!(
            UpstreamClient::new(Arc::new(invalid_name)),
            Err(ProxyError::InvalidConfig(_))
        ));

        let mut invalid_value = settings("http://127.0.0.1:1".to_owned());
        invalid_value
            .upstream_headers
            .insert("x-bad".to_owned(), "\n".to_owned());
        assert!(matches!(
            UpstreamClient::new(Arc::new(invalid_value)),
            Err(ProxyError::InvalidConfig(_))
        ));
    }

    #[tokio::test]
    async fn remaining_upstream_error_paths_are_covered() {
        let base_url = spawn_server(
            Router::new().route("/v1/chat/completions", post(|| async { "not-json" })),
        )
        .await;
        let client = UpstreamClient::new(Arc::new(settings(base_url))).expect("client");
        let error = client.chat_json(&json!({"x":1})).await.unwrap_err();
        assert!(matches!(error, ProxyError::Transport(_)));

        let mut invalid_auth_value = settings("http://127.0.0.1:1".to_owned());
        invalid_auth_value.upstream_api_key_prefix = "Bearer \n".to_owned();
        assert!(matches!(
            UpstreamClient::new(Arc::new(invalid_auth_value)),
            Err(ProxyError::InvalidConfig(_))
        ));

        let mut invalid_header_name = settings("http://127.0.0.1:1".to_owned());
        invalid_header_name
            .upstream_headers
            .insert("bad name".to_owned(), "x".to_owned());
        assert!(matches!(
            UpstreamClient::new(Arc::new(invalid_header_name)),
            Err(ProxyError::InvalidConfig(_))
        ));

        let timeout = UpstreamClient::new(Arc::new(settings("http://10.255.255.1:81".to_owned())))
            .expect("client")
            .chat_json(&json!({"x":1}))
            .await
            .unwrap_err();
        assert!(matches!(
            timeout,
            ProxyError::Timeout(_) | ProxyError::Transport(_)
        ));
    }
}
