use axum::{
    Json,
    response::{IntoResponse, Response},
};
use http::StatusCode;
use serde_json::json;
use thiserror::Error;
use tracing::warn;

#[derive(Debug, Error)]
pub enum ProxyError {
    #[error("{0}")]
    BadRequest(String),
    #[error("invalid proxy API key")]
    Unauthorized,
    #[error("{0}")]
    InvalidConfig(String),
    #[error("upstream returned {status}: {message}")]
    Upstream { status: StatusCode, message: String },
    #[error("request to upstream failed: {0}")]
    Transport(String),
    #[error("request to upstream timed out: {0}")]
    Timeout(String),
    #[error("internal error: {0}")]
    Internal(String),
}

impl ProxyError {
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self::BadRequest(message.into())
    }

    pub fn invalid_config(message: impl Into<String>) -> Self {
        Self::InvalidConfig(message.into())
    }

    pub fn from_upstream_body(status: StatusCode, body: String) -> Self {
        let message = serde_json::from_str::<serde_json::Value>(&body)
            .ok()
            .and_then(|value| {
                value
                    .get("error")
                    .and_then(|error| error.get("message"))
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned)
                    .or_else(|| {
                        value
                            .get("message")
                            .and_then(serde_json::Value::as_str)
                            .map(str::to_owned)
                    })
            })
            .unwrap_or_else(|| {
                let trimmed = body.trim();
                if trimmed.is_empty() {
                    format!("upstream returned HTTP {status}")
                } else {
                    trimmed.to_owned()
                }
            });
        Self::Upstream { status, message }
    }

    fn status_code(&self) -> StatusCode {
        match self {
            Self::BadRequest(_) => StatusCode::BAD_REQUEST,
            Self::Unauthorized => StatusCode::UNAUTHORIZED,
            Self::InvalidConfig(_) => StatusCode::INTERNAL_SERVER_ERROR,
            Self::Upstream { status, .. } => *status,
            Self::Transport(_) => StatusCode::BAD_GATEWAY,
            Self::Timeout(_) => StatusCode::GATEWAY_TIMEOUT,
            Self::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    fn error_type(&self) -> &'static str {
        match self {
            Self::BadRequest(_) => "invalid_request_error",
            Self::Unauthorized => "authentication_error",
            Self::InvalidConfig(_) | Self::Internal(_) => "server_error",
            Self::Upstream { .. } => "upstream_error",
            Self::Transport(_) | Self::Timeout(_) => "api_connection_error",
        }
    }

    fn code(&self) -> &'static str {
        match self {
            Self::BadRequest(_) => "bad_request",
            Self::Unauthorized => "invalid_api_key",
            Self::InvalidConfig(_) => "invalid_config",
            Self::Upstream { .. } => "upstream_error",
            Self::Transport(_) => "upstream_transport_error",
            Self::Timeout(_) => "upstream_timeout",
            Self::Internal(_) => "internal_error",
        }
    }
}

impl IntoResponse for ProxyError {
    fn into_response(self) -> Response {
        let status = self.status_code();
        let message = self.to_string();
        if status.is_client_error() && !matches!(self, Self::BadRequest(_)) {
            warn!(status = %status, error = %message, "request error");
        } else if status.is_server_error() {
            warn!(status = %status, error = %message, "server error");
        }
        let body = json!({
            "error": {
                "message": message,
                "type": self.error_type(),
                "code": self.code(),
            }
        });
        (status, Json(body)).into_response()
    }
}

#[cfg(test)]
mod tests {
    use http_body_util::BodyExt;

    use super::*;

    async fn response_body(error: ProxyError) -> (StatusCode, serde_json::Value) {
        let response = error.into_response();
        let status = response.status();
        let body = response
            .into_body()
            .collect()
            .await
            .expect("collect")
            .to_bytes();
        let json = serde_json::from_slice(&body).expect("json");
        (status, json)
    }

    #[tokio::test]
    async fn upstream_body_prefers_nested_error_message() {
        let error = ProxyError::from_upstream_body(
            StatusCode::BAD_REQUEST,
            r#"{"error":{"message":"bad upstream token"}}"#.to_owned(),
        );
        let (status, body) = response_body(error).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(
            body["error"]["message"],
            "upstream returned 400 Bad Request: bad upstream token"
        );
        assert_eq!(body["error"]["type"], "upstream_error");
        assert_eq!(body["error"]["code"], "upstream_error");
    }

    #[tokio::test]
    async fn upstream_body_falls_back_to_top_level_message_and_plain_text() {
        let top = ProxyError::from_upstream_body(
            StatusCode::BAD_GATEWAY,
            r#"{"message":"gateway broke"}"#.to_owned(),
        );
        let (status, body) = response_body(top).await;
        assert_eq!(status, StatusCode::BAD_GATEWAY);
        assert_eq!(
            body["error"]["message"],
            "upstream returned 502 Bad Gateway: gateway broke"
        );

        let text = ProxyError::from_upstream_body(StatusCode::BAD_GATEWAY, "oops".to_owned());
        let (_, body) = response_body(text).await;
        assert_eq!(
            body["error"]["message"],
            "upstream returned 502 Bad Gateway: oops"
        );

        let empty = ProxyError::from_upstream_body(StatusCode::BAD_GATEWAY, "   ".to_owned());
        let (_, body) = response_body(empty).await;
        assert_eq!(
            body["error"]["message"],
            "upstream returned 502 Bad Gateway: upstream returned HTTP 502 Bad Gateway"
        );
    }

    #[tokio::test]
    async fn error_variants_map_to_expected_status_type_and_code() {
        let cases = [
            (
                ProxyError::bad_request("bad"),
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "bad_request",
            ),
            (
                ProxyError::Unauthorized,
                StatusCode::UNAUTHORIZED,
                "authentication_error",
                "invalid_api_key",
            ),
            (
                ProxyError::invalid_config("bad config"),
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                "invalid_config",
            ),
            (
                ProxyError::Transport("transport".to_owned()),
                StatusCode::BAD_GATEWAY,
                "api_connection_error",
                "upstream_transport_error",
            ),
            (
                ProxyError::Timeout("timeout".to_owned()),
                StatusCode::GATEWAY_TIMEOUT,
                "api_connection_error",
                "upstream_timeout",
            ),
            (
                ProxyError::Internal("internal".to_owned()),
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                "internal_error",
            ),
        ];

        for (error, expected_status, expected_type, expected_code) in cases {
            let (status, body) = response_body(error).await;
            assert_eq!(status, expected_status);
            assert_eq!(body["error"]["type"], expected_type);
            assert_eq!(body["error"]["code"], expected_code);
        }
    }
}
