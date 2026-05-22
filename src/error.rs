use axum::{
    Json,
    response::{IntoResponse, Response},
};
use http::StatusCode;
use serde_json::json;
use thiserror::Error;

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
    #[error("streaming error: {0}")]
    Stream(String),
    #[error("internal error: {0}")]
    Internal(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
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
            Self::Stream(_) => StatusCode::BAD_GATEWAY,
            Self::Internal(_) | Self::Io(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    fn error_type(&self) -> &'static str {
        match self {
            Self::BadRequest(_) => "invalid_request_error",
            Self::Unauthorized => "authentication_error",
            Self::InvalidConfig(_) | Self::Internal(_) | Self::Io(_) => "server_error",
            Self::Upstream { .. } => "upstream_error",
            Self::Transport(_) | Self::Timeout(_) | Self::Stream(_) => "api_connection_error",
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
            Self::Stream(_) => "upstream_stream_error",
            Self::Internal(_) | Self::Io(_) => "internal_error",
        }
    }
}

impl IntoResponse for ProxyError {
    fn into_response(self) -> Response {
        let status = self.status_code();
        let message = self.to_string();
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
