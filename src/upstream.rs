use std::{sync::Arc, time::Duration};

use reqwest::{
    Client, Response,
    header::{HeaderMap, HeaderName, HeaderValue},
};
use serde_json::Value;

use crate::{config::Settings, error::ProxyError};

#[derive(Clone)]
pub struct UpstreamClient {
    settings: Arc<Settings>,
    http: Client,
}

impl UpstreamClient {
    pub fn new(settings: Arc<Settings>) -> Result<Self, ProxyError> {
        let http = Client::builder()
            .timeout(Duration::from_secs_f64(settings.request_timeout_seconds))
            .build()
            .map_err(|err| {
                ProxyError::invalid_config(format!("failed to build HTTP client: {err}"))
            })?;
        Ok(Self { settings, http })
    }

    pub async fn chat_json(&self, payload: &Value) -> Result<Value, ProxyError> {
        let response = self.send(payload).await?;
        response
            .json::<Value>()
            .await
            .map_err(|err| ProxyError::Transport(err.to_string()))
    }

    pub async fn chat_stream(&self, payload: &Value) -> Result<Response, ProxyError> {
        self.send(payload).await
    }

    async fn send(&self, payload: &Value) -> Result<Response, ProxyError> {
        let request = self
            .http
            .post(self.settings.upstream_url())
            .headers(self.build_headers()?)
            .json(payload);

        let response = request.send().await.map_err(map_reqwest_error)?;
        let status = response.status();
        if !status.is_success() {
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "failed to read upstream error body".to_owned());
            return Err(ProxyError::from_upstream_body(status, body));
        }
        Ok(response)
    }

    fn build_headers(&self) -> Result<HeaderMap, ProxyError> {
        let mut headers = HeaderMap::new();
        let auth_name: HeaderName =
            self.settings
                .upstream_api_key_header_name
                .parse()
                .map_err(|err| {
                    ProxyError::invalid_config(format!(
                        "invalid upstream_api_key_header_name: {err}"
                    ))
                })?;
        let auth_value = format!(
            "{}{}",
            self.settings.upstream_api_key_prefix, self.settings.upstream_api_key
        );
        headers.insert(
            auth_name,
            HeaderValue::from_str(&auth_value).map_err(|err| {
                ProxyError::invalid_config(format!("invalid upstream auth header value: {err}"))
            })?,
        );

        for (name, value) in &self.settings.upstream_headers {
            let header_name: HeaderName = name.parse().map_err(|err| {
                ProxyError::invalid_config(format!("invalid upstream header `{name}`: {err}"))
            })?;
            let header_value = HeaderValue::from_str(value).map_err(|err| {
                ProxyError::invalid_config(format!(
                    "invalid upstream header value for `{name}`: {err}"
                ))
            })?;
            headers.insert(header_name, header_value);
        }

        Ok(headers)
    }
}

fn map_reqwest_error(error: reqwest::Error) -> ProxyError {
    if error.is_timeout() {
        ProxyError::Timeout(error.to_string())
    } else {
        ProxyError::Transport(error.to_string())
    }
}
