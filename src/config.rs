use std::{
    collections::HashMap,
    fs,
    net::{SocketAddr, ToSocketAddrs},
    path::Path,
};

use serde::Deserialize;
use serde_json::Value;

use crate::error::ProxyError;

#[derive(Debug, Clone, Deserialize)]
pub struct Settings {
    pub proxy_host: String,
    pub proxy_port: u16,
    pub proxy_api_key: String,
    pub upstream_base_url: String,
    #[serde(default = "default_upstream_chat_path")]
    pub upstream_chat_path: String,
    pub upstream_model: String,
    pub upstream_api_key: String,
    #[serde(default)]
    pub upstream_headers: HashMap<String, String>,
    #[serde(default = "default_api_key_header_name")]
    pub upstream_api_key_header_name: String,
    #[serde(default = "default_api_key_prefix")]
    pub upstream_api_key_prefix: String,
    #[serde(default = "default_request_timeout_seconds")]
    pub request_timeout_seconds: f64,
    #[serde(default)]
    pub strict_protocol: bool,
    #[serde(default)]
    pub upstream_supports_image_input: bool,
    #[serde(default)]
    pub upstream_body: HashMap<String, Value>,
    #[serde(default)]
    pub model_aliases: HashMap<String, String>,
}

fn default_upstream_chat_path() -> String {
    "/v1/chat/completions".to_owned()
}

fn default_api_key_header_name() -> String {
    "Authorization".to_owned()
}

fn default_api_key_prefix() -> String {
    "Bearer ".to_owned()
}

fn default_request_timeout_seconds() -> f64 {
    120.0
}

impl Settings {
    pub fn load(path: &Path) -> Result<Self, ProxyError> {
        let raw = fs::read_to_string(path).map_err(ProxyError::Io)?;
        let settings: Self = serde_json::from_str(&raw)
            .map_err(|err| ProxyError::invalid_config(format!("invalid config JSON: {err}")))?;
        settings.validate()?;
        Ok(settings)
    }

    pub fn validate(&self) -> Result<(), ProxyError> {
        if self.proxy_host.trim().is_empty() {
            return Err(ProxyError::invalid_config("proxy_host cannot be empty"));
        }
        if self.proxy_api_key.trim().is_empty() {
            return Err(ProxyError::invalid_config("proxy_api_key cannot be empty"));
        }
        if self.upstream_base_url.trim().is_empty() {
            return Err(ProxyError::invalid_config(
                "upstream_base_url cannot be empty",
            ));
        }
        if self.upstream_model.trim().is_empty() {
            return Err(ProxyError::invalid_config("upstream_model cannot be empty"));
        }
        if self.upstream_api_key.trim().is_empty() {
            return Err(ProxyError::invalid_config(
                "upstream_api_key cannot be empty",
            ));
        }
        if self.request_timeout_seconds <= 0.0 {
            return Err(ProxyError::invalid_config(
                "request_timeout_seconds must be positive",
            ));
        }
        let _ = self.socket_addr()?;
        Ok(())
    }

    pub fn socket_addr(&self) -> Result<SocketAddr, ProxyError> {
        let addr = format!("{}:{}", self.proxy_host, self.proxy_port);
        addr.to_socket_addrs()
            .map_err(|err| ProxyError::invalid_config(format!("invalid listen address: {err}")))?
            .next()
            .ok_or_else(|| ProxyError::invalid_config("listen address resolved to no socket"))
    }

    pub fn upstream_url(&self) -> String {
        let base = self.upstream_base_url.trim_end_matches('/');
        let path = if self.upstream_chat_path.starts_with('/') {
            self.upstream_chat_path.clone()
        } else {
            format!("/{}", self.upstream_chat_path)
        };
        format!("{base}{path}")
    }

    pub fn mapped_model(&self, requested: Option<&str>) -> String {
        let requested = requested.unwrap_or(self.upstream_model.as_str());
        self.model_aliases
            .get(requested)
            .cloned()
            .unwrap_or_else(|| self.upstream_model.clone())
    }
}
