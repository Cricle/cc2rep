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
    pub upstream_supports_reasoning_content: Option<bool>,
    #[serde(default)]
    pub upstream_supports_tool_choice_required: Option<bool>,
    #[serde(default)]
    pub upstream_supports_named_tool_choice: Option<bool>,
    #[serde(default = "default_response_ttl_seconds")]
    pub response_ttl_seconds: u64,
    #[serde(default)]
    pub drop_input_reasoning: bool,
    #[serde(default)]
    pub drop_tools: bool,
    #[serde(default)]
    pub upstream_body: HashMap<String, Value>,
    #[serde(default)]
    pub model_aliases: HashMap<String, String>,
    #[serde(default)]
    pub local_tools: HashMap<String, LocalToolSettings>,
    #[serde(default = "default_max_auto_tool_rounds")]
    pub max_auto_tool_rounds: u32,
    #[serde(default = "default_upstream_max_retries")]
    pub upstream_max_retries: u32,
    #[serde(default = "default_upstream_retry_base_delay_ms")]
    pub upstream_retry_base_delay_ms: u64,
    #[serde(default = "default_upstream_reasoning_effort_field")]
    pub upstream_reasoning_effort_field: String,
    #[serde(default)]
    pub web_search_url: Option<String>,
    #[serde(default = "default_web_search_max_results")]
    pub web_search_max_results: usize,
    #[serde(default)]
    pub file_search_paths: Vec<String>,
    #[serde(default = "default_file_search_max_results")]
    pub file_search_max_results: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LocalToolSettings {
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: HashMap<String, String>,
    #[serde(default)]
    pub workdir: Option<String>,
    #[serde(default = "default_tool_timeout_seconds")]
    pub timeout_seconds: f64,
    #[serde(default = "default_true")]
    pub stdin_json: bool,
    #[serde(default = "default_true")]
    pub output_json: bool,
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

fn default_tool_timeout_seconds() -> f64 {
    30.0
}

fn default_max_auto_tool_rounds() -> u32 {
    8
}

fn default_upstream_max_retries() -> u32 {
    3
}

fn default_upstream_retry_base_delay_ms() -> u64 {
    1000
}

fn default_response_ttl_seconds() -> u64 {
    3600
}

fn default_upstream_reasoning_effort_field() -> String {
    "reasoning_effort".to_owned()
}

fn default_web_search_max_results() -> usize {
    5
}

fn default_file_search_max_results() -> usize {
    5
}

fn default_true() -> bool {
    true
}

impl Settings {
    pub fn load(path: &Path) -> Result<Self, ProxyError> {
        let raw = fs::read_to_string(path).map_err(|err| {
            ProxyError::invalid_config(format!("failed to read config file: {err}"))
        })?;
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
        if self.max_auto_tool_rounds == 0 {
            return Err(ProxyError::invalid_config(
                "max_auto_tool_rounds must be positive",
            ));
        }
        for (name, tool) in &self.local_tools {
            if name.trim().is_empty() {
                return Err(ProxyError::invalid_config(
                    "local_tools cannot contain an empty tool name",
                ));
            }
            if tool.command.trim().is_empty() {
                return Err(ProxyError::invalid_config(format!(
                    "local tool `{name}` command cannot be empty"
                )));
            }
            if tool.timeout_seconds <= 0.0 {
                return Err(ProxyError::invalid_config(format!(
                    "local tool `{name}` timeout_seconds must be positive"
                )));
            }
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

#[cfg(test)]
mod tests {
    use std::fs;

    use serde_json::json;
    use tempfile::tempdir;

    use super::*;

    fn valid_settings() -> Settings {
        Settings {
            proxy_host: "127.0.0.1".to_owned(),
            proxy_port: 8080,
            proxy_api_key: "proxy-key".to_owned(),
            upstream_base_url: "https://api.example.com/".to_owned(),
            upstream_chat_path: "chat/completions".to_owned(),
            upstream_model: "upstream-model".to_owned(),
            upstream_api_key: "upstream-key".to_owned(),
            upstream_headers: [("x-test".to_owned(), "1".to_owned())]
                .into_iter()
                .collect(),
            upstream_api_key_header_name: "Authorization".to_owned(),
            upstream_api_key_prefix: "Bearer ".to_owned(),
            request_timeout_seconds: 30.0,
            strict_protocol: false,
            upstream_supports_image_input: false,
            upstream_supports_reasoning_content: None,
            upstream_supports_tool_choice_required: None,
            upstream_supports_named_tool_choice: None,
            response_ttl_seconds: 3600,
            drop_input_reasoning: false,
            drop_tools: false,
            upstream_body: [("seed".to_owned(), json!(1))].into_iter().collect(),
            model_aliases: [("client-model".to_owned(), "aliased-model".to_owned())]
                .into_iter()
                .collect(),
            local_tools: HashMap::new(),
            max_auto_tool_rounds: 8,
            upstream_max_retries: 3,
            upstream_retry_base_delay_ms: 1000,
            upstream_reasoning_effort_field: "reasoning_effort".to_owned(),
            web_search_url: None,
            web_search_max_results: 5,
            file_search_paths: Vec::new(),
            file_search_max_results: 5,
        }
    }

    #[test]
    fn load_reads_and_validates_json() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("config.json");
        fs::write(
            &path,
            json!({
                "proxy_host": "127.0.0.1",
                "proxy_port": 8080,
                "proxy_api_key": "proxy-key",
                "upstream_base_url": "https://api.example.com",
                "upstream_model": "upstream-model",
                "upstream_api_key": "upstream-key"
            })
            .to_string(),
        )
        .expect("write");

        let settings = Settings::load(&path).expect("load");
        assert_eq!(settings.upstream_chat_path, "/v1/chat/completions");
        assert_eq!(settings.upstream_api_key_header_name, "Authorization");
        assert_eq!(settings.upstream_api_key_prefix, "Bearer ");
        assert_eq!(settings.request_timeout_seconds, 120.0);
    }

    #[test]
    fn load_reports_missing_file_and_invalid_json() {
        let missing = Settings::load(Path::new("/definitely/missing.json")).unwrap_err();
        assert!(matches!(missing, ProxyError::InvalidConfig(_)));

        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("broken.json");
        fs::write(&path, "{").expect("write");
        let invalid = Settings::load(&path).unwrap_err();
        assert!(matches!(invalid, ProxyError::InvalidConfig(_)));
    }

    #[test]
    fn validate_rejects_bad_fields() {
        let mut settings = valid_settings();
        settings.proxy_host.clear();
        assert!(matches!(
            settings.validate(),
            Err(ProxyError::InvalidConfig(_))
        ));

        let mut settings = valid_settings();
        settings.proxy_api_key.clear();
        assert!(matches!(
            settings.validate(),
            Err(ProxyError::InvalidConfig(_))
        ));

        let mut settings = valid_settings();
        settings.upstream_base_url.clear();
        assert!(matches!(
            settings.validate(),
            Err(ProxyError::InvalidConfig(_))
        ));

        let mut settings = valid_settings();
        settings.upstream_model.clear();
        assert!(matches!(
            settings.validate(),
            Err(ProxyError::InvalidConfig(_))
        ));

        let mut settings = valid_settings();
        settings.upstream_api_key.clear();
        assert!(matches!(
            settings.validate(),
            Err(ProxyError::InvalidConfig(_))
        ));

        let mut settings = valid_settings();
        settings.request_timeout_seconds = 0.0;
        assert!(matches!(
            settings.validate(),
            Err(ProxyError::InvalidConfig(_))
        ));

        let mut settings = valid_settings();
        settings.max_auto_tool_rounds = 0;
        assert!(matches!(
            settings.validate(),
            Err(ProxyError::InvalidConfig(_))
        ));

        let mut settings = valid_settings();
        settings.local_tools.insert(
            "lookup".to_owned(),
            LocalToolSettings {
                command: String::new(),
                args: vec![],
                env: HashMap::new(),
                workdir: None,
                timeout_seconds: 10.0,
                stdin_json: true,
                output_json: true,
            },
        );
        assert!(matches!(
            settings.validate(),
            Err(ProxyError::InvalidConfig(_))
        ));
    }

    #[test]
    fn socket_addr_and_url_and_model_alias_work() {
        let settings = valid_settings();
        assert_eq!(settings.socket_addr().expect("socket").port(), 8080);
        assert_eq!(
            settings.upstream_url(),
            "https://api.example.com/chat/completions"
        );
        assert_eq!(settings.mapped_model(Some("client-model")), "aliased-model");
        assert_eq!(settings.mapped_model(Some("other-model")), "upstream-model");
        assert_eq!(settings.mapped_model(None), "upstream-model");
    }

    #[test]
    fn socket_addr_rejects_invalid_host() {
        let mut settings = valid_settings();
        settings.proxy_host = "[]".to_owned();
        assert!(matches!(
            settings.socket_addr(),
            Err(ProxyError::InvalidConfig(_))
        ));
    }
}
