use std::time::Duration;

use reqwest::Client;
use serde_json::{Value, json};
use tracing::{info, warn};

use crate::config::Settings;

#[derive(Debug, Clone)]
pub struct Capabilities {
    pub supports_named_tool_choice: bool,
    pub supports_tool_choice_required: bool,
    pub supports_reasoning_content: bool,
    pub supports_image_input: bool,
    pub supports_reasoning_effort: bool,
}

/// Result of a probe operation, with human-readable labels.
#[derive(Debug, Clone)]
pub struct ProbeReport {
    pub named_tool_choice: bool,
    pub tool_choice_required: bool,
    pub reasoning_content: bool,
    pub reasoning_effort: bool,
    pub image_input: bool,
}

impl ProbeReport {
    pub fn from_caps(caps: &Capabilities, settings: &Settings) -> Self {
        Self {
            named_tool_choice: caps.supports_named_tool_choice,
            tool_choice_required: caps.supports_tool_choice_required,
            reasoning_content: caps.supports_reasoning_content,
            reasoning_effort: caps.supports_reasoning_effort,
            image_input: settings.upstream_supports_image_input,
        }
    }

    pub fn print(&self) {
        println!("Upstream capability probe results:");
        println!();
        println!("  {:<35} {}", "named tool_choice", fmt_bool(self.named_tool_choice));
        println!("  {:<35} {}", "tool_choice: required", fmt_bool(self.tool_choice_required));
        println!("  {:<35} {}", "reasoning_content", fmt_bool(self.reasoning_content));
        println!("  {:<35} {}", "reasoning_effort", fmt_bool(self.reasoning_effort));
        println!("  {:<35} {}", "image input", fmt_bool(self.image_input));
    }
}

fn fmt_bool(v: bool) -> &'static str {
    if v { "yes" } else { "no" }
}

/// Models discovered from the upstream /models endpoint.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub id: String,
}

/// Try to list models from the upstream provider.
/// Returns a list of model IDs, or empty if the endpoint is not available.
pub async fn list_models(settings: &Settings) -> Vec<ModelInfo> {
    let client = match Client::builder().timeout(Duration::from_secs(10)).build() {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    let mut headers = reqwest::header::HeaderMap::new();
    let auth_name: reqwest::header::HeaderName = match settings.upstream_api_key_header_name.parse()
    {
        Ok(n) => n,
        Err(_) => reqwest::header::AUTHORIZATION,
    };
    let auth_value = format!(
        "{}{}",
        settings.upstream_api_key_prefix, settings.upstream_api_key
    );
    if let Ok(val) = reqwest::header::HeaderValue::from_str(&auth_value) {
        headers.insert(auth_name, val);
    }
    for (name, value) in &settings.upstream_headers {
        if let (Ok(n), Ok(v)) = (
            name.parse::<reqwest::header::HeaderName>(),
            reqwest::header::HeaderValue::from_str(value),
        ) {
            headers.insert(n, v);
        }
    }

    // Try /models relative to the base URL (strip /v1 or /chat/completions suffix)
    let base = settings.upstream_base_url.trim_end_matches('/');
    let models_url = format!("{}/models", base);

    match client.get(&models_url).headers(headers).send().await {
        Ok(resp) if resp.status().is_success() => {
            let body = resp.text().await.unwrap_or_default();
            parse_model_list(&body)
        }
        Ok(resp) => {
            info!(status = %resp.status(), "/models endpoint not available");
            Vec::new()
        }
        Err(err) => {
            warn!("/models probe failed: {err}");
            Vec::new()
        }
    }
}

fn parse_model_list(body: &str) -> Vec<ModelInfo> {
    let Ok(value) = serde_json::from_str::<Value>(body) else {
        return Vec::new();
    };
    let Some(data) = value.get("data").and_then(Value::as_array) else {
        return Vec::new();
    };
    data.iter()
        .filter_map(|item| {
            let id = item.get("id").and_then(Value::as_str)?.to_owned();
            Some(ModelInfo { id })
        })
        .collect()
}

/// Suggest model aliases based on available models and the configured upstream model.
/// Returns a map of well-known model names → actual upstream model ID.
pub fn suggest_aliases(models: &[ModelInfo], upstream_model: &str) -> std::collections::HashMap<String, String> {
    use std::collections::HashMap;
    let mut aliases = HashMap::new();

    if models.is_empty() {
        return aliases;
    }

    // Find the best chat model and reasoning model
    let chat_model = find_chat_model(models, upstream_model);
    let reasoning_model = find_reasoning_model(models, upstream_model);

    // Common model names that clients might request
    let chat_names = ["gpt-5-codex", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"];
    let reasoning_names = ["o1", "o1-mini", "o1-preview", "o3", "o3-mini", "o4-mini", "deepseek-reasoner"];

    for name in chat_names {
        if name != chat_model {
            aliases.insert(name.to_owned(), chat_model.to_owned());
        }
    }
    for name in reasoning_names {
        if reasoning_model != chat_model {
            // Only add reasoning aliases if there's a distinct reasoning model
            if name != reasoning_model {
                aliases.insert(name.to_owned(), reasoning_model.to_owned());
            }
        } else if name != chat_model {
            aliases.insert(name.to_owned(), chat_model.to_owned());
        }
    }

    aliases
}

fn find_chat_model(models: &[ModelInfo], upstream_model: &str) -> String {
    // If the upstream_model is in the list, use it
    if models.iter().any(|m| m.id == upstream_model) {
        return upstream_model.to_owned();
    }
    // Otherwise pick the first non-reasoning model
    models
        .iter()
        .find(|m| !is_reasoning_model(&m.id))
        .or_else(|| models.first())
        .map(|m| m.id.clone())
        .unwrap_or_else(|| upstream_model.to_owned())
}

fn find_reasoning_model(models: &[ModelInfo], upstream_model: &str) -> String {
    // Look for a reasoning model
    if let Some(m) = models.iter().find(|m| is_reasoning_model(&m.id)) {
        return m.id.clone();
    }
    // Fall back to chat model
    find_chat_model(models, upstream_model)
}

fn is_reasoning_model(id: &str) -> bool {
    let lower = id.to_lowercase();
    lower.contains("reasoner")
        || lower.contains("r1")
        || lower.contains("thinking")
        || lower.contains("-o1")
        || lower.contains("-o3")
}

/// Probe upstream and return a report. Always probes (no skip logic).
pub async fn probe_report(settings: &Settings) -> (Capabilities, ProbeReport) {
    // Build capabilities without using config overrides — force full probe
    let mut probe_settings = settings.clone();
    probe_settings.upstream_supports_named_tool_choice = None;
    probe_settings.upstream_supports_tool_choice_required = None;
    probe_settings.upstream_supports_reasoning_content = None;
    let caps = probe_upstream(&probe_settings).await;
    let report = ProbeReport::from_caps(&caps, settings);
    (caps, report)
}

pub async fn probe_upstream(settings: &Settings) -> Capabilities {
    let mut caps = Capabilities {
        supports_named_tool_choice: settings
            .upstream_supports_named_tool_choice
            .unwrap_or(false),
        supports_tool_choice_required: settings
            .upstream_supports_tool_choice_required
            .unwrap_or(false),
        supports_reasoning_content: settings
            .upstream_supports_reasoning_content
            .unwrap_or(false),
        supports_image_input: settings.upstream_supports_image_input,
        supports_reasoning_effort: false,
    };

    // If all auto-detectable fields are already set, skip probing
    if settings.upstream_supports_named_tool_choice.is_some()
        && settings.upstream_supports_tool_choice_required.is_some()
        && settings.upstream_supports_reasoning_content.is_some()
    {
        info!("all capabilities explicitly configured, skipping probe");
        return caps;
    }

    let url = settings.upstream_url();
    let client = match Client::builder().timeout(Duration::from_secs(10)).build() {
        Ok(c) => c,
        Err(err) => {
            warn!("failed to build probe HTTP client: {err}");
            return caps;
        }
    };

    let mut headers = reqwest::header::HeaderMap::new();
    let auth_name: reqwest::header::HeaderName = match settings.upstream_api_key_header_name.parse()
    {
        Ok(n) => n,
        Err(_) => reqwest::header::AUTHORIZATION,
    };
    let auth_value = format!(
        "{}{}",
        settings.upstream_api_key_prefix, settings.upstream_api_key
    );
    if let Ok(val) = reqwest::header::HeaderValue::from_str(&auth_value) {
        headers.insert(auth_name, val);
    }
    for (name, value) in &settings.upstream_headers {
        if let (Ok(n), Ok(v)) = (
            name.parse::<reqwest::header::HeaderName>(),
            reqwest::header::HeaderValue::from_str(value),
        ) {
            headers.insert(n, v);
        }
    }

    // Run tool_choice and reasoning probes in parallel
    let need_tool_probe = settings.upstream_supports_named_tool_choice.is_none()
        || settings.upstream_supports_tool_choice_required.is_none();
    let need_reasoning_probe = settings.upstream_supports_reasoning_content.is_none();

    if need_tool_probe || need_reasoning_probe {
        let tool_probe = async {
            if !need_tool_probe {
                return None;
            }
            let probe_body = json!({
                "model": settings.upstream_model,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "_probe",
                        "parameters": {"type": "object", "properties": {}}
                    }
                }],
                "tool_choice": {"type": "function", "function": {"name": "_probe"}},
                "stream": false
            });
            Some(
                client
                    .post(&url)
                    .headers(headers.clone())
                    .json(&probe_body)
                    .send()
                    .await,
            )
        };

        let reasoning_probe = async {
            if !need_reasoning_probe {
                return None;
            }
            let simple_body = json!({
                "model": settings.upstream_model,
                "messages": [{"role": "user", "content": "1+1?"}],
                "max_tokens": 16,
                "stream": false
            });
            Some(
                client
                    .post(&url)
                    .headers(headers.clone())
                    .json(&simple_body)
                    .send()
                    .await,
            )
        };

        let (tool_result, reasoning_result) = tokio::join!(tool_probe, reasoning_probe);

        // Process tool_choice probe result
        if let Some(Ok(resp)) = tool_result {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            if status.is_success() {
                info!("upstream supports named tool_choice");
                caps.supports_named_tool_choice = true;
                caps.supports_tool_choice_required = true;
                check_reasoning_support(&body, &mut caps);
            } else {
                let lower = body.to_lowercase();
                if lower.contains("tool_choice") || lower.contains("tool choice") {
                    warn!("upstream does not support named tool_choice: {body}");
                    caps.supports_named_tool_choice = false;

                    // Try tool_choice: "required"
                    if settings.upstream_supports_tool_choice_required.is_none() {
                        let required_body = json!({
                            "model": settings.upstream_model,
                            "messages": [{"role": "user", "content": "hi"}],
                            "max_tokens": 1,
                            "tool_choice": "required",
                            "stream": false
                        });
                        match client
                            .post(&url)
                            .headers(headers.clone())
                            .json(&required_body)
                            .send()
                            .await
                        {
                            Ok(required_resp) => {
                                if required_resp.status().is_success() {
                                    info!("upstream supports tool_choice: required");
                                    caps.supports_tool_choice_required = true;
                                } else {
                                    info!("upstream does not support tool_choice: required");
                                    caps.supports_tool_choice_required = false;
                                }
                            }
                            Err(err) => {
                                warn!("tool_choice required probe failed: {err}");
                                caps.supports_tool_choice_required = false;
                            }
                        }
                    }
                } else {
                    warn!("probe returned {status}, assuming basic tool_choice support");
                    caps.supports_named_tool_choice = true;
                    caps.supports_tool_choice_required = true;
                    check_reasoning_support(&body, &mut caps);
                }
            }
        } else if let Some(Err(err)) = tool_result {
            warn!("tool probe request failed: {err}");
        }

        // Process reasoning probe result
        if !caps.supports_reasoning_content {
            if let Some(Ok(resp)) = reasoning_result {
                if resp.status().is_success() {
                    let body = resp.text().await.unwrap_or_default();
                    check_reasoning_support(&body, &mut caps);
                }
            } else if let Some(Err(err)) = reasoning_result {
                warn!("reasoning probe failed: {err}");
            }
        }
    }

    // Probe reasoning_effort support
    {
        let effort_body = json!({
            "model": settings.upstream_model,
            "messages": [{"role": "user", "content": "1+1?"}],
            "max_tokens": 16,
            "reasoning_effort": "low",
            "stream": false
        });
        match client.post(&url).headers(headers.clone()).json(&effort_body).send().await {
            Ok(resp) if resp.status().is_success() => {
                info!("upstream supports reasoning_effort");
                caps.supports_reasoning_effort = true;
            }
            Ok(resp) => {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                info!(status = %status, "upstream does not support reasoning_effort: {body}");
            }
            Err(err) => {
                warn!("reasoning_effort probe failed: {err}");
            }
        }
    }

    info!(
        named_tool_choice = caps.supports_named_tool_choice,
        tool_choice_required = caps.supports_tool_choice_required,
        reasoning_content = caps.supports_reasoning_content,
        reasoning_effort = caps.supports_reasoning_effort,
        image_input = caps.supports_image_input,
        "probed upstream capabilities"
    );

    caps
}

fn check_reasoning_support(body: &str, caps: &mut Capabilities) {
    if let Ok(value) = serde_json::from_str::<Value>(body)
        && let Some(choices) = value.get("choices").and_then(Value::as_array)
        && let Some(message) = choices
            .first()
            .and_then(|c| c.get("message"))
            .and_then(Value::as_object)
    {
        for key in ["reasoning_content", "reasoning", "thinking"] {
            if let Some(val) = message.get(key) {
                let has_content = match val {
                    Value::String(s) => !s.is_empty(),
                    Value::Array(a) => !a.is_empty(),
                    Value::Null => false,
                    _ => true,
                };
                if has_content {
                    info!("upstream supports reasoning content (found `{key}`)");
                    caps.supports_reasoning_content = true;
                    return;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_model_list_extracts_ids() {
        let body = r#"{"data":[{"id":"deepseek-chat","object":"model"},{"id":"deepseek-reasoner","object":"model"}]}"#;
        let models = parse_model_list(body);
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "deepseek-chat");
        assert_eq!(models[1].id, "deepseek-reasoner");
    }

    #[test]
    fn parse_model_list_handles_empty_and_invalid() {
        assert!(parse_model_list("").is_empty());
        assert!(parse_model_list("{}").is_empty());
        assert!(parse_model_list(r#"{"data":[]}"#).is_empty());
    }

    #[test]
    fn suggest_aliases_maps_common_names() {
        let models = vec![
            ModelInfo { id: "deepseek-chat".to_owned() },
            ModelInfo { id: "deepseek-reasoner".to_owned() },
        ];
        let aliases = suggest_aliases(&models, "deepseek-chat");
        assert_eq!(aliases.get("gpt-5-codex").unwrap(), "deepseek-chat");
        assert_eq!(aliases.get("gpt-4o").unwrap(), "deepseek-chat");
        // Reasoning model should be mapped to deepseek-reasoner
        assert_eq!(aliases.get("o1").unwrap(), "deepseek-reasoner");
    }

    #[test]
    fn suggest_aliases_single_model() {
        let models = vec![
            ModelInfo { id: "mimo-v2.5-pro".to_owned() },
        ];
        let aliases = suggest_aliases(&models, "mimo-v2.5-pro");
        assert_eq!(aliases.get("gpt-5-codex").unwrap(), "mimo-v2.5-pro");
        // No distinct reasoning model, so o1 maps to the same
        assert_eq!(aliases.get("o1").unwrap(), "mimo-v2.5-pro");
    }

    #[test]
    fn suggest_aliases_empty_models() {
        let aliases = suggest_aliases(&[], "some-model");
        assert!(aliases.is_empty());
    }

    #[test]
    fn is_reasoning_model_detects_variants() {
        assert!(is_reasoning_model("deepseek-reasoner"));
        assert!(is_reasoning_model("deepseek-r1"));
        assert!(is_reasoning_model("qwen-thinking"));
        assert!(!is_reasoning_model("deepseek-chat"));
        assert!(!is_reasoning_model("gpt-4o"));
    }
}

