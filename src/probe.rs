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

    info!(
        named_tool_choice = caps.supports_named_tool_choice,
        tool_choice_required = caps.supports_tool_choice_required,
        reasoning_content = caps.supports_reasoning_content,
        image_input = caps.supports_image_input,
        "probed upstream capabilities"
    );

    caps
}

fn check_reasoning_support(body: &str, caps: &mut Capabilities) {
    if let Ok(value) = serde_json::from_str::<Value>(body) {
        if let Some(choices) = value.get("choices").and_then(Value::as_array) {
            if let Some(message) = choices
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
    }
}
