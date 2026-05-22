use std::{collections::HashMap, process::Stdio, sync::Arc};

use serde_json::{Value, json};
use tokio::{
    io::AsyncWriteExt,
    process::Command,
    time::{Duration, timeout},
};
use tracing::{info, warn};

use crate::{
    config::{LocalToolSettings, Settings},
    error::ProxyError,
    translate::ToolCall,
};

#[derive(Clone)]
pub struct ToolExecutor {
    settings: Arc<Settings>,
}

#[derive(Debug, Clone)]
pub struct ToolOutput {
    pub call_id: String,
    pub output: String,
}

impl ToolExecutor {
    pub fn new(settings: Arc<Settings>) -> Self {
        Self { settings }
    }

    pub fn has_local_tools(&self) -> bool {
        !self.settings.local_tools.is_empty()
    }

    pub fn supports_tool(&self, name: &str) -> bool {
        self.settings.local_tools.contains_key(name)
    }

    pub async fn execute_calls(
        &self,
        tool_calls: &[ToolCall],
    ) -> Result<Option<Vec<ToolOutput>>, ProxyError> {
        if tool_calls.is_empty() {
            return Ok(None);
        }

        let mut outputs = Vec::new();
        for tool_call in tool_calls {
            let Some(tool) = self.settings.local_tools.get(&tool_call.name) else {
                return Ok(None);
            };
            outputs.push(self.execute_call(tool_call, tool).await?);
        }
        Ok(Some(outputs))
    }

    async fn execute_call(
        &self,
        tool_call: &ToolCall,
        tool: &LocalToolSettings,
    ) -> Result<ToolOutput, ProxyError> {
        let arguments = parse_arguments(&tool_call.arguments)?;
        let mut command = Command::new(&tool.command);
        command.args(&tool.args);
        if let Some(workdir) = &tool.workdir {
            command.current_dir(workdir);
        }
        for (key, value) in &tool.env {
            command.env(key, value);
        }

        let stdin_payload =
            build_stdin_payload(tool, &tool_call.name, &tool_call.call_id, &arguments);
        if tool.stdin_json {
            command.stdin(Stdio::piped());
        }
        command.stdout(Stdio::piped());
        command.stderr(Stdio::piped());

        info!(tool = %tool_call.name, call_id = %tool_call.call_id, command = %tool.command, "executing local tool");
        let mut child = command.spawn().map_err(|err| {
            ProxyError::Internal(format!(
                "failed to spawn local tool `{}`: {err}",
                tool_call.name
            ))
        })?;

        if let Some(payload) = stdin_payload {
            let Some(mut stdin) = child.stdin.take() else {
                return Err(ProxyError::Internal(format!(
                    "local tool `{}` stdin was not available",
                    tool_call.name
                )));
            };
            stdin.write_all(payload.as_bytes()).await.map_err(|err| {
                ProxyError::Internal(format!(
                    "failed to write stdin for local tool `{}`: {err}",
                    tool_call.name
                ))
            })?;
            stdin.flush().await.map_err(|err| {
                ProxyError::Internal(format!(
                    "failed to flush stdin for local tool `{}`: {err}",
                    tool_call.name
                ))
            })?;
            drop(stdin);
        }

        let output = timeout(
            Duration::from_secs_f64(tool.timeout_seconds),
            child.wait_with_output(),
        )
        .await
        .map_err(|_| {
            warn!(tool = %tool_call.name, call_id = %tool_call.call_id, "local tool timed out");
            ProxyError::bad_request(format!("local tool `{}` timed out", tool_call.name))
        })?
        .map_err(|err| {
            ProxyError::Internal(format!(
                "failed to wait for local tool `{}`: {err}",
                tool_call.name
            ))
        })?;

        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
        if !output.status.success() {
            let message = if stderr.is_empty() {
                format!(
                    "local tool `{}` exited with status {}",
                    tool_call.name, output.status
                )
            } else {
                format!("local tool `{}` failed: {}", tool_call.name, stderr)
            };
            return Err(ProxyError::bad_request(message));
        }

        let output = if tool.output_json {
            normalize_tool_json_output(&stdout)?
        } else if stdout.is_empty() {
            json!({
                "ok": true,
                "tool": tool_call.name,
                "call_id": tool_call.call_id,
                "arguments": arguments,
            })
            .to_string()
        } else {
            stdout
        };

        Ok(ToolOutput {
            call_id: tool_call.call_id.clone(),
            output,
        })
    }
}

fn parse_arguments(raw: &str) -> Result<Value, ProxyError> {
    if raw.trim().is_empty() {
        return Ok(json!({}));
    }
    serde_json::from_str(raw)
        .map_err(|err| ProxyError::bad_request(format!("tool arguments must be valid JSON: {err}")))
}

fn build_stdin_payload(
    tool: &LocalToolSettings,
    name: &str,
    call_id: &str,
    arguments: &Value,
) -> Option<String> {
    if !tool.stdin_json {
        return None;
    }
    Some(
        json!({
            "name": name,
            "call_id": call_id,
            "arguments": arguments,
        })
        .to_string(),
    )
}

fn normalize_tool_json_output(stdout: &str) -> Result<String, ProxyError> {
    if stdout.trim().is_empty() {
        return Ok(json!({}).to_string());
    }
    let parsed: Value = serde_json::from_str(stdout).map_err(|err| {
        ProxyError::bad_request(format!("local tool output must be valid JSON: {err}"))
    })?;
    serde_json::to_string(&parsed).map_err(|err| {
        ProxyError::Internal(format!("failed to serialize local tool output: {err}"))
    })
}

pub fn append_tool_outputs(
    messages: &mut Vec<Value>,
    tool_calls: &[ToolCall],
    outputs: &[ToolOutput],
) {
    messages.push(json!({
        "role": "assistant",
        "content": null,
        "tool_calls": tool_calls.iter().map(|tool_call| {
            json!({
                "id": tool_call.call_id,
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": tool_call.arguments,
                }
            })
        }).collect::<Vec<_>>(),
    }));

    let output_by_call_id: HashMap<&str, &str> = outputs
        .iter()
        .map(|output| (output.call_id.as_str(), output.output.as_str()))
        .collect();
    for tool_call in tool_calls {
        let output = output_by_call_id
            .get(tool_call.call_id.as_str())
            .copied()
            .unwrap_or("{}");
        messages.push(json!({
            "role": "tool",
            "tool_call_id": tool_call.call_id,
            "content": output,
        }));
    }
}

pub fn output_items_from_tool_outputs(outputs: &[ToolOutput]) -> Vec<Value> {
    outputs
        .iter()
        .map(|output| {
            json!({
                "type": "function_call_output",
                "call_id": output.call_id,
                "output": output.output,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn parse_arguments_and_json_output_are_normalized() {
        assert_eq!(parse_arguments("").expect("empty"), json!({}));
        assert_eq!(parse_arguments("{\"x\":1}").expect("json"), json!({"x":1}));
        assert!(parse_arguments("{").is_err());

        assert_eq!(normalize_tool_json_output("").expect("empty"), "{}");
        assert_eq!(
            normalize_tool_json_output("{\"x\":1}").expect("json"),
            "{\"x\":1}"
        );
        assert!(normalize_tool_json_output("{").is_err());
    }

    #[test]
    fn append_tool_outputs_builds_assistant_and_tool_messages() {
        let mut messages = vec![];
        let tool_calls = vec![ToolCall {
            call_id: "call_1".to_owned(),
            name: "lookup".to_owned(),
            arguments: "{\"q\":\"v\"}".to_owned(),
        }];
        let outputs = vec![ToolOutput {
            call_id: "call_1".to_owned(),
            output: "{\"ok\":true}".to_owned(),
        }];
        append_tool_outputs(&mut messages, &tool_calls, &outputs);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "assistant");
        assert_eq!(messages[1]["role"], "tool");
        assert_eq!(messages[1]["content"], "{\"ok\":true}");
        assert_eq!(
            output_items_from_tool_outputs(&outputs)[0]["type"],
            "function_call_output"
        );
    }
}
