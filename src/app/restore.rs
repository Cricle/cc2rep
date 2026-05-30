use serde_json::{Value, json};

use crate::{
    error::ProxyError,
    translate::{AssistantTurn, RequestContext, ToolCall},
};
pub(crate) fn assistant_turn_from_output(output: &[Value]) -> Result<AssistantTurn, ProxyError> {
    let mut turn = AssistantTurn::default();
    for item in output {
        let item_type = item.get("type").and_then(Value::as_str).unwrap_or("");
        match item_type {
            "reasoning" => {
                if let Some(summary) = item
                    .get("summary")
                    .and_then(Value::as_array)
                    .and_then(|items| items.first())
                    .and_then(|item| item.get("text"))
                    .and_then(Value::as_str)
                {
                    turn.reasoning = summary.to_owned();
                }
            }
            "message" => {
                if let Some(content) = item.get("content") {
                    turn.text = crate::translate::extract_message_text(content)?;
                }
            }
            "function_call" => {
                let call_id = item.get("call_id").and_then(Value::as_str).ok_or_else(|| {
                    ProxyError::Internal("stored function_call missing call_id".to_owned())
                })?;
                let name = item.get("name").and_then(Value::as_str).ok_or_else(|| {
                    ProxyError::Internal("stored function_call missing name".to_owned())
                })?;
                let arguments = item
                    .get("arguments")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_owned();
                turn.tool_calls.push(ToolCall {
                    call_id: call_id.to_owned(),
                    name: name.to_owned(),
                    arguments,
                });
            }
            _ => {}
        }
    }
    Ok(turn)
}

pub(crate) fn context_from_response(response: &Value) -> Result<RequestContext, ProxyError> {
    let output = response.get("output").and_then(Value::as_array);
    Ok(RequestContext {
        response_id: response
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| ProxyError::Internal("stored response missing id".to_owned()))?
            .to_owned(),
        reasoning_id: output
            .and_then(|items| {
                items.iter().find_map(|item| {
                    if item.get("type").and_then(Value::as_str) == Some("reasoning") {
                        item.get("id").and_then(Value::as_str)
                    } else {
                        None
                    }
                })
            })
            .unwrap_or("rs_cancelled")
            .to_owned(),
        message_id: output
            .and_then(|items| {
                items.iter().find_map(|item| {
                    if item.get("type").and_then(Value::as_str) == Some("message") {
                        item.get("id").and_then(Value::as_str)
                    } else {
                        None
                    }
                })
            })
            .unwrap_or("msg_cancelled")
            .to_owned(),
        created_at: response
            .get("created_at")
            .and_then(Value::as_i64)
            .ok_or_else(|| ProxyError::Internal("stored response missing created_at".to_owned()))?,
        client_model: response
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned(),
        upstream_model: String::new(),
        stream: false,
        store: response
            .get("store")
            .and_then(Value::as_bool)
            .unwrap_or(true),
        parallel_tool_calls: response
            .get("parallel_tool_calls")
            .and_then(Value::as_bool)
            .unwrap_or(true),
        instructions: response
            .get("instructions")
            .and_then(Value::as_str)
            .map(str::to_owned),
        metadata: response
            .get("metadata")
            .cloned()
            .unwrap_or_else(|| json!({})),
        tool_choice: response
            .get("tool_choice")
            .cloned()
            .unwrap_or_else(|| json!("auto")),
        tools: response
            .get("tools")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default(),
        max_output_tokens: response.get("max_output_tokens").and_then(Value::as_u64),
        max_tool_calls: response
            .get("max_tool_calls")
            .and_then(Value::as_u64)
            .map(|v| v as u32),
        hosted_output_items: Vec::new(),
        previous_response_id: response
            .get("previous_response_id")
            .and_then(Value::as_str)
            .map(str::to_owned),
        reasoning_effort: None,
        reasoning_summary: None,
        truncation: response
            .get("truncation")
            .and_then(Value::as_str)
            .unwrap_or("auto")
            .to_owned(),
        include: Vec::new(),
        temperature: response.get("temperature").and_then(Value::as_f64),
        top_p: response.get("top_p").and_then(Value::as_f64),
        skip_reasoning_output: false,
    })
}

