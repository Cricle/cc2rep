use std::{
    sync::OnceLock,
    time::{SystemTime, UNIX_EPOCH},
};

use serde_json::{Map, Value, json};
use uuid::Uuid;

use crate::{
    config::Settings,
    error::ProxyError,
    normalize::{build_messages, normalize_input_items, normalize_tool_choice, normalize_tools},
    probe::Capabilities,
    protocol::ProtocolReport,
};

// Re-export from response module for backward compatibility
pub use crate::response::{
    build_cancelled_response, build_content_part, build_failed_response, build_function_call_item,
    build_in_progress_response, build_message_item, build_reasoning_item, build_response,
};

#[derive(Debug, Clone)]
pub struct RequestContext {
    pub response_id: String,
    pub reasoning_id: String,
    pub message_id: String,
    pub created_at: i64,
    pub client_model: String,
    pub upstream_model: String,
    pub stream: bool,
    pub store: bool,
    pub parallel_tool_calls: bool,
    pub instructions: Option<String>,
    pub metadata: Value,
    pub tool_choice: Value,
    pub tools: Vec<Value>,
    pub max_output_tokens: Option<u64>,
    pub max_tool_calls: Option<u32>,
    pub hosted_output_items: Vec<Value>,
}

#[derive(Debug, Clone)]
pub struct ToolCall {
    pub call_id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Default)]
pub struct AssistantTurn {
    pub text: String,
    pub reasoning: String,
    pub tool_calls: Vec<ToolCall>,
}

#[derive(Debug, Clone)]
pub struct TranslatedRequest {
    pub upstream_payload: Value,
    pub context: RequestContext,
    pub input_items: Vec<Value>,
    pub request_messages: Vec<Value>,
}

pub fn translate_request(
    payload: &Map<String, Value>,
    settings: &Settings,
    report: &ProtocolReport,
    previous_messages: &[Value],
    capabilities: &Capabilities,
) -> Result<TranslatedRequest, ProxyError> {
    validate_text_format(payload)?;

    let client_model =
        optional_string(payload, "model").unwrap_or_else(|| settings.upstream_model.clone());
    let upstream_model = settings.mapped_model(Some(&client_model));
    let instructions = optional_string(payload, "instructions");
    let stream = optional_bool(payload, "stream")?.unwrap_or(false);
    let store = optional_bool(payload, "store")?.unwrap_or(true);
    let parallel_tool_calls = optional_bool(payload, "parallel_tool_calls")?.unwrap_or(true);
    let metadata = build_metadata(payload.get("metadata"), report)?;
    let max_output_tokens = optional_u64(payload, "max_output_tokens")?
        .and_then(|v| if v == 0 { None } else { Some(v) });
    let max_tool_calls = optional_u64(payload, "max_tool_calls")?.map(|v| v as u32);
    let input_items = normalize_input_items(payload.get("input"), settings, capabilities)?;
    let tools = if settings.drop_tools {
        Vec::new()
    } else {
        normalize_tools(payload.get("tools"))?
    };
    let tool_choice = if settings.drop_tools {
        Value::String("none".to_owned())
    } else {
        payload
            .get("tool_choice")
            .cloned()
            .unwrap_or_else(|| Value::String("auto".to_owned()))
    };

    let messages = build_messages(
        previous_messages,
        &input_items,
        instructions.as_deref(),
        settings,
        capabilities,
    )?;
    let mut upstream: Map<String, Value> = settings.upstream_body.clone().into_iter().collect();
    upstream.insert("model".to_owned(), Value::String(upstream_model.clone()));
    upstream.insert("stream".to_owned(), Value::Bool(stream));

    for field in [
        "temperature",
        "top_p",
        "presence_penalty",
        "frequency_penalty",
        "stop",
        "user",
        "response_format",
    ] {
        if let Some(value) = payload.get(field) {
            upstream.insert(field.to_owned(), value.clone());
        }
    }
    if let Some(max_tokens) = max_output_tokens {
        upstream.insert("max_tokens".to_owned(), json!(max_tokens));
    }
    if let Some(top_logprobs) = optional_u64(payload, "top_logprobs")? {
        upstream.insert("logprobs".to_owned(), Value::Bool(true));
        upstream.insert("top_logprobs".to_owned(), json!(top_logprobs));
    }
    if !tools.is_empty() {
        upstream.insert("tools".to_owned(), Value::Array(tools.clone()));
        upstream.insert(
            "tool_choice".to_owned(),
            normalize_tool_choice(payload.get("tool_choice"), settings, capabilities)?,
        );
    }

    let response_id = format!("resp_{}", Uuid::new_v4().simple());
    let context = RequestContext {
        reasoning_id: format!("rs_{}", Uuid::new_v4().simple()),
        message_id: format!("msg_{}", Uuid::new_v4().simple()),
        created_at: unix_timestamp(),
        client_model,
        upstream_model,
        stream,
        store,
        parallel_tool_calls,
        instructions,
        metadata,
        tool_choice,
        tools,
        max_output_tokens,
        max_tool_calls,
        response_id,
        hosted_output_items: Vec::new(),
    };

    upstream.insert("messages".to_owned(), Value::Array(messages));

    let request_messages = upstream
        .get("messages")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();

    Ok(TranslatedRequest {
        upstream_payload: Value::Object(upstream),
        context,
        input_items,
        request_messages,
    })
}

pub fn usage_from_upstream(usage: Option<&Value>) -> Value {
    let Some(usage) = usage else {
        return empty_usage();
    };
    let input_tokens = usage
        .get("prompt_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let output_tokens = usage
        .get("completion_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let total_tokens = usage
        .get("total_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(input_tokens + output_tokens);
    let cached_tokens = usage
        .get("prompt_tokens_details")
        .and_then(|details| details.get("cached_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let reasoning_tokens = usage
        .get("completion_tokens_details")
        .and_then(|details| details.get("reasoning_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);

    json!({
        "input_tokens": input_tokens,
        "input_tokens_details": {
            "cached_tokens": cached_tokens,
        },
        "output_tokens": output_tokens,
        "output_tokens_details": {
            "reasoning_tokens": reasoning_tokens,
        },
        "total_tokens": total_tokens,
    })
}

/// Accumulate usage from an additional round into an existing total.
pub fn merge_usage(total: &mut Value, round: &Value) {
    let input = round
        .get("input_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let output = round
        .get("output_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let cached = round
        .get("input_tokens_details")
        .and_then(|d| d.get("cached_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let reasoning = round
        .get("output_tokens_details")
        .and_then(|d| d.get("reasoning_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);

    if let Some(t) = total.get_mut("input_tokens") {
        *t = json!(t.as_u64().unwrap_or(0) + input);
    }
    if let Some(t) = total.get_mut("output_tokens") {
        *t = json!(t.as_u64().unwrap_or(0) + output);
    }
    if let Some(t) = total.get_mut("total_tokens") {
        *t = json!(t.as_u64().unwrap_or(0) + input + output);
    }
    if let Some(details) = total.get_mut("input_tokens_details")
        && let Some(t) = details.get_mut("cached_tokens")
    {
        *t = json!(t.as_u64().unwrap_or(0) + cached);
    }
    if let Some(details) = total.get_mut("output_tokens_details")
        && let Some(t) = details.get_mut("reasoning_tokens")
    {
        *t = json!(t.as_u64().unwrap_or(0) + reasoning);
    }
}

pub fn parse_assistant_turn_from_response(upstream: &Value) -> Result<AssistantTurn, ProxyError> {
    let choice = upstream
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .ok_or_else(|| ProxyError::bad_request("upstream response is missing choices[0]"))?;
    let message = choice
        .get("message")
        .and_then(Value::as_object)
        .ok_or_else(|| ProxyError::bad_request("upstream response is missing message"))?;
    let text = match message.get("content") {
        Some(value) => extract_message_text(value)?,
        None => String::new(),
    };
    let reasoning = extract_reasoning_content(message);
    let tool_calls = parse_tool_calls(message.get("tool_calls"))?;
    Ok(AssistantTurn {
        text,
        reasoning,
        tool_calls,
    })
}

pub fn build_history_message(turn: &AssistantTurn) -> Option<Value> {
    if turn.text.is_empty() && turn.tool_calls.is_empty() {
        return None;
    }
    let mut message = json!({
        "role": "assistant",
        "content": if turn.text.is_empty() { Value::Null } else { Value::String(turn.text.clone()) },
    });
    if !turn.tool_calls.is_empty() {
        message["tool_calls"] = json!(
            turn.tool_calls
                .iter()
                .map(|tc| json!({
                    "id": tc.call_id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments,
                    }
                }))
                .collect::<Vec<_>>()
        );
    }
    if !turn.reasoning.is_empty() {
        message["reasoning_content"] = Value::String(turn.reasoning.clone());
    }
    Some(message)
}

pub fn extract_message_text(value: &Value) -> Result<String, ProxyError> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::Array(parts) => {
            let mut text = String::new();
            for part in parts {
                text.push_str(&extract_text_part(part)?);
            }
            Ok(text)
        }
        Value::Null => Ok(String::new()),
        _ => Err(ProxyError::bad_request(
            "message content must be a string or array",
        )),
    }
}

pub fn function_item_id(call_id: &str) -> String {
    format!("fc_{call_id}")
}

pub fn reasoning_output_offset(turn: &AssistantTurn) -> usize {
    if turn.reasoning.is_empty() { 0 } else { 1 }
}

pub fn message_output_offset(turn: &AssistantTurn) -> usize {
    reasoning_output_offset(turn)
        + if !turn.text.is_empty() || turn.tool_calls.is_empty() {
            1
        } else {
            0
        }
}

/// Total number of output items before the assistant turn (hosted + reasoning + message).
pub fn pre_turn_output_count(turn: &AssistantTurn, hosted_count: usize) -> usize {
    hosted_count + message_output_offset(turn)
}

fn build_metadata(source: Option<&Value>, report: &ProtocolReport) -> Result<Value, ProxyError> {
    let mut metadata = match source {
        Some(Value::Object(map)) => map.clone(),
        Some(_) => {
            return Err(ProxyError::bad_request(
                "metadata must be a JSON object when provided",
            ));
        }
        None => Map::new(),
    };

    if report.has_compatibility_notes() {
        metadata.insert("response_proxy".to_owned(), report.metadata_fragment());
    }

    Ok(Value::Object(metadata))
}

fn parse_tool_calls(value: Option<&Value>) -> Result<Vec<ToolCall>, ProxyError> {
    let Some(Value::Array(tool_calls)) = value else {
        return Ok(Vec::new());
    };
    let mut parsed = Vec::new();
    for tool_call in tool_calls {
        let map = tool_call
            .as_object()
            .ok_or_else(|| ProxyError::bad_request("upstream tool_calls must be objects"))?;
        let function = map
            .get("function")
            .and_then(Value::as_object)
            .ok_or_else(|| ProxyError::bad_request("upstream tool_calls need function"))?;
        let name = function
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| ProxyError::bad_request("upstream tool_calls need function.name"))?;
        let arguments = stringify_json_value(function.get("arguments").ok_or_else(|| {
            ProxyError::bad_request("upstream tool_calls need function.arguments")
        })?)?;
        let call_id = map
            .get("id")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .unwrap_or_else(|| format!("call_{}", Uuid::new_v4().simple()));
        parsed.push(ToolCall {
            call_id,
            name: name.to_owned(),
            arguments,
        });
    }
    Ok(parsed)
}

fn extract_text_part(part: &Value) -> Result<String, ProxyError> {
    if let Some(text) = part.as_str() {
        return Ok(text.to_owned());
    }
    let map = part
        .as_object()
        .ok_or_else(|| ProxyError::bad_request("content parts must be strings or objects"))?;
    let part_type = map.get("type").and_then(Value::as_str).unwrap_or("text");
    match part_type {
        "input_text" | "output_text" | "text" => map
            .get("text")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| ProxyError::bad_request("text part is missing text")),
        "input_image" | "image_url" => Ok(String::new()),
        other => Err(ProxyError::bad_request(format!(
            "unsupported content part type `{other}`"
        ))),
    }
}

fn extract_reasoning_content(message: &Map<String, Value>) -> String {
    extract_first_reasoning(message).unwrap_or_default()
}

/// Extract the first non-empty reasoning value from a map with standard keys.
/// Shared by stream delta extraction and message content extraction.
pub fn extract_first_reasoning(map: &Map<String, Value>) -> Option<String> {
    for key in [
        "reasoning_content",
        "reasoning",
        "thinking",
        "reasoning_summary",
    ] {
        if let Some(value) = map.get(key)
            && let Some(reasoning) = flatten_reasoning_value(value)
            && !reasoning.is_empty()
        {
            return Some(reasoning);
        }
    }
    None
}

pub fn flatten_reasoning_value(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => {
            if text.is_empty() {
                None
            } else {
                Some(text.clone())
            }
        }
        Value::Array(items) => {
            let mut joined = String::new();
            for item in items {
                if let Some(text) = flatten_reasoning_value(item) {
                    joined.push_str(&text);
                }
            }
            if joined.is_empty() {
                None
            } else {
                Some(joined)
            }
        }
        Value::Object(map) => {
            for key in ["text", "content", "summary", "reasoning_content"] {
                if let Some(value) = map.get(key)
                    && let Some(text) = flatten_reasoning_value(value)
                {
                    return Some(text);
                }
            }
            None
        }
        _ => None,
    }
}

pub(crate) fn stringify_json_value(value: &Value) -> Result<String, ProxyError> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::Null => Ok(String::new()),
        _ => serde_json::to_string(value)
            .map_err(|err| ProxyError::Internal(format!("failed to encode JSON value: {err}"))),
    }
}

fn validate_text_format(payload: &Map<String, Value>) -> Result<(), ProxyError> {
    let Some(text) = payload.get("text") else {
        return Ok(());
    };
    let Some(format_type) = text
        .get("format")
        .and_then(|format| format.get("type"))
        .and_then(Value::as_str)
    else {
        return Ok(());
    };
    if format_type == "text" {
        Ok(())
    } else {
        Err(ProxyError::bad_request(format!(
            "text.format.type `{format_type}` is not supported"
        )))
    }
}

fn optional_string(payload: &Map<String, Value>, key: &str) -> Option<String> {
    payload.get(key).and_then(Value::as_str).map(str::to_owned)
}

fn optional_bool(payload: &Map<String, Value>, key: &str) -> Result<Option<bool>, ProxyError> {
    match payload.get(key) {
        Some(Value::Bool(value)) => Ok(Some(*value)),
        Some(Value::Null) | None => Ok(None),
        Some(_) => Err(ProxyError::bad_request(format!("{key} must be a boolean"))),
    }
}

fn optional_u64(payload: &Map<String, Value>, key: &str) -> Result<Option<u64>, ProxyError> {
    match payload.get(key) {
        Some(Value::Number(value)) => value
            .as_u64()
            .ok_or_else(|| ProxyError::bad_request(format!("{key} must be a positive integer")))
            .map(Some),
        Some(Value::Null) | None => Ok(None),
        Some(_) => Err(ProxyError::bad_request(format!("{key} must be a number"))),
    }
}

fn empty_usage() -> Value {
    static EMPTY: OnceLock<Value> = OnceLock::new();
    EMPTY
        .get_or_init(|| {
            json!({
                "input_tokens": 0,
                "input_tokens_details": { "cached_tokens": 0 },
                "output_tokens": 0,
                "output_tokens_details": { "reasoning_tokens": 0 },
                "total_tokens": 0,
            })
        })
        .clone()
}

pub fn unix_timestamp() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usage_from_upstream_with_cached_tokens() {
        let upstream = json!({
            "prompt_tokens": 1000,
            "completion_tokens": 200,
            "total_tokens": 1200,
            "prompt_tokens_details": {
                "cached_tokens": 800
            },
            "completion_tokens_details": {
                "reasoning_tokens": 50
            }
        });
        let usage = usage_from_upstream(Some(&upstream));
        assert_eq!(usage["input_tokens"], 1000);
        assert_eq!(usage["output_tokens"], 200);
        assert_eq!(usage["total_tokens"], 1200);
        assert_eq!(usage["input_tokens_details"]["cached_tokens"], 800);
        assert_eq!(usage["output_tokens_details"]["reasoning_tokens"], 50);
    }

    #[test]
    fn usage_from_upstream_with_zero_cached_tokens() {
        let upstream = json!({
            "prompt_tokens": 500,
            "completion_tokens": 100,
            "prompt_tokens_details": { "cached_tokens": 0 }
        });
        let usage = usage_from_upstream(Some(&upstream));
        assert_eq!(usage["input_tokens"], 500);
        assert_eq!(usage["input_tokens_details"]["cached_tokens"], 0);
    }

    #[test]
    fn usage_from_upstream_without_prompt_tokens_details() {
        let upstream = json!({
            "prompt_tokens": 300,
            "completion_tokens": 50,
            "total_tokens": 350
        });
        let usage = usage_from_upstream(Some(&upstream));
        assert_eq!(usage["input_tokens"], 300);
        assert_eq!(usage["input_tokens_details"]["cached_tokens"], 0);
    }

    #[test]
    fn usage_from_upstream_none_returns_empty() {
        let usage = usage_from_upstream(None);
        assert_eq!(usage["input_tokens"], 0);
        assert_eq!(usage["output_tokens"], 0);
        assert_eq!(usage["total_tokens"], 0);
        assert_eq!(usage["input_tokens_details"]["cached_tokens"], 0);
        assert_eq!(usage["output_tokens_details"]["reasoning_tokens"], 0);
    }

    #[test]
    fn usage_from_upstream_fully_cached() {
        let upstream = json!({
            "prompt_tokens": 1000,
            "completion_tokens": 200,
            "total_tokens": 1200,
            "prompt_tokens_details": {
                "cached_tokens": 1000
            }
        });
        let usage = usage_from_upstream(Some(&upstream));
        assert_eq!(usage["input_tokens"], 1000);
        assert_eq!(usage["input_tokens_details"]["cached_tokens"], 1000);
    }

    #[test]
    fn usage_from_upstream_total_auto_calculated() {
        let upstream = json!({
            "prompt_tokens": 100,
            "completion_tokens": 50
        });
        let usage = usage_from_upstream(Some(&upstream));
        assert_eq!(usage["total_tokens"], 150);
    }

    #[test]
    fn merge_usage_accumulates_all_fields() {
        let mut total = usage_from_upstream(Some(&json!({
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "prompt_tokens_details": { "cached_tokens": 80 },
            "completion_tokens_details": { "reasoning_tokens": 10 },
        })));
        let round2 = usage_from_upstream(Some(&json!({
            "prompt_tokens": 200,
            "completion_tokens": 100,
            "total_tokens": 300,
            "prompt_tokens_details": { "cached_tokens": 150 },
            "completion_tokens_details": { "reasoning_tokens": 30 },
        })));
        merge_usage(&mut total, &round2);
        assert_eq!(total["input_tokens"], 300);
        assert_eq!(total["output_tokens"], 150);
        assert_eq!(total["total_tokens"], 450);
        assert_eq!(total["input_tokens_details"]["cached_tokens"], 230);
        assert_eq!(total["output_tokens_details"]["reasoning_tokens"], 40);
    }

    #[test]
    fn merge_usage_with_zero_round() {
        let mut total = usage_from_upstream(Some(&json!({
            "prompt_tokens": 100,
            "completion_tokens": 50,
        })));
        merge_usage(&mut total, &usage_from_upstream(None));
        assert_eq!(total["input_tokens"], 100);
        assert_eq!(total["output_tokens"], 50);
    }
}
