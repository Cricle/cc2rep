use std::{
    sync::OnceLock,
    time::{SystemTime, UNIX_EPOCH},
};

use serde_json::{Map, Value, json};
use uuid::Uuid;

use crate::{config::Settings, error::ProxyError, protocol::ProtocolReport};

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
    let max_output_tokens = optional_u64(payload, "max_output_tokens")?;
    let input_items = normalize_input_items(payload.get("input"), settings)?;
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
    if !tools.is_empty() {
        upstream.insert("tools".to_owned(), Value::Array(tools.clone()));
        upstream.insert(
            "tool_choice".to_owned(),
            normalize_tool_choice(payload.get("tool_choice"), settings)?,
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
        response_id,
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

pub fn build_response(
    context: &RequestContext,
    turn: &AssistantTurn,
    usage: Value,
    completed_at: i64,
) -> Value {
    build_response_with_status(context, turn, usage, completed_at, "completed", Value::Null)
}

pub fn build_cancelled_response(
    context: &RequestContext,
    turn: &AssistantTurn,
    usage: Value,
    completed_at: i64,
) -> Value {
    build_response_with_status(
        context,
        turn,
        usage,
        completed_at,
        "cancelled",
        json!({"reason":"cancelled"}),
    )
}

pub fn build_failed_response(
    context: &RequestContext,
    turn: &AssistantTurn,
    completed_at: i64,
    message: &str,
    code: &str,
) -> Value {
    json!({
        "id": context.response_id,
        "object": "response",
        "created_at": context.created_at,
        "completed_at": completed_at,
        "status": "failed",
        "error": {
            "message": message,
            "type": "api_connection_error",
            "code": code,
        },
        "incomplete_details": { "reason": "error" },
        "instructions": context.instructions,
        "max_output_tokens": context.max_output_tokens,
        "metadata": context.metadata,
        "model": context.client_model,
        "output": build_output_items(context, turn, "incomplete"),
        "output_text": turn.text,
        "parallel_tool_calls": context.parallel_tool_calls,
        "previous_response_id": null,
        "store": context.store,
        "text": { "format": { "type": "text" } },
        "tool_choice": context.tool_choice,
        "tools": context.tools,
        "usage": empty_usage(),
    })
}

pub fn build_in_progress_response(context: &RequestContext) -> Value {
    json!({
        "id": context.response_id,
        "object": "response",
        "created_at": context.created_at,
        "completed_at": null,
        "status": "in_progress",
        "error": null,
        "incomplete_details": null,
        "instructions": context.instructions,
        "max_output_tokens": context.max_output_tokens,
        "metadata": context.metadata,
        "model": context.client_model,
        "output": [],
        "output_text": "",
        "parallel_tool_calls": context.parallel_tool_calls,
        "previous_response_id": null,
        "store": context.store,
        "text": { "format": { "type": "text" } },
        "tool_choice": context.tool_choice,
        "tools": context.tools,
        "usage": empty_usage(),
    })
}

pub fn build_reasoning_item(context: &RequestContext, status: &str, reasoning: &str) -> Value {
    json!({
        "id": context.reasoning_id,
        "type": "reasoning",
        "status": status,
        "summary": [{
            "type": "summary_text",
            "text": reasoning,
        }],
    })
}

pub fn build_message_item(context: &RequestContext, status: &str, text: &str) -> Value {
    json!({
        "id": context.message_id,
        "type": "message",
        "status": status,
        "role": "assistant",
        "phase": "final_answer",
        "content": [{
            "type": "output_text",
            "text": text,
            "annotations": [],
            "logprobs": [],
        }],
    })
}

pub fn build_function_call_item(tool_call: &ToolCall, status: &str) -> Value {
    json!({
        "id": function_item_id(&tool_call.call_id),
        "type": "function_call",
        "call_id": tool_call.call_id,
        "name": tool_call.name,
        "arguments": tool_call.arguments,
        "status": status,
    })
}

pub fn build_content_part(text: &str) -> Value {
    json!({
        "type": "output_text",
        "text": text,
        "annotations": [],
        "logprobs": [],
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
    if turn.text.is_empty() && turn.reasoning.is_empty() && turn.tool_calls.is_empty() {
        return None;
    }
    let content = if turn.text.is_empty() {
        Value::Null
    } else {
        Value::String(turn.text.clone())
    };
    let mut message = json!({
        "role": "assistant",
        "content": content,
    });
    if !turn.reasoning.is_empty() {
        message["reasoning_content"] = Value::String(turn.reasoning.clone());
    }
    if !turn.tool_calls.is_empty() {
        message["tool_calls"] = Value::Array(
            turn.tool_calls
                .iter()
                .map(|tool_call| {
                    json!({
                        "id": tool_call.call_id,
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": tool_call.arguments,
                        }
                    })
                })
                .collect(),
        );
    }
    Some(message)
}

pub fn extract_message_text(value: &Value) -> Result<String, ProxyError> {
    match value {
        Value::Null => Ok(String::new()),
        Value::String(text) => Ok(text.clone()),
        Value::Array(parts) => {
            let mut text = String::new();
            for part in parts {
                text.push_str(&extract_text_part(part)?);
            }
            Ok(text)
        }
        Value::Object(map) => {
            if let Some(text) = map.get("text").and_then(Value::as_str) {
                Ok(text.to_owned())
            } else {
                Err(ProxyError::bad_request("message content must include text"))
            }
        }
        _ => Err(ProxyError::bad_request(
            "message content must be a string or an array of text parts",
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

fn build_response_with_status(
    context: &RequestContext,
    turn: &AssistantTurn,
    usage: Value,
    completed_at: i64,
    status: &str,
    incomplete_details: Value,
) -> Value {
    json!({
        "id": context.response_id,
        "object": "response",
        "created_at": context.created_at,
        "completed_at": completed_at,
        "status": status,
        "error": null,
        "incomplete_details": incomplete_details,
        "instructions": context.instructions,
        "max_output_tokens": context.max_output_tokens,
        "metadata": context.metadata,
        "model": context.client_model,
        "output": build_output_items(context, turn, status),
        "output_text": turn.text,
        "parallel_tool_calls": context.parallel_tool_calls,
        "previous_response_id": null,
        "store": context.store,
        "text": { "format": { "type": "text" } },
        "tool_choice": context.tool_choice,
        "tools": context.tools,
        "usage": usage,
    })
}

fn build_output_items(context: &RequestContext, turn: &AssistantTurn, status: &str) -> Vec<Value> {
    let mut output = Vec::new();
    if !turn.reasoning.is_empty() {
        output.push(build_reasoning_item(context, status, &turn.reasoning));
    }
    if !turn.text.is_empty() || turn.tool_calls.is_empty() {
        output.push(build_message_item(context, status, &turn.text));
    }
    output.extend(
        turn.tool_calls
            .iter()
            .map(|tool_call| build_function_call_item(tool_call, status)),
    );
    output
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

fn normalize_input_items(
    input: Option<&Value>,
    settings: &Settings,
) -> Result<Vec<Value>, ProxyError> {
    let Some(input) = input else {
        return Err(ProxyError::bad_request("input is required"));
    };
    match input {
        Value::String(text) => Ok(vec![normalized_message(
            "user",
            vec![normalized_text_part(text)],
        )]),
        Value::Array(items) => items
            .iter()
            .map(|item| normalize_input_item(item, settings))
            .collect(),
        Value::Object(_) => Ok(vec![normalize_input_item(input, settings)?]),
        _ => Err(ProxyError::bad_request(
            "input must be a string, object, or array",
        )),
    }
}

fn normalize_input_item(item: &Value, settings: &Settings) -> Result<Value, ProxyError> {
    let map = item
        .as_object()
        .ok_or_else(|| ProxyError::bad_request("input array entries must be objects"))?;
    match map.get("type").and_then(Value::as_str) {
        Some("message") | None => {
            let role = map
                .get("role")
                .and_then(Value::as_str)
                .ok_or_else(|| ProxyError::bad_request("input message is missing role"))?;
            let content = map
                .get("content")
                .ok_or_else(|| ProxyError::bad_request("input message is missing content"))?;
            let normalized_content = normalize_content_parts(content, settings)?;
            Ok(normalized_message(role, normalized_content))
        }
        Some("input_text") | Some("text") => {
            let text = map
                .get("text")
                .and_then(Value::as_str)
                .ok_or_else(|| ProxyError::bad_request("text item is missing text"))?;
            Ok(normalized_message("user", vec![normalized_text_part(text)]))
        }
        Some("reasoning") if settings.drop_input_reasoning => Ok(json!({
            "type": "reasoning",
            "summary": Value::Null,
            "text": "",
        })),
        Some("reasoning") => Ok(json!({
            "type": "reasoning",
            "summary": map.get("summary").cloned().unwrap_or(Value::Null),
            "text": map
                .get("summary")
                .and_then(flatten_reasoning_value)
                .or_else(|| extract_first_reasoning(map))
                .unwrap_or_default(),
        })),
        Some("function_call") => {
            let name = map
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| ProxyError::bad_request("function_call items need `name`"))?;
            let call_id = map
                .get("call_id")
                .or_else(|| map.get("id"))
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    ProxyError::bad_request("function_call items need `call_id` or `id`")
                })?;
            let arguments =
                stringify_json_value(map.get("arguments").ok_or_else(|| {
                    ProxyError::bad_request("function_call items need `arguments`")
                })?)?;
            Ok(json!({
                "type": "function_call",
                "call_id": call_id,
                "name": name,
                "arguments": arguments,
            }))
        }
        Some("function_call_output") => {
            let call_id = map
                .get("call_id")
                .or_else(|| map.get("id"))
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    ProxyError::bad_request("function_call_output items need `call_id` or `id`")
                })?;
            let output = stringify_json_value(
                map.get("output")
                    .or_else(|| map.get("content"))
                    .ok_or_else(|| {
                        ProxyError::bad_request("function_call_output items need `output`")
                    })?,
            )?;
            Ok(json!({
                "type": "function_call_output",
                "call_id": call_id,
                "output": output,
            }))
        }
        Some("input_image") | Some("image_url") => {
            if !settings.upstream_supports_image_input {
                return Err(ProxyError::bad_request(
                    "image input is disabled for this upstream; enable `upstream_supports_image_input` to forward image_url content",
                ));
            }
            let image_url = map
                .get("image_url")
                .or_else(|| map.get("url"))
                .cloned()
                .ok_or_else(|| ProxyError::bad_request("input_image items need `image_url`"))?;
            Ok(normalized_message(
                "user",
                vec![normalized_image_part(image_url)],
            ))
        }
        Some(other) => Err(ProxyError::bad_request(format!(
            "unsupported input item type `{other}`"
        ))),
    }
}

fn normalize_content_parts(content: &Value, settings: &Settings) -> Result<Vec<Value>, ProxyError> {
    match content {
        Value::String(text) => Ok(vec![normalized_text_part(text)]),
        Value::Array(parts) => {
            let mut normalized = Vec::new();
            for part in parts {
                normalized.push(normalize_content_part(part, settings)?);
            }
            Ok(normalized)
        }
        Value::Object(_) => Ok(vec![normalize_content_part(content, settings)?]),
        _ => Err(ProxyError::bad_request(
            "message content must be a string, object, or array",
        )),
    }
}

fn normalize_content_part(part: &Value, settings: &Settings) -> Result<Value, ProxyError> {
    if let Some(text) = part.as_str() {
        return Ok(normalized_text_part(text));
    }
    let map = part
        .as_object()
        .ok_or_else(|| ProxyError::bad_request("content parts must be strings or objects"))?;
    let part_type = map
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("input_text");
    match part_type {
        "input_text" | "output_text" | "text" => {
            let text = map
                .get("text")
                .and_then(Value::as_str)
                .ok_or_else(|| ProxyError::bad_request("text part is missing text"))?;
            Ok(normalized_text_part(text))
        }
        "input_image" | "image_url" => {
            if !settings.upstream_supports_image_input {
                return Err(ProxyError::bad_request(
                    "image input is disabled for this upstream; enable `upstream_supports_image_input` to forward image_url content",
                ));
            }
            let image_url = map
                .get("image_url")
                .or_else(|| map.get("url"))
                .cloned()
                .ok_or_else(|| ProxyError::bad_request("input_image parts need `image_url`"))?;
            Ok(normalized_image_part(image_url))
        }
        other => Err(ProxyError::bad_request(format!(
            "unsupported content part type `{other}`"
        ))),
    }
}

fn build_messages(
    previous_messages: &[Value],
    input_items: &[Value],
    instructions: Option<&str>,
    settings: &Settings,
) -> Result<Vec<Value>, ProxyError> {
    let mut messages = previous_messages.to_vec();
    if let Some(instructions) = instructions {
        messages.push(json!({
            "role": "system",
            "content": instructions,
        }));
    }

    let mut pending_tool_calls = Vec::new();
    let mut pending_reasoning = String::new();
    for item in input_items {
        let map = item
            .as_object()
            .ok_or_else(|| ProxyError::bad_request("normalized input items must be objects"))?;
        let item_type = map.get("type").and_then(Value::as_str).unwrap_or("message");
        match item_type {
            "message" => {
                flush_pending_tool_calls(
                    &mut messages,
                    &mut pending_tool_calls,
                    &mut pending_reasoning,
                    settings,
                );
                let role = map
                    .get("role")
                    .and_then(Value::as_str)
                    .ok_or_else(|| ProxyError::bad_request("message item is missing role"))?;
                let content = map
                    .get("content")
                    .and_then(Value::as_array)
                    .ok_or_else(|| ProxyError::bad_request("message item is missing content"))?;
                let role = match role {
                    "developer" | "system" => "system",
                    "user" => "user",
                    "assistant" => "assistant",
                    other => {
                        return Err(ProxyError::bad_request(format!(
                            "unsupported input role `{other}`"
                        )));
                    }
                };
                messages.push(json!({
                    "role": role,
                    "content": upstream_content_from_parts(content),
                }));
            }
            "reasoning" => {
                if settings.drop_input_reasoning {
                    continue;
                }
                let reasoning = map
                    .get("text")
                    .and_then(Value::as_str)
                    .filter(|text| !text.is_empty())
                    .unwrap_or("");
                if !reasoning.is_empty() {
                    pending_reasoning.push_str(reasoning);
                }
            }
            "function_call" => {
                if settings.drop_tools {
                    continue;
                }
                pending_tool_calls.push(json!({
                    "id": map["call_id"],
                    "type": "function",
                    "function": {
                        "name": map["name"],
                        "arguments": map["arguments"],
                    }
                }));
            }
            "function_call_output" => {
                if settings.drop_tools {
                    continue;
                }
                flush_pending_tool_calls(
                    &mut messages,
                    &mut pending_tool_calls,
                    &mut pending_reasoning,
                    settings,
                );
                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": map["call_id"],
                    "content": map["output"],
                }));
            }
            other => {
                return Err(ProxyError::bad_request(format!(
                    "unsupported normalized input item type `{other}`"
                )));
            }
        }
    }
    flush_pending_tool_calls(
        &mut messages,
        &mut pending_tool_calls,
        &mut pending_reasoning,
        settings,
    );

    if messages.is_empty() {
        return Err(ProxyError::bad_request("input produced no chat messages"));
    }

    Ok(messages)
}

fn upstream_content_from_parts(parts: &[Value]) -> Value {
    let has_non_text = parts
        .iter()
        .any(|part| part.get("type").and_then(Value::as_str) != Some("input_text"));
    if !has_non_text && parts.len() == 1 {
        return Value::String(
            parts[0]
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_owned(),
        );
    }
    Value::Array(
        parts
            .iter()
            .map(|part| {
                match part.get("type").and_then(Value::as_str).unwrap_or("input_text") {
                    "input_text" => json!({
                        "type": "text",
                        "text": part.get("text").and_then(Value::as_str).unwrap_or(""),
                    }),
                    "input_image" => json!({
                        "type": "image_url",
                        "image_url": part.get("image_url").cloned().unwrap_or_else(|| json!({"url": ""})),
                    }),
                    _ => json!({
                        "type": "text",
                        "text": part.get("text").and_then(Value::as_str).unwrap_or(""),
                    }),
                }
            })
            .collect(),
    )
}

fn flush_pending_tool_calls(
    messages: &mut Vec<Value>,
    pending_tool_calls: &mut Vec<Value>,
    pending_reasoning: &mut String,
    settings: &Settings,
) {
    if pending_tool_calls.is_empty() {
        pending_reasoning.clear();
        return;
    }
    let mut assistant_message = json!({
        "role": "assistant",
        "content": null,
        "tool_calls": Value::Array(std::mem::take(pending_tool_calls)),
    });
    if settings.upstream_supports_reasoning_content && !pending_reasoning.is_empty() {
        assistant_message["reasoning_content"] = Value::String(std::mem::take(pending_reasoning));
    }
    messages.push(assistant_message);
}

fn normalize_tools(tools: Option<&Value>) -> Result<Vec<Value>, ProxyError> {
    let Some(Value::Array(tools)) = tools else {
        return Ok(Vec::new());
    };
    let mut normalized = Vec::new();
    for tool in tools {
        normalized.extend(convert_tool(tool)?);
    }
    Ok(normalized)
}

fn convert_tool(tool: &Value) -> Result<Vec<Value>, ProxyError> {
    let map = tool
        .as_object()
        .ok_or_else(|| ProxyError::bad_request("tools entries must be objects"))?;
    match map
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("function")
    {
        "function" => Ok(vec![convert_function_tool(map)?]),
        "namespace" => {
            let nested = map
                .get("tools")
                .and_then(Value::as_array)
                .ok_or_else(|| ProxyError::bad_request("namespace tools need a `tools` array"))?;
            let mut output = Vec::new();
            for nested_tool in nested {
                output.extend(convert_tool(nested_tool)?);
            }
            Ok(output)
        }
        "custom" => Ok(Vec::new()),
        _ => Ok(Vec::new()),
    }
}

fn convert_function_tool(map: &Map<String, Value>) -> Result<Value, ProxyError> {
    let function_map = map
        .get("function")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_else(|| {
            let mut function = Map::new();
            if let Some(name) = map.get("name") {
                function.insert("name".to_owned(), name.clone());
            }
            if let Some(description) = map.get("description") {
                function.insert("description".to_owned(), description.clone());
            }
            if let Some(parameters) = map.get("parameters") {
                function.insert("parameters".to_owned(), parameters.clone());
            }
            function
        });
    let name = function_map
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| ProxyError::bad_request("function tools need a `name`"))?;
    let parameters = function_map
        .get("parameters")
        .cloned()
        .unwrap_or_else(|| json!({"type":"object","properties":{}}));

    let mut function = Map::new();
    function.insert("name".to_owned(), Value::String(name.to_owned()));
    function.insert("parameters".to_owned(), parameters);
    if let Some(description) = function_map.get("description") {
        function.insert("description".to_owned(), description.clone());
    }

    Ok(json!({
        "type": "function",
        "function": function,
    }))
}

fn normalize_tool_choice(
    tool_choice: Option<&Value>,
    settings: &Settings,
) -> Result<Value, ProxyError> {
    let Some(value) = tool_choice else {
        return Ok(Value::String("auto".to_owned()));
    };
    match value {
        Value::String(choice) => match choice.as_str() {
            "auto" | "none" => Ok(Value::String(choice.clone())),
            "required" => {
                if settings.upstream_supports_tool_choice_required {
                    Ok(Value::String(choice.clone()))
                } else {
                    Ok(Value::String("auto".to_owned()))
                }
            }
            other => Err(ProxyError::bad_request(format!(
                "unsupported tool_choice `{other}`"
            ))),
        },
        Value::Object(map) => {
            if !settings.upstream_supports_named_tool_choice {
                return Ok(Value::String("auto".to_owned()));
            }
            let tool_type = map
                .get("type")
                .and_then(Value::as_str)
                .ok_or_else(|| ProxyError::bad_request("tool_choice objects need `type`"))?;
            if tool_type != "function" {
                return Err(ProxyError::bad_request(format!(
                    "unsupported tool_choice type `{tool_type}`"
                )));
            }
            let name = map
                .get("name")
                .or_else(|| {
                    map.get("function")
                        .and_then(|function| function.get("name"))
                })
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    ProxyError::bad_request("function tool_choice objects need `name`")
                })?;
            Ok(json!({
                "type": "function",
                "function": { "name": name }
            }))
        }
        _ => Err(ProxyError::bad_request(
            "`tool_choice` must be a string or object",
        )),
    }
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

fn normalized_message(role: &str, content: Vec<Value>) -> Value {
    json!({
        "type": "message",
        "role": role,
        "content": content,
    })
}

fn normalized_text_part(text: &str) -> Value {
    json!({
        "type": "input_text",
        "text": text,
    })
}

fn normalized_image_part(image_url: Value) -> Value {
    let image_url = match image_url {
        Value::String(url) => json!({ "url": url }),
        Value::Object(map) => Value::Object(map),
        other => json!({ "url": other.to_string() }),
    };
    json!({
        "type": "input_image",
        "image_url": image_url,
    })
}

fn stringify_json_value(value: &Value) -> Result<String, ProxyError> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::Null => Ok(String::new()),
        _ => serde_json::to_string(value)
            .map_err(|err| ProxyError::Internal(format!("failed to encode JSON value: {err}"))),
    }
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
        if let Some(value) = map.get(key) {
            if let Some(reasoning) = flatten_reasoning_value(value) {
                if !reasoning.is_empty() {
                    return Some(reasoning);
                }
            }
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
                if let Some(value) = map.get(key) {
                    if let Some(text) = flatten_reasoning_value(value) {
                        return Some(text);
                    }
                }
            }
            None
        }
        _ => None,
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
    use serde_json::json;

    use super::*;
    use crate::protocol::analyze_protocol;

    fn settings(image_input: bool) -> Settings {
        Settings {
            proxy_host: "127.0.0.1".to_owned(),
            proxy_port: 8080,
            proxy_api_key: "proxy-key".to_owned(),
            upstream_base_url: "https://api.example.com".to_owned(),
            upstream_chat_path: "/v1/chat/completions".to_owned(),
            upstream_model: "upstream-model".to_owned(),
            upstream_api_key: "upstream-key".to_owned(),
            upstream_headers: Default::default(),
            upstream_api_key_header_name: "Authorization".to_owned(),
            upstream_api_key_prefix: "Bearer ".to_owned(),
            request_timeout_seconds: 30.0,
            strict_protocol: false,
            upstream_supports_image_input: image_input,
            upstream_supports_reasoning_content: false,
            upstream_supports_tool_choice_required: false,
            upstream_supports_named_tool_choice: false,
            drop_input_reasoning: false,
            drop_tools: false,
            upstream_body: [("seed".to_owned(), json!(7))].into_iter().collect(),
            model_aliases: [("client-model".to_owned(), "aliased-model".to_owned())]
                .into_iter()
                .collect(),
            local_tools: Default::default(),
            max_auto_tool_rounds: 8,
        }
    }

    fn context() -> RequestContext {
        RequestContext {
            response_id: "resp_1".to_owned(),
            reasoning_id: "rs_1".to_owned(),
            message_id: "msg_1".to_owned(),
            created_at: 123,
            client_model: "client-model".to_owned(),
            upstream_model: "upstream-model".to_owned(),
            stream: false,
            store: true,
            parallel_tool_calls: true,
            instructions: Some("system".to_owned()),
            metadata: json!({"meta":true}),
            tool_choice: json!("auto"),
            tools: vec![json!({"type":"function"})],
            max_output_tokens: Some(55),
        }
    }

    #[test]
    fn translate_request_builds_upstream_payload_and_context() {
        let mut cfg = settings(false);
        cfg.upstream_supports_named_tool_choice = true;
        cfg.upstream_supports_reasoning_content = true;
        let payload = json!({
            "model": "client-model",
            "instructions": "be helpful",
            "stream": true,
            "store": false,
            "parallel_tool_calls": false,
            "metadata": {"request_id":"r1"},
            "max_output_tokens": 99,
            "temperature": 0.2,
            "response_format": {"type":"json_object"},
            "tool_choice": {"type":"function","name":"lookup"},
            "tools": [
                {
                    "type": "function",
                    "name": "lookup",
                    "description": "look up data",
                    "parameters": {"type":"object","properties":{"q":{"type":"string"}}}
                },
                {
                    "type": "namespace",
                    "tools": [
                        {"type":"function","function":{"name":"nested","parameters":{"type":"object"}}},
                        {"type":"custom","name":"ignored"}
                    ]
                },
                {"type":"custom","name":"ignored"}
            ],
            "input": [
                {"type":"message","role":"developer","content":"dev"},
                {"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]},
                {"type":"function_call","call_id":"call_1","name":"lookup","arguments":{"q":"v"}},
                {"type":"function_call_output","call_id":"call_1","output":{"result":1}}
            ],
            "reasoning": {"effort":"high"}
        });
        let object = payload.as_object().expect("object");
        let report = analyze_protocol(object);

        let translated = translate_request(object, &cfg, &report, &[]).expect("translate");
        let upstream = translated.upstream_payload.as_object().expect("upstream");
        assert_eq!(upstream["model"], "aliased-model");
        assert_eq!(upstream["stream"], true);
        assert_eq!(upstream["max_tokens"], 99);
        assert_eq!(upstream["temperature"], 0.2);
        assert_eq!(upstream["response_format"]["type"], "json_object");
        assert_eq!(upstream["seed"], 7);
        assert_eq!(upstream["tools"].as_array().expect("tools").len(), 2);
        assert_eq!(upstream["tool_choice"]["function"]["name"], "lookup");
        assert_eq!(translated.context.client_model, "client-model");
        assert_eq!(translated.context.upstream_model, "aliased-model");
        assert!(translated.context.stream);
        assert!(!translated.context.store);
        assert!(!translated.context.parallel_tool_calls);
        assert_eq!(
            translated.context.instructions.as_deref(),
            Some("be helpful")
        );
        assert_eq!(
            translated.context.metadata["response_proxy"]["compatibility"]["ignored_fields"][0],
            "reasoning"
        );
        assert_eq!(translated.request_messages.len(), 5);
        assert_eq!(translated.request_messages[0]["role"], "system");
        assert_eq!(translated.request_messages[1]["role"], "system");
        assert_eq!(translated.request_messages[2]["role"], "user");
        assert_eq!(translated.request_messages[3]["role"], "assistant");
        assert_eq!(translated.request_messages[4]["role"], "tool");
    }

    #[test]
    fn translate_request_accepts_string_input_and_previous_messages() {
        let payload = json!({"input":"hello"});
        let object = payload.as_object().expect("object");
        let report = analyze_protocol(object);
        let translated = translate_request(
            object,
            &settings(false),
            &report,
            &[json!({"role":"assistant","content":"previous"})],
        )
        .expect("translate");

        assert_eq!(translated.input_items[0]["role"], "user");
        assert_eq!(translated.request_messages[0]["role"], "assistant");
        assert_eq!(translated.request_messages[1]["role"], "user");
        assert_eq!(translated.request_messages[1]["content"], "hello");
    }

    #[test]
    fn response_builders_emit_expected_shapes() {
        let context = context();
        let turn = AssistantTurn {
            text: "final".to_owned(),
            reasoning: "think".to_owned(),
            tool_calls: vec![ToolCall {
                call_id: "call_1".to_owned(),
                name: "lookup".to_owned(),
                arguments: "{\"q\":\"v\"}".to_owned(),
            }],
        };

        let response = build_response(&context, &turn, json!({"total_tokens":3}), 200);
        assert_eq!(response["status"], "completed");
        assert_eq!(response["output"].as_array().expect("output").len(), 3);
        assert_eq!(response["output_text"], "final");

        let cancelled = build_cancelled_response(&context, &turn, json!({"total_tokens":1}), 201);
        assert_eq!(cancelled["status"], "cancelled");
        assert_eq!(cancelled["incomplete_details"]["reason"], "cancelled");

        let failed = build_failed_response(&context, &turn, 202, "boom", "stream_error");
        assert_eq!(failed["status"], "failed");
        assert_eq!(failed["error"]["message"], "boom");
        assert_eq!(failed["error"]["code"], "stream_error");

        let in_progress = build_in_progress_response(&context);
        assert_eq!(in_progress["status"], "in_progress");
        assert_eq!(in_progress["output"], json!([]));

        assert_eq!(
            build_reasoning_item(&context, "completed", "think")["summary"][0]["text"],
            "think"
        );
        assert_eq!(
            build_message_item(&context, "completed", "final")["content"][0]["text"],
            "final"
        );
        assert_eq!(
            build_function_call_item(&turn.tool_calls[0], "completed")["call_id"],
            "call_1"
        );
        assert_eq!(build_content_part("x")["type"], "output_text");
        assert_eq!(function_item_id("call_1"), "fc_call_1");
        assert_eq!(reasoning_output_offset(&turn), 1);
        assert_eq!(message_output_offset(&turn), 2);

        let no_reasoning = AssistantTurn {
            reasoning: String::new(),
            text: String::new(),
            tool_calls: vec![],
        };
        assert_eq!(reasoning_output_offset(&no_reasoning), 0);
        assert_eq!(message_output_offset(&no_reasoning), 1);
    }

    #[test]
    fn usage_and_history_and_parsing_helpers_work() {
        let usage = usage_from_upstream(Some(&json!({
            "prompt_tokens": 3,
            "completion_tokens": 4,
            "prompt_tokens_details": {"cached_tokens": 1},
            "completion_tokens_details": {"reasoning_tokens": 2}
        })));
        assert_eq!(usage["input_tokens"], 3);
        assert_eq!(usage["output_tokens"], 4);
        assert_eq!(usage["total_tokens"], 7);
        assert_eq!(usage["input_tokens_details"]["cached_tokens"], 1);
        assert_eq!(usage["output_tokens_details"]["reasoning_tokens"], 2);
        assert_eq!(usage_from_upstream(None)["total_tokens"], 0);

        let turn = AssistantTurn {
            text: "answer".to_owned(),
            reasoning: "think".to_owned(),
            tool_calls: vec![ToolCall {
                call_id: "call_1".to_owned(),
                name: "lookup".to_owned(),
                arguments: "{}".to_owned(),
            }],
        };
        let history = build_history_message(&turn).expect("history");
        assert_eq!(history["role"], "assistant");
        assert_eq!(history["reasoning_content"], "think");
        assert_eq!(history["tool_calls"][0]["id"], "call_1");
        assert!(build_history_message(&AssistantTurn::default()).is_none());

        let upstream = json!({
            "choices": [{
                "message": {
                    "content": [{"type":"text","text":"hello"},{"type":"input_image","image_url":{"url":"x"}}],
                    "reasoning": [{"text":"step "},"done"],
                    "tool_calls": [{
                        "id":"call_1",
                        "function":{"name":"lookup","arguments":{"x":1}}
                    }]
                }
            }]
        });
        let parsed = parse_assistant_turn_from_response(&upstream).expect("parse");
        assert_eq!(parsed.text, "hello");
        assert_eq!(parsed.reasoning, "step done");
        assert_eq!(parsed.tool_calls[0].arguments, "{\"x\":1}");
    }

    #[test]
    fn extraction_helpers_cover_supported_shapes() {
        assert_eq!(extract_message_text(&Value::Null).expect("null"), "");
        assert_eq!(extract_message_text(&json!("text")).expect("text"), "text");
        assert_eq!(
            extract_message_text(&json!({"text":"inline"})).expect("object"),
            "inline"
        );
        assert_eq!(
            extract_message_text(&json!([{"type":"text","text":"a"},"b"])).expect("array"),
            "ab"
        );

        let reasoning = extract_first_reasoning(
            json!({"thinking":[{"content":[{"text":"a"},{"summary":"b"}]}]})
                .as_object()
                .expect("object"),
        );
        assert_eq!(reasoning.as_deref(), Some("ab"));
        assert_eq!(
            flatten_reasoning_value(&json!(["a",{"text":"b"},{"summary":"c"}])),
            Some("abc".to_owned())
        );
        assert_eq!(flatten_reasoning_value(&json!("")), None);
        assert_eq!(
            extract_first_reasoning(json!({}).as_object().expect("obj")),
            None
        );
    }

    #[test]
    fn translate_request_rejects_invalid_payload_shapes() {
        let cases = [
            json!({"input":"hi","stream":"x"}),
            json!({"input":"hi","store":"x"}),
            json!({"input":"hi","parallel_tool_calls":"x"}),
            json!({"input":"hi","max_output_tokens":"x"}),
            json!({"input":"hi","max_output_tokens":-1}),
            json!({"input":"hi","metadata":[]}),
            json!({"input":"hi","text":{"format":{"type":"json_schema"}}}),
            json!({"input":[1]}),
            json!({"input":{"type":"message","content":"x"}}),
            json!({"input":{"type":"message","role":"user"}}),
            json!({"input":{"type":"text"}}),
            json!({"input":{"type":"function_call","call_id":"c","arguments":{}}}),
            json!({"input":{"type":"function_call","name":"n","arguments":{}}}),
            json!({"input":{"type":"function_call","name":"n","call_id":"c"}}),
            json!({"input":{"type":"function_call_output","call_id":"c"}}),
            json!({"input":{"type":"input_image","image_url":"https://x"}}),
            json!({"input":{"type":"unknown"}}),
            json!({"input":{"type":"message","role":"user","content":1}}),
            json!({"input":{"type":"message","role":"user","content":[1]}}),
            json!({"input":{"type":"message","role":"user","content":[{"type":"text"}]}}),
            json!({"input":{"type":"message","role":"user","content":[{"type":"input_image"}]}}),
            json!({"input":{"type":"message","role":"weird","content":"x"}}),
            json!({"input":[]}),
            json!({"input":"hi","tools":[1]}),
            json!({"input":"hi","tools":[{"type":"function"}]}),
            json!({"input":"hi","tools":[{"type":"namespace"}]}),
        ];

        for payload in cases {
            let object = payload.as_object().expect("object");
            let report = analyze_protocol(object);
            assert!(
                translate_request(object, &settings(false), &report, &[]).is_err(),
                "payload should fail: {payload}"
            );
        }

        let payload = json!({
            "input":"hi",
            "tools":[{"type":"function","name":"lookup"}],
            "tool_choice":{"type":"custom","name":"x"}
        });
        let object = payload.as_object().expect("object");
        let report = analyze_protocol(object);
        let mut named_settings = settings(false);
        named_settings.upstream_supports_named_tool_choice = true;
        assert!(translate_request(object, &named_settings, &report, &[]).is_err());

        let payload = json!({
            "input":"hi",
            "tools":[{"type":"function","name":"lookup"}],
            "tool_choice":{"type":"function"}
        });
        let object = payload.as_object().expect("object");
        let report = analyze_protocol(object);
        assert!(translate_request(object, &named_settings, &report, &[]).is_err());
    }

    #[test]
    fn translate_request_supports_image_and_object_inputs() {
        let payload = json!({
            "input": {
                "type": "message",
                "role": "user",
                "content": [
                    {"type":"input_text","text":"look"},
                    {"type":"input_image","image_url":{"url":"https://img"}}
                ]
            }
        });
        let object = payload.as_object().expect("object");
        let report = analyze_protocol(object);
        let translated =
            translate_request(object, &settings(true), &report, &[]).expect("translate");
        assert!(translated.request_messages[0]["content"].is_array());
        assert_eq!(
            translated.request_messages[0]["content"][1]["type"],
            "image_url"
        );
    }

    #[test]
    fn translate_request_ignores_reasoning_input_items() {
        let payload = json!({
            "input": [
                {
                    "type": "reasoning",
                    "summary": [
                        {"type":"summary_text","text":"thinking"},
                        " done"
                    ]
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": "hi"
                }
            ]
        });
        let object = payload.as_object().expect("object");
        let report = analyze_protocol(object);
        let translated =
            translate_request(object, &settings(false), &report, &[]).expect("translate");
        assert_eq!(translated.input_items[0]["type"], "reasoning");
        assert_eq!(translated.input_items[0]["text"], "thinking done");
        assert_eq!(translated.request_messages.len(), 1);
        assert_eq!(translated.request_messages[0]["role"], "user");
        assert_eq!(translated.request_messages[0]["content"], "hi");

        let payload = json!({
            "input": [
                {
                    "type": "reasoning",
                    "summary": [{"type":"summary_text","text":"plan first"}]
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "lookup",
                    "arguments": {"q":"weather"}
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": {"ok": true}
                }
            ]
        });
        let object = payload.as_object().expect("object");
        let report = analyze_protocol(object);
        let mut cfg = settings(false);
        cfg.upstream_supports_reasoning_content = true;
        let translated = translate_request(object, &cfg, &report, &[]).expect("translate");
        assert_eq!(translated.request_messages.len(), 2);
        assert_eq!(translated.request_messages[0]["role"], "assistant");
        assert_eq!(
            translated.request_messages[0]["reasoning_content"],
            "plan first"
        );
        assert_eq!(
            translated.request_messages[0]["tool_calls"][0]["id"],
            "call_1"
        );
        assert_eq!(translated.request_messages[1]["role"], "tool");
    }

    #[test]
    fn parse_assistant_turn_reports_invalid_upstream_shapes() {
        assert!(parse_assistant_turn_from_response(&json!({})).is_err());
        assert!(parse_assistant_turn_from_response(&json!({"choices":[{}]})).is_err());
        assert!(
            parse_assistant_turn_from_response(&json!({
                "choices":[{"message":{"content":1}}]
            }))
            .is_err()
        );
        assert!(
            parse_assistant_turn_from_response(&json!({
                "choices":[{"message":{"tool_calls":[1]}}]
            }))
            .is_err()
        );
    }

    #[test]
    fn unix_timestamp_is_non_negative() {
        assert!(unix_timestamp() >= 0);
    }

    #[test]
    fn helper_edge_cases_are_covered() {
        let tool_only = AssistantTurn {
            text: String::new(),
            reasoning: String::new(),
            tool_calls: vec![ToolCall {
                call_id: "call_1".to_owned(),
                name: "lookup".to_owned(),
                arguments: "{}".to_owned(),
            }],
        };
        let history = build_history_message(&tool_only).expect("history");
        assert_eq!(history["content"], Value::Null);
        assert_eq!(history["tool_calls"][0]["id"], "call_1");
        assert_eq!(message_output_offset(&tool_only), 0);

        assert!(extract_message_text(&json!({"no_text":true})).is_err());
        assert!(extract_message_text(&json!(1)).is_err());
        assert_eq!(
            normalize_input_items(None, &settings(false))
                .unwrap_err()
                .to_string(),
            "input is required"
        );
        assert!(normalize_input_items(Some(&json!(1)), &settings(false)).is_err());
        assert_eq!(
            normalize_input_item(&json!({"role":"user","content":"x"}), &settings(false))
                .expect("message")["type"],
            "message"
        );
        assert_eq!(
            normalize_input_item(
                &json!({"type":"function_call_output","id":"call_1","content":{"ok":true}}),
                &settings(false)
            )
            .expect("tool output")["output"],
            "{\"ok\":true}"
        );
        assert!(
            normalize_input_item(
                &json!({"type":"function_call_output","output":"x"}),
                &settings(false)
            )
            .is_err()
        );
        assert_eq!(
            normalize_input_item(&json!({"type":"text","text":"inline"}), &settings(false))
                .expect("text item")["content"][0]["text"],
            "inline"
        );
        assert_eq!(
            normalize_input_item(
                &json!({"type":"reasoning","summary":[{"text":"step "},"done"]}),
                &settings(false)
            )
            .expect("reasoning item")["text"],
            "step done"
        );
        assert_eq!(
            normalize_input_item(
                &json!({"type":"image_url","url":"https://img"}),
                &settings(true)
            )
            .expect("image")["content"][0]["image_url"]["url"],
            "https://img"
        );

        assert_eq!(
            normalize_content_parts(&json!({"type":"text","text":"x"}), &settings(false))
                .expect("object part")[0]["text"],
            "x"
        );
        assert_eq!(
            normalize_content_part(&json!("plain"), &settings(false)).expect("string part")["text"],
            "plain"
        );
        assert_eq!(
            normalize_content_part(
                &json!({"type":"image_url","url":"https://img"}),
                &settings(true)
            )
            .expect("image part")["image_url"]["url"],
            "https://img"
        );
        assert!(normalize_content_part(&json!({"type":"custom"}), &settings(false)).is_err());

        let messages = build_messages(
            &[],
            &[
                json!({"type":"reasoning","text":"thinking"}),
                json!({"type":"function_call","call_id":"call_1","name":"lookup","arguments":"{}"}),
            ],
            None,
            &settings(false),
        )
        .expect("messages");
        assert_eq!(messages[0]["role"], "assistant");
        assert!(build_messages(&[], &[json!({"type":"custom"})], None, &settings(false)).is_err());

        let mut drop_tools_settings = settings(false);
        drop_tools_settings.drop_tools = true;
        let messages = build_messages(
            &[],
            &[
                json!({"type":"reasoning","text":"thinking"}),
                json!({"type":"function_call","call_id":"call_1","name":"lookup","arguments":"{}"}),
                json!({"type":"function_call_output","call_id":"call_1","output":"{}"}),
                json!({"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}),
            ],
            None,
            &drop_tools_settings,
        )
        .expect("messages");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "user");

        assert_eq!(
            upstream_content_from_parts(&[json!({"type":"other","text":"fallback"})])[0]["text"],
            "fallback"
        );
        let mut messages = vec![];
        let mut pending = vec![];
        let mut pending_reasoning = String::new();
        flush_pending_tool_calls(
            &mut messages,
            &mut pending,
            &mut pending_reasoning,
            &settings(false),
        );
        assert!(messages.is_empty());

        let mut messages = vec![];
        let mut pending = vec![json!({"id":"call_1"})];
        let mut pending_reasoning = "think".to_owned();
        let mut reasoning_settings = settings(false);
        reasoning_settings.upstream_supports_reasoning_content = true;
        flush_pending_tool_calls(
            &mut messages,
            &mut pending,
            &mut pending_reasoning,
            &reasoning_settings,
        );
        assert_eq!(messages[0]["reasoning_content"], "think");

        assert_eq!(normalize_tools(None).expect("none"), Vec::<Value>::new());
        assert!(convert_tool(&json!(1)).is_err());
        assert_eq!(
            convert_tool(&json!({"type":"other"})).expect("ignored"),
            Vec::<Value>::new()
        );
        assert_eq!(
            convert_function_tool(
                json!({"function":{"name":"lookup"}})
                    .as_object()
                    .expect("object")
            )
            .expect("function")["function"]["parameters"]["type"],
            "object"
        );

        assert_eq!(
            normalize_tool_choice(None, &settings(false)).expect("none"),
            Value::String("auto".to_owned())
        );
        assert_eq!(
            normalize_tool_choice(Some(&json!("required")), &settings(false)).expect("required"),
            Value::String("auto".to_owned())
        );
        let mut required_settings = settings(false);
        required_settings.upstream_supports_tool_choice_required = true;
        assert_eq!(
            normalize_tool_choice(Some(&json!("required")), &required_settings).expect("required"),
            Value::String("required".to_owned())
        );
        let mut named_settings = settings(false);
        named_settings.upstream_supports_named_tool_choice = true;
        assert_eq!(
            normalize_tool_choice(
                Some(&json!({"type":"function","name":"lookup"})),
                &named_settings
            )
            .expect("object name")["function"]["name"],
            "lookup"
        );
        assert_eq!(
            normalize_tool_choice(
                Some(&json!({"type":"function","name":"lookup"})),
                &settings(false)
            )
            .expect("fallback"),
            Value::String("auto".to_owned())
        );
        assert!(normalize_tool_choice(Some(&json!("weird")), &settings(false)).is_err());
        assert_eq!(
            normalize_tool_choice(
                Some(&json!({"type":"function","function":{"name":"lookup"}})),
                &named_settings
            )
            .expect("object")["function"]["name"],
            "lookup"
        );
        assert!(normalize_tool_choice(Some(&json!(1)), &settings(false)).is_err());

        assert!(parse_tool_calls(None).expect("none").is_empty());
        let generated = parse_tool_calls(Some(&json!([{
            "function":{"name":"lookup","arguments":null}
        }])))
        .expect("generated");
        assert!(generated[0].call_id.starts_with("call_"));
        assert_eq!(generated[0].arguments, "");
        assert!(parse_tool_calls(Some(&json!([{"function":{"name":"lookup"}}]))).is_err());
        assert!(parse_tool_calls(Some(&json!([{"function":{"arguments":"{}"}}]))).is_err());

        assert_eq!(normalized_image_part(json!(7))["image_url"]["url"], "7");
        assert_eq!(stringify_json_value(&Value::Null).expect("null"), "");
        assert!(extract_text_part(&json!({"type":"text"})).is_err());
        assert_eq!(
            extract_text_part(&json!({"type":"image_url","image_url":{"url":"x"}})).expect("image"),
            ""
        );
        assert!(extract_text_part(&json!({"type":"custom"})).is_err());

        let reasoning_message = json!({"reasoning_summary":"x"});
        assert_eq!(
            extract_reasoning_content(reasoning_message.as_object().expect("object")),
            "x"
        );
        assert_eq!(
            extract_first_reasoning(
                json!({"reasoning_content":["",{"text":"x"}]})
                    .as_object()
                    .expect("object")
            )
            .as_deref(),
            Some("x")
        );
        assert_eq!(
            extract_first_reasoning(
                json!({"reasoning_content":[""]})
                    .as_object()
                    .expect("object")
            ),
            None
        );
        assert_eq!(flatten_reasoning_value(&json!([])), None);
        assert_eq!(flatten_reasoning_value(&json!([null])), None);
        assert_eq!(flatten_reasoning_value(&json!({"unknown":"x"})), None);
        assert_eq!(
            flatten_reasoning_value(&json!({"reasoning_content":"x"})),
            Some("x".to_owned())
        );
        assert_eq!(
            flatten_reasoning_value(&json!({"content":{"text":"x"}})),
            Some("x".to_owned())
        );

        let text_missing_type = json!({"text":{}});
        assert!(validate_text_format(text_missing_type.as_object().expect("obj")).is_ok());
        let no_text_field = json!({});
        assert!(validate_text_format(no_text_field.as_object().expect("obj")).is_ok());
        let plain_text_type = json!({"text":{"format":{"type":"text"}}});
        assert!(validate_text_format(plain_text_type.as_object().expect("obj")).is_ok());
        let misc = json!({"model":1,"stream":null,"max_output_tokens":null});
        let misc = misc.as_object().expect("obj");
        assert_eq!(optional_string(misc, "model"), None);
        assert_eq!(optional_bool(misc, "stream").expect("stream"), None);
        assert_eq!(optional_u64(misc, "max_output_tokens").expect("max"), None);
    }
}
