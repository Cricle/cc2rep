use std::time::{SystemTime, UNIX_EPOCH};

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
    let tools = normalize_tools(payload.get("tools"))?;
    let tool_choice = payload
        .get("tool_choice")
        .cloned()
        .unwrap_or_else(|| Value::String("auto".to_owned()));

    let messages = build_messages(previous_messages, &input_items, instructions.as_deref())?;
    let mut upstream: Map<String, Value> = settings.upstream_body.clone().into_iter().collect();
    upstream.insert("model".to_owned(), Value::String(upstream_model.clone()));
    upstream.insert("messages".to_owned(), Value::Array(messages.clone()));
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
            normalize_tool_choice(payload.get("tool_choice"))?,
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

    Ok(TranslatedRequest {
        upstream_payload: Value::Object(upstream),
        context,
        input_items,
        request_messages: messages,
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
) -> Result<Vec<Value>, ProxyError> {
    let mut messages = previous_messages.to_vec();
    if let Some(instructions) = instructions {
        messages.push(json!({
            "role": "system",
            "content": instructions,
        }));
    }

    let mut pending_tool_calls = Vec::new();
    for item in input_items {
        let map = item
            .as_object()
            .ok_or_else(|| ProxyError::bad_request("normalized input items must be objects"))?;
        let item_type = map.get("type").and_then(Value::as_str).unwrap_or("message");
        match item_type {
            "message" => {
                flush_pending_tool_calls(&mut messages, &mut pending_tool_calls);
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
            "function_call" => {
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
                flush_pending_tool_calls(&mut messages, &mut pending_tool_calls);
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
    flush_pending_tool_calls(&mut messages, &mut pending_tool_calls);

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

fn flush_pending_tool_calls(messages: &mut Vec<Value>, pending_tool_calls: &mut Vec<Value>) {
    if pending_tool_calls.is_empty() {
        return;
    }
    messages.push(json!({
        "role": "assistant",
        "content": null,
        "tool_calls": Value::Array(std::mem::take(pending_tool_calls)),
    }));
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

fn normalize_tool_choice(tool_choice: Option<&Value>) -> Result<Value, ProxyError> {
    let Some(value) = tool_choice else {
        return Ok(Value::String("auto".to_owned()));
    };
    match value {
        Value::String(choice) => match choice.as_str() {
            "auto" | "required" | "none" => Ok(Value::String(choice.clone())),
            other => Err(ProxyError::bad_request(format!(
                "unsupported tool_choice `{other}`"
            ))),
        },
        Value::Object(map) => {
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
    for key in [
        "reasoning_content",
        "reasoning",
        "thinking",
        "reasoning_summary",
    ] {
        if let Some(value) = message.get(key) {
            if let Some(reasoning) = flatten_reasoning_value(value) {
                if !reasoning.is_empty() {
                    return reasoning;
                }
            }
        }
    }
    String::new()
}

fn flatten_reasoning_value(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::Array(items) => {
            let joined = items
                .iter()
                .filter_map(flatten_reasoning_value)
                .collect::<Vec<_>>()
                .join("");
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
    json!({
        "input_tokens": 0,
        "input_tokens_details": { "cached_tokens": 0 },
        "output_tokens": 0,
        "output_tokens_details": { "reasoning_tokens": 0 },
        "total_tokens": 0,
    })
}

pub fn unix_timestamp() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or(0)
}
