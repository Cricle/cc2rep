use serde_json::{Map, Value, json};

use crate::{
    config::Settings,
    error::ProxyError,
    probe::Capabilities,
    translate::{
        extract_first_reasoning, flatten_reasoning_value, stringify_json_value,
    },
};

pub(crate) fn normalize_input_items(
    input: Option<&Value>,
    settings: &Settings,
    capabilities: &Capabilities,
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
            .map(|item| normalize_input_item(item, settings, capabilities))
            .collect(),
        Value::Object(_) => Ok(vec![normalize_input_item(input, settings, capabilities)?]),
        _ => Err(ProxyError::bad_request(
            "input must be a string, object, or array",
        )),
    }
}

fn normalize_input_item(
    item: &Value,
    settings: &Settings,
    capabilities: &Capabilities,
) -> Result<Value, ProxyError> {
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
            let normalized_content = normalize_content_parts(content, settings, capabilities)?;
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
            if !capabilities.supports_image_input {
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

fn normalize_content_parts(
    content: &Value,
    settings: &Settings,
    capabilities: &Capabilities,
) -> Result<Vec<Value>, ProxyError> {
    match content {
        Value::String(text) => Ok(vec![normalized_text_part(text)]),
        Value::Array(parts) => {
            let mut normalized = Vec::new();
            for part in parts {
                normalized.push(normalize_content_part(part, settings, capabilities)?);
            }
            Ok(normalized)
        }
        Value::Object(_) => Ok(vec![normalize_content_part(
            content,
            settings,
            capabilities,
        )?]),
        _ => Err(ProxyError::bad_request(
            "message content must be a string, object, or array",
        )),
    }
}

fn normalize_content_part(
    part: &Value,
    _settings: &Settings,
    capabilities: &Capabilities,
) -> Result<Value, ProxyError> {
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
            if !capabilities.supports_image_input {
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

pub(crate) fn build_messages(
    previous_messages: &[Value],
    input_items: &[Value],
    instructions: Option<&str>,
    settings: &Settings,
    capabilities: &Capabilities,
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
                    capabilities,
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
                    capabilities,
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
        capabilities,
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
    capabilities: &Capabilities,
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
    if capabilities.supports_reasoning_content && !pending_reasoning.is_empty() {
        assistant_message["reasoning_content"] = Value::String(std::mem::take(pending_reasoning));
    }
    messages.push(assistant_message);
}

pub(crate) fn normalize_tools(tools: Option<&Value>) -> Result<Vec<Value>, ProxyError> {
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

pub(crate) fn normalize_tool_choice(
    tool_choice: Option<&Value>,
    _settings: &Settings,
    capabilities: &Capabilities,
) -> Result<Value, ProxyError> {
    let Some(value) = tool_choice else {
        return Ok(Value::String("auto".to_owned()));
    };
    match value {
        Value::String(choice) => match choice.as_str() {
            "auto" | "none" => Ok(Value::String(choice.clone())),
            "required" => {
                if capabilities.supports_tool_choice_required {
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
            if !capabilities.supports_named_tool_choice {
                return if capabilities.supports_tool_choice_required {
                    Ok(Value::String("required".to_owned()))
                } else {
                    Ok(Value::String("auto".to_owned()))
                };
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

pub(crate) fn normalized_message(role: &str, content: Vec<Value>) -> Value {
    json!({
        "type": "message",
        "role": role,
        "content": content,
    })
}

pub(crate) fn normalized_text_part(text: &str) -> Value {
    json!({
        "type": "input_text",
        "text": text,
    })
}

fn normalized_image_part(image_url: Value) -> Value {
    json!({
        "type": "input_image",
        "image_url": image_url,
    })
}
