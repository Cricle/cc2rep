use std::{
    collections::BTreeMap,
    sync::atomic::{AtomicBool, Ordering},
};

use axum::response::sse::Event;
use futures_util::StreamExt;
use serde_json::{Value, json};

use crate::{
    error::ProxyError,
    translate::{
        AssistantTurn, RequestContext, ToolCall, build_content_part, build_function_call_item,
        build_message_item, build_reasoning_item, extract_first_reasoning, function_item_id,
        message_output_offset, reasoning_output_offset, usage_from_upstream,
    },
};

pub(crate) struct StreamState {
    pub reasoning_started: bool,
    pub message_started: bool,
    pub reasoning: String,
    pub text: String,
    pub tool_calls: BTreeMap<usize, StreamToolCall>,
    pub usage: Value,
}

#[derive(Default)]
pub(crate) struct StreamToolCall {
    pub call_id: String,
    pub name: String,
    pub arguments: String,
    pub added: bool,
}

impl StreamState {
    pub fn new() -> Self {
        Self {
            reasoning_started: false,
            message_started: false,
            reasoning: String::new(),
            text: String::new(),
            tool_calls: BTreeMap::new(),
            usage: usage_from_upstream(None),
        }
    }

    pub fn take_turn(&mut self) -> AssistantTurn {
        AssistantTurn {
            reasoning: std::mem::take(&mut self.reasoning),
            text: std::mem::take(&mut self.text),
            tool_calls: self
                .tool_calls
                .values()
                .map(|tc| ToolCall {
                    call_id: tc.call_id.clone(),
                    name: tc.name.clone(),
                    arguments: tc.arguments.clone(),
                })
                .collect(),
        }
    }

    pub fn to_turn(&self) -> AssistantTurn {
        AssistantTurn {
            reasoning: self.reasoning.clone(),
            text: self.text.clone(),
            tool_calls: self
                .tool_calls
                .values()
                .map(|tc| ToolCall {
                    call_id: tc.call_id.clone(),
                    name: tc.name.clone(),
                    arguments: tc.arguments.clone(),
                })
                .collect(),
        }
    }

    pub fn has_reasoning(&self) -> bool {
        !self.reasoning.is_empty()
    }

    #[allow(dead_code)]
    pub fn has_message(&self) -> bool {
        !self.text.is_empty() || self.tool_calls.is_empty()
    }
}

pub(crate) enum StreamRoundOutcome {
    Completed {
        turn: AssistantTurn,
        usage: Value,
    },
    Failed(String),
    Cancelled,
}

pub(crate) fn stream_round_context() -> RequestContext {
    RequestContext {
        response_id: String::new(),
        reasoning_id: String::new(),
        message_id: String::new(),
        created_at: 0,
        client_model: String::new(),
        upstream_model: String::new(),
        stream: true,
        store: false,
        parallel_tool_calls: true,
        instructions: None,
        metadata: json!({}),
        tool_choice: json!("auto"),
        tools: Vec::new(),
        max_output_tokens: None,
        max_tool_calls: None,
    }
}

pub(crate) fn json_event(
    name: &str,
    sequence_number: &mut u64,
    mut payload: Value,
) -> Event {
    if let Some(map) = payload.as_object_mut() {
        map.insert("sequence_number".to_owned(), json!(*sequence_number));
    }
    *sequence_number += 1;
    Event::default()
        .event(name)
        .json_data(payload)
        .unwrap_or_else(|_| Event::default().event(name).data("{}"))
}

pub(crate) fn finalize_stream_items(
    assistant_turn: &AssistantTurn,
    context: &RequestContext,
    sequence_number: &mut u64,
) -> Vec<Event> {
    let mut events = Vec::new();
    if !assistant_turn.reasoning.is_empty() {
        events.push(json_event(
            "response.reasoning_summary_text.done",
            sequence_number,
            json!({
                "type": "response.reasoning_summary_text.done",
                "output_index": 0,
                "item_id": context.reasoning_id,
                "text": assistant_turn.reasoning,
            }),
        ));
        events.push(json_event(
            "response.output_item.done",
            sequence_number,
            json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": build_reasoning_item(context, "completed", &assistant_turn.reasoning),
            }),
        ));
    }

    if !assistant_turn.text.is_empty() || assistant_turn.tool_calls.is_empty() {
        let message_index = reasoning_output_offset(assistant_turn);
        events.push(json_event(
            "response.output_text.done",
            sequence_number,
            json!({
                "type": "response.output_text.done",
                "output_index": message_index,
                "content_index": 0,
                "item_id": context.message_id,
                "text": assistant_turn.text,
                "logprobs": [],
            }),
        ));
        events.push(json_event(
            "response.content_part.done",
            sequence_number,
            json!({
                "type": "response.content_part.done",
                "output_index": message_index,
                "content_index": 0,
                "item_id": context.message_id,
                "part": build_content_part(&assistant_turn.text),
            }),
        ));
        events.push(json_event(
            "response.output_item.done",
            sequence_number,
            json!({
                "type": "response.output_item.done",
                "output_index": message_index,
                "item": build_message_item(context, "completed", &assistant_turn.text),
            }),
        ));
    }

    for (tool_index, tool_call) in assistant_turn.tool_calls.iter().enumerate() {
        let output_index = message_output_offset(assistant_turn) + tool_index;
        events.push(json_event(
            "response.function_call_arguments.done",
            sequence_number,
            json!({
                "type": "response.function_call_arguments.done",
                "output_index": output_index,
                "item_id": function_item_id(&tool_call.call_id),
                "arguments": tool_call.arguments,
            }),
        ));
        events.push(json_event(
            "response.output_item.done",
            sequence_number,
            json!({
                "type": "response.output_item.done",
                "output_index": output_index,
                "item": build_function_call_item(tool_call, "completed"),
            }),
        ));
    }
    events
}

pub(crate) fn apply_stream_delta(
    state: &mut StreamState,
    value: &Value,
    context: &RequestContext,
    sequence_number: &mut u64,
) -> Result<Vec<Event>, ProxyError> {
    let mut events = Vec::new();
    let choice = value
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first());
    let Some(choice) = choice else {
        return Ok(events);
    };
    let delta = choice.get("delta").and_then(Value::as_object);
    let Some(delta) = delta else {
        return Ok(events);
    };

    if let Some(reasoning_delta) = extract_first_reasoning(delta)
        && !reasoning_delta.is_empty() {
            if !state.reasoning_started {
                state.reasoning_started = true;
                events.push(json_event(
                    "response.output_item.added",
                    sequence_number,
                    json!({
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": build_reasoning_item(context, "in_progress", ""),
                    }),
                ));
            }
            state.reasoning.push_str(&reasoning_delta);
            events.push(json_event(
                "response.reasoning_summary_text.delta",
                sequence_number,
                json!({
                    "type": "response.reasoning_summary_text.delta",
                    "output_index": 0,
                    "item_id": context.reasoning_id,
                    "delta": reasoning_delta,
                }),
            ));
        }

    if let Some(content) = delta.get("content") {
        let delta_text = match content {
            Value::String(text) => text.clone(),
            Value::Array(parts) => parts
                .iter()
                .filter_map(|part| part.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join(""),
            _ => String::new(),
        };
        if !delta_text.is_empty() {
            if !state.message_started {
                state.message_started = true;
                let output_index = if state.has_reasoning() { 1 } else { 0 };
                events.push(json_event(
                    "response.output_item.added",
                    sequence_number,
                    json!({
                        "type": "response.output_item.added",
                        "output_index": output_index,
                        "item": build_message_item(context, "in_progress", ""),
                    }),
                ));
                events.push(json_event(
                    "response.content_part.added",
                    sequence_number,
                    json!({
                        "type": "response.content_part.added",
                        "output_index": output_index,
                        "content_index": 0,
                        "item_id": context.message_id,
                        "part": build_content_part(""),
                    }),
                ));
            }
            state.text.push_str(&delta_text);
            let output_index = if state.has_reasoning() { 1 } else { 0 };
            events.push(json_event(
                "response.output_text.delta",
                sequence_number,
                json!({
                    "type": "response.output_text.delta",
                    "output_index": output_index,
                    "content_index": 0,
                    "item_id": context.message_id,
                    "delta": delta_text,
                    "logprobs": [],
                }),
            ));
        }
    }

    if let Some(tool_calls) = delta.get("tool_calls").and_then(Value::as_array) {
        for tc_delta in tool_calls {
            let index = tc_delta.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
            let entry = state.tool_calls.entry(index).or_default();

            if let Some(id) = tc_delta.get("id").and_then(Value::as_str) {
                entry.call_id = id.to_owned();
            }
            if let Some(function) = tc_delta.get("function") {
                if let Some(name) = function.get("name").and_then(Value::as_str) {
                    entry.name.push_str(name);
                }
                if let Some(args) = function.get("arguments").and_then(Value::as_str) {
                    entry.arguments.push_str(args);
                }
            }

            // Collect entry data before releasing the mutable borrow
            let call_id = entry.call_id.clone();
            let name = entry.name.clone();
            let arguments = entry.arguments.clone();
            let was_added = entry.added;

            if !was_added && !call_id.is_empty() && !name.is_empty() {
                entry.added = true;
                let output_index = message_output_offset(&state.to_turn()) + index;
                events.push(json_event(
                    "response.output_item.added",
                    sequence_number,
                    json!({
                        "type": "response.output_item.added",
                        "output_index": output_index,
                        "item": build_function_call_item(
                            &ToolCall {
                                call_id: call_id.clone(),
                                name: name.clone(),
                                arguments: String::new(),
                            },
                            "in_progress",
                        ),
                    }),
                ));
                events.push(json_event(
                    "response.function_call_arguments.delta",
                    sequence_number,
                    json!({
                        "type": "response.function_call_arguments.delta",
                        "output_index": output_index,
                        "item_id": function_item_id(&call_id),
                        "delta": arguments,
                    }),
                ));
            } else if was_added {
                let output_index = message_output_offset(&state.to_turn()) + index;
                if let Some(args) = tc_delta
                    .get("function")
                    .and_then(|f| f.get("arguments"))
                    .and_then(Value::as_str)
                    && !args.is_empty() {
                        events.push(json_event(
                            "response.function_call_arguments.delta",
                            sequence_number,
                            json!({
                                "type": "response.function_call_arguments.delta",
                                "output_index": output_index,
                                "item_id": function_item_id(&call_id),
                                "delta": args,
                            }),
                        ));
                    }
            }
        }
    }

    Ok(events)
}

#[derive(Default)]
pub(crate) struct SseParser {
    buffer: String,
}

impl SseParser {
    pub fn push(&mut self, chunk: &str) -> Vec<String> {
        self.buffer.push_str(chunk);
        let mut events = Vec::new();
        while let Some(index) = self.buffer.find("\n\n") {
            let raw: String = self.buffer.drain(..index + 2).collect();
            if let Some(event) = parse_sse_event(&raw) {
                events.push(event);
            }
        }
        events
    }
}

pub(crate) fn parse_sse_event(raw: &str) -> Option<String> {
    let mut data_lines = Vec::new();
    for line in raw.lines() {
        if let Some(value) = line.strip_prefix("data:") {
            data_lines.push(value.trim_start().to_owned());
        }
    }
    if data_lines.is_empty() {
        None
    } else {
        Some(data_lines.join("\n"))
    }
}

pub(crate) async fn collect_stream_turn(
    cancel_flag: &AtomicBool,
    response: reqwest::Response,
) -> Result<StreamRoundOutcome, ProxyError> {
    let mut parser = SseParser::default();
    let mut stream = response.bytes_stream();
    let mut state = StreamState::new();

    while let Some(chunk) = stream.next().await {
        if cancel_flag.load(Ordering::SeqCst) {
            return Ok(StreamRoundOutcome::Cancelled);
        }
        match chunk {
            Ok(bytes) => {
                let chunk_text = String::from_utf8_lossy(&bytes);
                for event_data in parser.push(&chunk_text) {
                    if cancel_flag.load(Ordering::SeqCst) {
                        return Ok(StreamRoundOutcome::Cancelled);
                    }
                    if event_data == "[DONE]" {
                        return Ok(StreamRoundOutcome::Completed {
                            turn: state.take_turn(),
                            usage: state.usage.clone(),
                        });
                    }

                    let value: Value = serde_json::from_str(&event_data).map_err(|err| {
                        ProxyError::bad_request(format!("failed to parse upstream SSE JSON: {err}"))
                    })?;
                    if let Some(usage_value) = value.get("usage") {
                        state.usage = usage_from_upstream(Some(usage_value));
                    }
                    apply_stream_delta(&mut state, &value, &stream_round_context(), &mut 0)?;
                }
            }
            Err(err) => {
                return Ok(StreamRoundOutcome::Failed(err.to_string()));
            }
        }
    }

    Ok(StreamRoundOutcome::Completed {
        turn: state.take_turn(),
        usage: state.usage.clone(),
    })
}
