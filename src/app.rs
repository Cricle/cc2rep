use std::{
    collections::{BTreeMap, HashMap},
    convert::Infallible,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use async_stream::stream;
use axum::{
    Json, Router,
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
    routing::{get, post},
};
use futures_util::StreamExt;
use serde_json::{Value, json};
use tokio::sync::RwLock;
use tracing::warn;

use crate::{
    config::Settings,
    error::ProxyError,
    protocol::analyze_protocol,
    store::{ResponseStore, StoredResponse},
    translate::{
        AssistantTurn, RequestContext, ToolCall, build_cancelled_response, build_content_part,
        build_failed_response, build_function_call_item, build_history_message,
        build_in_progress_response, build_message_item, build_reasoning_item, build_response,
        function_item_id, message_output_offset, parse_assistant_turn_from_response,
        reasoning_output_offset, translate_request, unix_timestamp, usage_from_upstream,
    },
    upstream::UpstreamClient,
};

#[derive(Clone)]
struct AppState {
    settings: Arc<Settings>,
    upstream: UpstreamClient,
    store: ResponseStore,
    inflight: InflightRegistry,
}

#[derive(Clone, Default)]
struct InflightRegistry {
    inner: Arc<RwLock<HashMap<String, Arc<AtomicBool>>>>,
}

impl InflightRegistry {
    async fn start(&self, response_id: String) -> Arc<AtomicBool> {
        let token = Arc::new(AtomicBool::new(false));
        self.inner.write().await.insert(response_id, token.clone());
        token
    }

    async fn cancel(&self, response_id: &str) -> bool {
        let guard = self.inner.read().await;
        let Some(flag) = guard.get(response_id) else {
            return false;
        };
        flag.store(true, Ordering::SeqCst);
        true
    }

    async fn finish(&self, response_id: &str) {
        self.inner.write().await.remove(response_id);
    }
}

pub fn build_router(settings: Settings) -> Router {
    let settings = Arc::new(settings);
    let upstream = UpstreamClient::new(settings.clone()).expect("invalid settings");
    let state = AppState {
        settings,
        upstream,
        store: ResponseStore::new(),
        inflight: InflightRegistry::default(),
    };

    Router::new()
        .route("/healthz", get(healthz))
        .route("/v1/responses", post(create_response))
        .route(
            "/v1/responses/{response_id}",
            get(get_response).delete(delete_response),
        )
        .route(
            "/v1/responses/{response_id}/input_items",
            get(list_input_items),
        )
        .route("/v1/responses/{response_id}/cancel", post(cancel_response))
        .with_state(state)
}

async fn healthz() -> impl IntoResponse {
    Json(json!({ "ok": true }))
}

async fn create_response(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(payload): Json<Value>,
) -> Result<Response, ProxyError> {
    authorize(&state.settings, &headers)?;

    let object = payload
        .as_object()
        .ok_or_else(|| ProxyError::bad_request("request body must be a JSON object"))?;
    let report = analyze_protocol(object);
    if state.settings.strict_protocol {
        if let Some(message) = report.strict_error() {
            return Err(ProxyError::bad_request(message));
        }
    }

    let previous_messages = load_previous_messages(&state.store, object).await?;
    let translated = translate_request(object, &state.settings, &report, &previous_messages)?;

    if translated.context.store {
        state
            .store
            .put(
                translated.context.response_id.clone(),
                StoredResponse {
                    response: build_in_progress_response(&translated.context),
                    input_items: translated.input_items.clone(),
                    request_messages: translated.request_messages.clone(),
                },
            )
            .await?;
    }

    if translated.context.stream {
        let cancel_flag = state
            .inflight
            .start(translated.context.response_id.clone())
            .await;
        let response = state
            .upstream
            .chat_stream(&translated.upstream_payload)
            .await?;
        Ok(stream_response(
            state.store.clone(),
            state.inflight.clone(),
            cancel_flag,
            response,
            translated,
        ))
    } else {
        let upstream = state
            .upstream
            .chat_json(&translated.upstream_payload)
            .await?;
        let turn = parse_assistant_turn_from_response(&upstream)?;
        let usage = usage_from_upstream(upstream.get("usage"));
        let payload = build_response(&translated.context, &turn, usage, unix_timestamp());
        store_final_response(&state.store, &translated, payload.clone()).await?;
        Ok((StatusCode::OK, Json(payload)).into_response())
    }
}

async fn get_response(
    State(state): State<AppState>,
    Path(response_id): Path<String>,
    headers: HeaderMap,
) -> Result<Response, ProxyError> {
    authorize(&state.settings, &headers)?;
    let stored =
        state.store.get(&response_id).await?.ok_or_else(|| {
            ProxyError::bad_request(format!("response `{response_id}` not found"))
        })?;
    Ok((StatusCode::OK, Json(stored.response)).into_response())
}

async fn list_input_items(
    State(state): State<AppState>,
    Path(response_id): Path<String>,
    headers: HeaderMap,
) -> Result<Response, ProxyError> {
    authorize(&state.settings, &headers)?;
    let stored =
        state.store.get(&response_id).await?.ok_or_else(|| {
            ProxyError::bad_request(format!("response `{response_id}` not found"))
        })?;
    Ok((
        StatusCode::OK,
        Json(json!({
            "object": "list",
            "data": stored.input_items,
            "has_more": false,
            "first_id": null,
            "last_id": null,
        })),
    )
        .into_response())
}

async fn delete_response(
    State(state): State<AppState>,
    Path(response_id): Path<String>,
    headers: HeaderMap,
) -> Result<Response, ProxyError> {
    authorize(&state.settings, &headers)?;
    let deleted = state.store.delete(&response_id).await?.is_some();
    if !deleted {
        return Err(ProxyError::bad_request(format!(
            "response `{response_id}` not found"
        )));
    }
    Ok((
        StatusCode::OK,
        Json(json!({"id": response_id, "object": "response.deleted", "deleted": true})),
    )
        .into_response())
}

async fn cancel_response(
    State(state): State<AppState>,
    Path(response_id): Path<String>,
    headers: HeaderMap,
) -> Result<Response, ProxyError> {
    authorize(&state.settings, &headers)?;
    let Some(stored) = state.store.get(&response_id).await? else {
        return Err(ProxyError::bad_request(format!(
            "response `{response_id}` not found"
        )));
    };

    let status = stored
        .response
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("completed");
    if status != "in_progress" {
        return Ok((StatusCode::OK, Json(stored.response)).into_response());
    }

    let _ = state.inflight.cancel(&response_id).await;
    let context = context_from_response(&stored.response)?;
    let turn = assistant_turn_from_output(
        stored
            .response
            .get("output")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or(&[]),
    )?;
    let cancelled =
        build_cancelled_response(&context, &turn, usage_from_upstream(None), unix_timestamp());
    state
        .store
        .update_response(&response_id, cancelled.clone())
        .await?;
    Ok((StatusCode::OK, Json(cancelled)).into_response())
}

fn stream_response(
    store: ResponseStore,
    inflight: InflightRegistry,
    cancel_flag: Arc<AtomicBool>,
    response: reqwest::Response,
    translated: crate::translate::TranslatedRequest,
) -> Response {
    let context = translated.context.clone();
    let translated_for_store = translated.clone();
    let response_id = context.response_id.clone();
    let event_stream = stream! {
        let mut sequence_number: u64 = 0;
        let mut state = StreamState::new();

        yield Ok::<Event, Infallible>(json_event("response.created", &mut sequence_number, json!({
            "type": "response.created",
            "response": build_in_progress_response(&context),
        })));

        let mut parser = SseParser::default();
        let mut stream = response.bytes_stream();
        let mut failed_message: Option<String> = None;
        let mut cancelled = false;

        'outer: while let Some(chunk) = stream.next().await {
            if cancel_flag.load(Ordering::SeqCst) {
                cancelled = true;
                break 'outer;
            }
            match chunk {
                Ok(bytes) => {
                    let chunk_text = String::from_utf8_lossy(&bytes);
                    for event_data in parser.push(&chunk_text) {
                        if cancel_flag.load(Ordering::SeqCst) {
                            cancelled = true;
                            break 'outer;
                        }
                        if event_data == "[DONE]" {
                            break 'outer;
                        }

                        match serde_json::from_str::<Value>(&event_data) {
                            Ok(value) => {
                                if let Some(usage_value) = value.get("usage") {
                                    state.usage = usage_from_upstream(Some(usage_value));
                                }
                                match apply_stream_delta(&mut state, &value, &context, &mut sequence_number) {
                                    Ok(events) => {
                                        for event in events {
                                            yield Ok::<Event, Infallible>(event);
                                        }
                                    }
                                    Err(err) => {
                                        failed_message = Some(err.to_string());
                                        break 'outer;
                                    }
                                }
                            }
                            Err(err) => {
                                failed_message = Some(format!("failed to parse upstream SSE JSON: {err}"));
                                break 'outer;
                            }
                        }
                    }
                }
                Err(err) => {
                    failed_message = Some(err.to_string());
                    break;
                }
            }
        }

        let assistant_turn = state.assistant_turn();

        if cancelled {
            let cancelled_response = build_cancelled_response(
                &context,
                &assistant_turn,
                state.usage.clone(),
                unix_timestamp(),
            );
            let _ = store_final_response(&store, &translated_for_store, cancelled_response.clone()).await;
            let _ = inflight.finish(&response_id).await;
            yield Ok::<Event, Infallible>(json_event("response.completed", &mut sequence_number, json!({
                "type": "response.completed",
                "response": cancelled_response,
            })));
            return;
        }

        if let Some(message) = failed_message {
            warn!("stream failed: {message}");
            let failed = build_failed_response(
                &context,
                &assistant_turn,
                unix_timestamp(),
                &message,
                "upstream_stream_error",
            );
            let _ = store_final_response(&store, &translated_for_store, failed.clone()).await;
            let _ = inflight.finish(&response_id).await;
            yield Ok::<Event, Infallible>(json_event("response.failed", &mut sequence_number, json!({
                "type": "response.failed",
                "response": failed,
            })));
            return;
        }

        for event in finalize_stream_items(&assistant_turn, &context, &mut sequence_number) {
            yield Ok::<Event, Infallible>(event);
        }

        let final_response = build_response(
            &context,
            &assistant_turn,
            state.usage.clone(),
            unix_timestamp(),
        );
        let _ = store_final_response(&store, &translated_for_store, final_response.clone()).await;
        let _ = inflight.finish(&response_id).await;
        yield Ok::<Event, Infallible>(json_event("response.completed", &mut sequence_number, json!({
            "type": "response.completed",
            "response": final_response,
        })));
    };

    Sse::new(event_stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

async fn load_previous_messages(
    store: &ResponseStore,
    payload: &serde_json::Map<String, Value>,
) -> Result<Vec<Value>, ProxyError> {
    let Some(previous_response_id) = payload.get("previous_response_id").and_then(Value::as_str)
    else {
        return Ok(Vec::new());
    };
    let stored = store.get(previous_response_id).await?.ok_or_else(|| {
        ProxyError::bad_request(format!(
            "previous_response_id `{previous_response_id}` not found"
        ))
    })?;
    let mut messages = stored.request_messages.clone();
    if let Some(output) = stored.response.get("output").and_then(Value::as_array) {
        let turn = assistant_turn_from_output(output)?;
        if let Some(message) = build_history_message(&turn) {
            messages.push(message);
        }
    }
    Ok(messages)
}

async fn store_final_response(
    store: &ResponseStore,
    translated: &crate::translate::TranslatedRequest,
    response: Value,
) -> Result<(), ProxyError> {
    if !translated.context.store {
        return Ok(());
    }
    store
        .put(
            translated.context.response_id.clone(),
            StoredResponse {
                response,
                input_items: translated.input_items.clone(),
                request_messages: translated.request_messages.clone(),
            },
        )
        .await
}

fn authorize(settings: &Settings, headers: &HeaderMap) -> Result<(), ProxyError> {
    let header = headers
        .get("authorization")
        .ok_or(ProxyError::Unauthorized)?
        .to_str()
        .map_err(|_| ProxyError::Unauthorized)?;
    let token = header
        .strip_prefix("Bearer ")
        .ok_or(ProxyError::Unauthorized)?;
    if token == settings.proxy_api_key {
        Ok(())
    } else {
        Err(ProxyError::Unauthorized)
    }
}

fn json_event(name: &str, sequence_number: &mut u64, mut payload: Value) -> Event {
    if let Some(map) = payload.as_object_mut() {
        map.insert("sequence_number".to_owned(), json!(*sequence_number));
    }
    *sequence_number += 1;
    Event::default()
        .event(name)
        .json_data(payload)
        .expect("valid SSE event")
}

fn finalize_stream_items(
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

fn apply_stream_delta(
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

    if let Some(reasoning_delta) = extract_stream_reasoning_delta(delta) {
        if !reasoning_delta.is_empty() {
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
                let output_index = reasoning_output_offset(&state.assistant_turn());
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
            let output_index = reasoning_output_offset(&state.assistant_turn());
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
        for raw_tool_call in tool_calls {
            let index = raw_tool_call
                .get("index")
                .and_then(Value::as_u64)
                .unwrap_or(state.tool_calls.len() as u64) as usize;
            let output_offset = message_output_offset(&state.assistant_turn());
            let entry = state
                .tool_calls
                .entry(index)
                .or_insert_with(|| StreamToolCall {
                    call_id: raw_tool_call
                        .get("id")
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                        .unwrap_or_else(|| format!("call_{}", index + 1)),
                    name: String::new(),
                    arguments: String::new(),
                    added: false,
                });
            if let Some(id) = raw_tool_call.get("id").and_then(Value::as_str) {
                entry.call_id = id.to_owned();
            }
            if let Some(function) = raw_tool_call.get("function").and_then(Value::as_object) {
                if let Some(name) = function.get("name").and_then(Value::as_str) {
                    entry.name = name.to_owned();
                }
                if let Some(arguments) = function.get("arguments") {
                    let delta = match arguments {
                        Value::String(text) => text.clone(),
                        _ => arguments.to_string(),
                    };
                    if !delta.is_empty() {
                        if !entry.added {
                            entry.added = true;
                            events.push(json_event(
                                "response.output_item.added",
                                sequence_number,
                                json!({
                                    "type": "response.output_item.added",
                                    "output_index": output_offset + index,
                                    "item": build_function_call_item(&ToolCall {
                                        call_id: entry.call_id.clone(),
                                        name: entry.name.clone(),
                                        arguments: entry.arguments.clone(),
                                    }, "in_progress"),
                                }),
                            ));
                        }
                        entry.arguments.push_str(&delta);
                        events.push(json_event(
                            "response.function_call_arguments.delta",
                            sequence_number,
                            json!({
                                "type": "response.function_call_arguments.delta",
                                "output_index": output_offset + index,
                                "item_id": function_item_id(&entry.call_id),
                                "delta": delta,
                            }),
                        ));
                    }
                }
            }
        }
    }

    Ok(events)
}

fn extract_stream_reasoning_delta(delta: &serde_json::Map<String, Value>) -> Option<String> {
    for key in [
        "reasoning_content",
        "reasoning",
        "thinking",
        "reasoning_summary",
    ] {
        if let Some(value) = delta.get(key) {
            return flatten_reasoning_value(value);
        }
    }
    None
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

fn assistant_turn_from_output(output: &[Value]) -> Result<AssistantTurn, ProxyError> {
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

fn context_from_response(response: &Value) -> Result<RequestContext, ProxyError> {
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
    })
}

struct StreamState {
    reasoning_started: bool,
    message_started: bool,
    reasoning: String,
    text: String,
    tool_calls: BTreeMap<usize, StreamToolCall>,
    usage: Value,
}

impl StreamState {
    fn new() -> Self {
        Self {
            reasoning_started: false,
            message_started: false,
            reasoning: String::new(),
            text: String::new(),
            tool_calls: BTreeMap::new(),
            usage: usage_from_upstream(None),
        }
    }

    fn assistant_turn(&self) -> AssistantTurn {
        let tool_calls = self
            .tool_calls
            .values()
            .map(|tool_call| ToolCall {
                call_id: tool_call.call_id.clone(),
                name: tool_call.name.clone(),
                arguments: tool_call.arguments.clone(),
            })
            .collect();
        AssistantTurn {
            reasoning: self.reasoning.clone(),
            text: self.text.clone(),
            tool_calls,
        }
    }
}

#[derive(Default)]
struct StreamToolCall {
    call_id: String,
    name: String,
    arguments: String,
    added: bool,
}

#[derive(Default)]
struct SseParser {
    buffer: String,
}

impl SseParser {
    fn push(&mut self, chunk: &str) -> Vec<String> {
        self.buffer.push_str(chunk);
        let mut events = Vec::new();
        while let Some(index) = self.buffer.find("\n\n") {
            let raw = self.buffer[..index].to_owned();
            self.buffer.drain(..index + 2);
            if let Some(event) = parse_sse_event(&raw) {
                events.push(event);
            }
        }
        events
    }
}

fn parse_sse_event(raw: &str) -> Option<String> {
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

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use axum::{body::Body, http::Request, response::sse::Event, routing::post};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    use super::*;

    fn settings(upstream_base_url: String, image_input: bool) -> Settings {
        Settings {
            proxy_host: "127.0.0.1".to_owned(),
            proxy_port: 8800,
            proxy_api_key: "proxy-secret".to_owned(),
            upstream_base_url,
            upstream_chat_path: "/v1/chat/completions".to_owned(),
            upstream_model: "deepseek-chat".to_owned(),
            upstream_api_key: "upstream-secret".to_owned(),
            upstream_headers: Default::default(),
            upstream_api_key_header_name: "Authorization".to_owned(),
            upstream_api_key_prefix: "Bearer ".to_owned(),
            request_timeout_seconds: 10.0,
            strict_protocol: false,
            upstream_supports_image_input: image_input,
            upstream_body: Default::default(),
            model_aliases: [("gpt-5-codex".to_owned(), "deepseek-chat".to_owned())]
                .into_iter()
                .collect(),
        }
    }

    async fn spawn_upstream() -> String {
        let app = Router::new().route(
            "/v1/chat/completions",
            post(|Json(payload): Json<Value>| async move {
                if payload.get("stream").and_then(Value::as_bool) == Some(true) {
                    let stream = stream! {
                        yield Ok::<Event, Infallible>(Event::default().data(r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1714444444,"model":"deepseek-chat","choices":[{"index":0,"delta":{"reasoning_content":"Think "},"finish_reason":null}]}"#));
                        yield Ok::<Event, Infallible>(Event::default().data(r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1714444444,"model":"deepseek-chat","choices":[{"index":0,"delta":{"reasoning_content":"carefully.","content":"Final"},"finish_reason":null}]}"#));
                        tokio::time::sleep(Duration::from_millis(50)).await;
                        yield Ok::<Event, Infallible>(Event::default().data(r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1714444444,"model":"deepseek-chat","choices":[{"index":0,"delta":{"content":" answer"},"finish_reason":"stop"}],"usage":{"prompt_tokens":11,"completion_tokens":4,"total_tokens":15}}"#));
                        yield Ok::<Event, Infallible>(Event::default().data("[DONE]"));
                    };
                    Sse::new(stream).into_response()
                } else {
                    let has_image = payload
                        .get("messages")
                        .and_then(Value::as_array)
                        .and_then(|messages| messages.first())
                        .and_then(|message| message.get("content"))
                        .map(|content| content.is_array())
                        .unwrap_or(false);
                    Json(json!({
                        "id": "chatcmpl-1",
                        "object": "chat.completion",
                        "created": 1714444444,
                        "model": "deepseek-chat",
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": if has_image { "Vision answer" } else { "Hello from upstream" },
                                "reasoning_content": "I checked the constraints."
                            },
                            "finish_reason": "stop"
                        }],
                        "usage": {
                            "prompt_tokens": 11,
                            "completion_tokens": 4,
                            "total_tokens": 15
                        }
                    }))
                    .into_response()
                }
            }),
        );

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind test upstream");
        let addr = listener.local_addr().expect("local addr");
        tokio::spawn(async move {
            axum::serve(listener, app).await.expect("serve upstream");
        });
        format!("http://{addr}")
    }

    async fn send(router: &Router, request: Request<Body>) -> axum::response::Response {
        router.clone().oneshot(request).await.expect("response")
    }

    #[tokio::test]
    async fn non_stream_reasoning_is_mapped_to_output_item() {
        let upstream_base_url = spawn_upstream().await;
        let router = build_router(settings(upstream_base_url, false));

        let response = send(
            &router,
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header("authorization", "Bearer proxy-secret")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"model":"gpt-5-codex","input":"Say hello"}).to_string(),
                ))
                .expect("request"),
        )
        .await;

        assert_eq!(response.status(), StatusCode::OK);
        let body = response
            .into_body()
            .collect()
            .await
            .expect("collect")
            .to_bytes();
        let payload: Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(payload["output"][0]["type"], "reasoning");
        assert_eq!(payload["output"][1]["type"], "message");
    }

    #[tokio::test]
    async fn stream_reasoning_events_are_emitted() {
        let upstream_base_url = spawn_upstream().await;
        let router = build_router(settings(upstream_base_url, false));

        let response = send(
            &router,
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header("authorization", "Bearer proxy-secret")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"model":"gpt-5-codex","input":"Think","stream":true}).to_string(),
                ))
                .expect("request"),
        )
        .await;

        assert_eq!(response.status(), StatusCode::OK);
        let body = String::from_utf8(
            response
                .into_body()
                .collect()
                .await
                .expect("collect")
                .to_bytes()
                .to_vec(),
        )
        .expect("utf8");
        assert!(body.contains("event: response.reasoning_summary_text.delta"));
        assert!(body.contains("event: response.reasoning_summary_text.done"));
        assert!(body.contains("event: response.completed"));
    }

    #[tokio::test]
    async fn image_input_requires_flag() {
        let upstream_base_url = spawn_upstream().await;
        let router = build_router(settings(upstream_base_url, false));

        let response = send(
            &router,
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header("authorization", "Bearer proxy-secret")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model":"gpt-5-codex",
                        "input":[{
                            "type":"message",
                            "role":"user",
                            "content":[{"type":"input_image","image_url":"https://example.test/cat.png"}]
                        }]
                    })
                    .to_string(),
                ))
                .expect("request"),
        )
        .await;

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn image_input_is_forwarded_when_enabled() {
        let upstream_base_url = spawn_upstream().await;
        let router = build_router(settings(upstream_base_url, true));

        let response = send(
            &router,
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header("authorization", "Bearer proxy-secret")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model":"gpt-5-codex",
                        "input":[{
                            "type":"message",
                            "role":"user",
                            "content":[{"type":"input_image","image_url":"https://example.test/cat.png"}]
                        }]
                    })
                    .to_string(),
                ))
                .expect("request"),
        )
        .await;

        assert_eq!(response.status(), StatusCode::OK);
        let body = response
            .into_body()
            .collect()
            .await
            .expect("collect")
            .to_bytes();
        let payload: Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(payload["output_text"], "Vision answer");
    }

    #[tokio::test]
    async fn cancel_marks_inflight_stream_as_cancelled() {
        let upstream_base_url = spawn_upstream().await;
        let router = build_router(settings(upstream_base_url, false));

        let create = send(
            &router,
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header("authorization", "Bearer proxy-secret")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"model":"gpt-5-codex","input":"Think","stream":true,"store":true})
                        .to_string(),
                ))
                .expect("request"),
        );

        tokio::time::sleep(Duration::from_millis(10)).await;

        let stored_create = send(
            &router,
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header("authorization", "Bearer proxy-secret")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"model":"gpt-5-codex","input":"seed"}).to_string(),
                ))
                .expect("request"),
        )
        .await;
        let stored_body = stored_create
            .into_body()
            .collect()
            .await
            .expect("collect")
            .to_bytes();
        let stored_payload: Value = serde_json::from_slice(&stored_body).expect("json");
        let response_id = stored_payload["id"]
            .as_str()
            .expect("response id")
            .to_owned();

        let _stream_response = create.await;

        let cancelled = send(
            &router,
            Request::builder()
                .method("POST")
                .uri(format!("/v1/responses/{response_id}/cancel"))
                .header("authorization", "Bearer proxy-secret")
                .body(Body::empty())
                .expect("request"),
        )
        .await;

        assert!(matches!(
            cancelled.status(),
            StatusCode::OK | StatusCode::BAD_REQUEST
        ));
    }
}
