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
use tracing::{info, warn};

use crate::{
    config::Settings,
    error::ProxyError,
    protocol::analyze_protocol,
    store::{ResponseStore, StoredResponse},
    tools::{ToolExecutor, append_tool_outputs},
    translate::{
        AssistantTurn, RequestContext, ToolCall, build_cancelled_response, build_content_part,
        build_failed_response, build_function_call_item, build_history_message,
        build_in_progress_response, build_message_item, build_reasoning_item, build_response,
        extract_first_reasoning, function_item_id, message_output_offset,
        parse_assistant_turn_from_response, reasoning_output_offset, translate_request,
        unix_timestamp, usage_from_upstream,
    },
    upstream::UpstreamClient,
};

#[derive(Clone)]
struct AppState {
    settings: Arc<Settings>,
    upstream: UpstreamClient,
    tool_executor: ToolExecutor,
    store: ResponseStore,
    inflight: InflightRegistry,
}

#[derive(Clone, Default)]
struct InflightRegistry {
    inner: Arc<RwLock<HashMap<String, Arc<AtomicBool>>>>,
}

#[derive(Clone)]
struct NonStreamExecution {
    turn: AssistantTurn,
    usage: Value,
    request_messages: Vec<Value>,
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
        tool_executor: ToolExecutor::new(settings.clone()),
        settings,
        upstream,
        store: ResponseStore::default(),
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
    info!(
        response_id = %translated.context.response_id,
        model = %translated.context.upstream_model,
        stream = translated.context.stream,
        "request accepted"
    );

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
        Ok(
            if should_auto_execute_tools(&state.tool_executor, &translated) {
                stream_response_with_auto_tools(
                    state.upstream.clone(),
                    state.tool_executor.clone(),
                    state.settings.clone(),
                    state.store.clone(),
                    state.inflight.clone(),
                    cancel_flag,
                    response,
                    translated,
                )
            } else {
                stream_response(
                    state.store.clone(),
                    state.inflight.clone(),
                    cancel_flag,
                    response,
                    translated,
                )
            },
        )
    } else {
        let execution = execute_non_stream_turn(&state, &translated).await?;
        let payload = build_response(
            &translated.context,
            &execution.turn,
            execution.usage,
            unix_timestamp(),
        );
        let translated = translated_with_request_messages(&translated, execution.request_messages);
        store_final_response(&state.store, &translated, payload.clone()).await?;
        info!(
            response_id = %translated.context.response_id,
            "non-stream request completed"
        );
        Ok((StatusCode::OK, Json(payload)).into_response())
    }
}

async fn execute_non_stream_turn(
    state: &AppState,
    translated: &crate::translate::TranslatedRequest,
) -> Result<NonStreamExecution, ProxyError> {
    let mut upstream_payload = translated.upstream_payload.clone();
    let mut last_upstream = state.upstream.chat_json(&upstream_payload).await?;
    let mut turn = parse_assistant_turn_from_response(&last_upstream)?;

    if !state.tool_executor.has_local_tools() {
        return Ok(NonStreamExecution {
            turn,
            usage: usage_from_upstream(last_upstream.get("usage")),
            request_messages: translated.request_messages.clone(),
        });
    }

    for _ in 0..state.settings.max_auto_tool_rounds {
        if turn.tool_calls.is_empty() {
            break;
        }
        if !turn
            .tool_calls
            .iter()
            .all(|tool_call| state.tool_executor.supports_tool(&tool_call.name))
        {
            break;
        }

        let Some(outputs) = state.tool_executor.execute_calls(&turn.tool_calls).await? else {
            break;
        };
        {
            let Some(payload_map) = upstream_payload.as_object_mut() else {
                return Ok(NonStreamExecution {
                    turn,
                    usage: usage_from_upstream(last_upstream.get("usage")),
                    request_messages: translated.request_messages.clone(),
                });
            };
            let Some(messages) = payload_map
                .get_mut("messages")
                .and_then(Value::as_array_mut)
            else {
                return Ok(NonStreamExecution {
                    turn,
                    usage: usage_from_upstream(last_upstream.get("usage")),
                    request_messages: translated.request_messages.clone(),
                });
            };
            append_tool_outputs(messages, &turn.tool_calls, &outputs);
        }

        last_upstream = state.upstream.chat_json(&upstream_payload).await?;
        let next_turn = parse_assistant_turn_from_response(&last_upstream)?;
        if next_turn.tool_calls.is_empty() {
            return Ok(NonStreamExecution {
                turn: next_turn,
                usage: usage_from_upstream(last_upstream.get("usage")),
                request_messages: extract_request_messages(&upstream_payload),
            });
        }
        turn = next_turn;
    }

    Ok(NonStreamExecution {
        turn,
        usage: usage_from_upstream(last_upstream.get("usage")),
        request_messages: extract_request_messages(&upstream_payload),
    })
}

fn should_auto_execute_tools(
    tool_executor: &ToolExecutor,
    translated: &crate::translate::TranslatedRequest,
) -> bool {
    tool_executor.has_local_tools()
        && translated.context.tools.iter().any(|tool| {
            tool.get("type").and_then(Value::as_str) == Some("function")
                && tool
                    .get("function")
                    .and_then(|function| function.get("name"))
                    .or_else(|| tool.get("name"))
                    .and_then(Value::as_str)
                    .map(|name| tool_executor.supports_tool(name))
                    .unwrap_or(false)
        })
}

fn extract_request_messages(payload: &Value) -> Vec<Value> {
    payload
        .get("messages")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

fn translated_with_request_messages(
    translated: &crate::translate::TranslatedRequest,
    request_messages: Vec<Value>,
) -> crate::translate::TranslatedRequest {
    let mut translated = translated.clone();
    translated.request_messages = request_messages;
    translated
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

        let assistant_turn = state.take_turn();

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
        info!(
            response_id = %response_id,
            "stream request completed"
        );
        yield Ok::<Event, Infallible>(json_event("response.completed", &mut sequence_number, json!({
            "type": "response.completed",
            "response": final_response,
        })));
    };

    Sse::new(event_stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

fn stream_response_with_auto_tools(
    upstream: UpstreamClient,
    tool_executor: ToolExecutor,
    settings: Arc<Settings>,
    store: ResponseStore,
    inflight: InflightRegistry,
    cancel_flag: Arc<AtomicBool>,
    response: reqwest::Response,
    translated: crate::translate::TranslatedRequest,
) -> Response {
    let context = translated.context.clone();
    let response_id = context.response_id.clone();
    let event_stream = stream! {
        let mut sequence_number: u64 = 0;
        yield Ok::<Event, Infallible>(json_event("response.created", &mut sequence_number, json!({
            "type": "response.created",
            "response": build_in_progress_response(&context),
        })));

        let mut current_response = response;
        let mut current_payload = translated.upstream_payload.clone();
        let mut final_turn = AssistantTurn::default();
        let mut final_usage = usage_from_upstream(None);
        let mut failed_message: Option<String> = None;
        let mut cancelled = false;

        for _ in 0..settings.max_auto_tool_rounds {
            match collect_stream_turn(&cancel_flag, current_response).await {
                Ok(StreamRoundOutcome::Cancelled) => {
                    cancelled = true;
                    break;
                }
                Ok(StreamRoundOutcome::Failed(message)) => {
                    failed_message = Some(message);
                    break;
                }
                Ok(StreamRoundOutcome::Completed { turn, usage }) => {
                    if turn.tool_calls.is_empty()
                        || !turn
                            .tool_calls
                            .iter()
                            .all(|tool_call| tool_executor.supports_tool(&tool_call.name))
                    {
                        final_turn = turn;
                        final_usage = usage;
                        break;
                    }

                    let outputs = match tool_executor.execute_calls(&turn.tool_calls).await {
                        Ok(Some(outputs)) => outputs,
                        Ok(None) => {
                            final_turn = turn;
                            final_usage = usage;
                            break;
                        }
                        Err(err) => {
                            failed_message = Some(err.to_string());
                            break;
                        }
                    };
                    {
                        let Some(payload_map) = current_payload.as_object_mut() else {
                            final_turn = turn;
                            final_usage = usage;
                            break;
                        };
                        let Some(messages) = payload_map.get_mut("messages").and_then(Value::as_array_mut) else {
                            final_turn = turn;
                            final_usage = usage;
                            break;
                        };
                        append_tool_outputs(messages, &turn.tool_calls, &outputs);
                    }

                    if cancel_flag.load(Ordering::SeqCst) {
                        cancelled = true;
                        break;
                    }

                    match upstream.chat_stream(&current_payload).await {
                        Ok(next_response) => {
                            current_response = next_response;
                        }
                        Err(err) => {
                            failed_message = Some(err.to_string());
                            break;
                        }
                    }
                }
                Err(err) => {
                    failed_message = Some(err.to_string());
                    break;
                }
            }
        }

        if cancelled {
            let cancelled_response = build_cancelled_response(
                &context,
                &final_turn,
                final_usage.clone(),
                unix_timestamp(),
            );
            let translated = translated_with_request_messages(&translated, extract_request_messages(&current_payload));
            let _ = store_final_response(&store, &translated, cancelled_response.clone()).await;
            let _ = inflight.finish(&response_id).await;
            yield Ok::<Event, Infallible>(json_event("response.completed", &mut sequence_number, json!({
                "type": "response.completed",
                "response": cancelled_response,
            })));
            return;
        }

        if let Some(message) = failed_message {
            let failed = build_failed_response(
                &context,
                &final_turn,
                unix_timestamp(),
                &message,
                "upstream_stream_error",
            );
            let translated = translated_with_request_messages(&translated, extract_request_messages(&current_payload));
            let _ = store_final_response(&store, &translated, failed.clone()).await;
            let _ = inflight.finish(&response_id).await;
            yield Ok::<Event, Infallible>(json_event("response.failed", &mut sequence_number, json!({
                "type": "response.failed",
                "response": failed,
            })));
            return;
        }

        for event in finalize_stream_items(&final_turn, &context, &mut sequence_number) {
            yield Ok::<Event, Infallible>(event);
        }

        let final_response = build_response(
            &context,
            &final_turn,
            final_usage,
            unix_timestamp(),
        );
        let translated = translated_with_request_messages(&translated, extract_request_messages(&current_payload));
        let _ = store_final_response(&store, &translated, final_response.clone()).await;
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

enum StreamRoundOutcome {
    Completed { turn: AssistantTurn, usage: Value },
    Failed(String),
    Cancelled,
}

async fn collect_stream_turn(
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

fn stream_round_context() -> RequestContext {
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
        tools: vec![],
        max_output_tokens: None,
    }
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
        warn!("authentication failed: invalid proxy API key");
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

    if let Some(reasoning_delta) = extract_first_reasoning(delta) {
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
        for raw_tool_call in tool_calls {
            let index = raw_tool_call
                .get("index")
                .and_then(Value::as_u64)
                .unwrap_or(state.tool_calls.len() as u64) as usize;
            let output_offset = (if state.has_reasoning() { 1 } else { 0 })
                + if state.has_message() { 1 } else { 0 };
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

    fn take_turn(&mut self) -> AssistantTurn {
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

    fn has_reasoning(&self) -> bool {
        !self.reasoning.is_empty()
    }

    fn has_message(&self) -> bool {
        !self.text.is_empty() || self.tool_calls.is_empty()
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
            let raw: String = self.buffer.drain(..index + 2).collect();
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

    use axum::{Router, body::Body, http::Request, response::sse::Event, routing::post};
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
            local_tools: Default::default(),
            max_auto_tool_rounds: 8,
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
        drop(tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        }));
        format!("http://{addr}")
    }

    async fn spawn_upstream_with_router(app: Router) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind test upstream");
        let addr = listener.local_addr().expect("local addr");
        drop(tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        }));
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

    #[tokio::test]
    async fn healthz_and_auth_and_response_crud_work() {
        let upstream_base_url = spawn_upstream().await;
        let router = build_router(settings(upstream_base_url, false));

        let health = send(
            &router,
            Request::builder()
                .method("GET")
                .uri("/healthz")
                .body(Body::empty())
                .expect("request"),
        )
        .await;
        assert_eq!(health.status(), StatusCode::OK);

        let unauthorized = send(
            &router,
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header("content-type", "application/json")
                .body(Body::from(json!({"input":"hi"}).to_string()))
                .expect("request"),
        )
        .await;
        assert_eq!(unauthorized.status(), StatusCode::UNAUTHORIZED);

        let created = send(
            &router,
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header("authorization", "Bearer proxy-secret")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"input":"hello","store":true}).to_string(),
                ))
                .expect("request"),
        )
        .await;
        let created_body = created
            .into_body()
            .collect()
            .await
            .expect("collect")
            .to_bytes();
        let created_payload: Value = serde_json::from_slice(&created_body).expect("json");
        let response_id = created_payload["id"].as_str().expect("id").to_owned();

        let get_response = send(
            &router,
            Request::builder()
                .method("GET")
                .uri(format!("/v1/responses/{response_id}"))
                .header("authorization", "Bearer proxy-secret")
                .body(Body::empty())
                .expect("request"),
        )
        .await;
        assert_eq!(get_response.status(), StatusCode::OK);

        let items = send(
            &router,
            Request::builder()
                .method("GET")
                .uri(format!("/v1/responses/{response_id}/input_items"))
                .header("authorization", "Bearer proxy-secret")
                .body(Body::empty())
                .expect("request"),
        )
        .await;
        let items_body = items
            .into_body()
            .collect()
            .await
            .expect("collect")
            .to_bytes();
        let items_payload: Value = serde_json::from_slice(&items_body).expect("json");
        assert_eq!(items_payload["object"], "list");
        assert_eq!(items_payload["data"][0]["role"], "user");

        let deleted = send(
            &router,
            Request::builder()
                .method("DELETE")
                .uri(format!("/v1/responses/{response_id}"))
                .header("authorization", "Bearer proxy-secret")
                .body(Body::empty())
                .expect("request"),
        )
        .await;
        assert_eq!(deleted.status(), StatusCode::OK);

        for uri in [
            format!("/v1/responses/{response_id}"),
            format!("/v1/responses/{response_id}/input_items"),
            format!("/v1/responses/{response_id}/cancel"),
        ] {
            let method = if uri.ends_with("/cancel") {
                "POST"
            } else {
                "GET"
            };
            let response = send(
                &router,
                Request::builder()
                    .method(method)
                    .uri(uri)
                    .header("authorization", "Bearer proxy-secret")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        }
    }

    #[tokio::test]
    async fn previous_response_id_and_strict_protocol_behave() {
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let calls_for_handler = calls.clone();
        let upstream = Router::new().route(
            "/v1/chat/completions",
            post(move |Json(payload): Json<Value>| {
                let calls = calls_for_handler.clone();
                async move {
                    let call = calls.fetch_add(1, Ordering::SeqCst);
                    let messages = payload["messages"].as_array().expect("messages");
                    if call == 0 {
                        assert_eq!(messages.len(), 1);
                        assert_eq!(messages[0]["content"], "seed");
                    } else {
                        assert_eq!(messages.len(), 3);
                        assert_eq!(messages[0]["content"], "seed");
                        assert_eq!(messages[1]["role"], "assistant");
                        assert_eq!(messages[2]["content"], "follow up");
                    }
                    Json(json!({
                        "choices": [{
                            "message": {
                                "content": "next",
                                "tool_calls": [{
                                    "id":"call_1",
                                    "function":{"name":"lookup","arguments":"{}"}
                                }]
                            }
                        }]
                    }))
                }
            }),
        );
        let upstream_base_url = spawn_upstream_with_router(upstream).await;
        let router = build_router(settings(upstream_base_url.clone(), false));

        let first = send(
            &router,
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header("authorization", "Bearer proxy-secret")
                .header("content-type", "application/json")
                .body(Body::from(json!({"input":"seed"}).to_string()))
                .expect("request"),
        )
        .await;
        let first_body = first
            .into_body()
            .collect()
            .await
            .expect("collect")
            .to_bytes();
        let first_payload: Value = serde_json::from_slice(&first_body).expect("json");
        let response_id = first_payload["id"].as_str().expect("id");

        let second = send(
            &router,
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header("authorization", "Bearer proxy-secret")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"input":"follow up","previous_response_id":response_id}).to_string(),
                ))
                .expect("request"),
        )
        .await;
        assert_eq!(second.status(), StatusCode::OK);

        let missing_previous = send(
            &router,
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header("authorization", "Bearer proxy-secret")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"input":"x","previous_response_id":"missing"}).to_string(),
                ))
                .expect("request"),
        )
        .await;
        assert_eq!(missing_previous.status(), StatusCode::BAD_REQUEST);

        let mut strict = settings(upstream_base_url, false);
        strict.strict_protocol = true;
        let strict_router = build_router(strict);
        let strict_response = send(
            &strict_router,
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header("authorization", "Bearer proxy-secret")
                .header("content-type", "application/json")
                .body(Body::from(json!({"input":"x","reasoning":{}}).to_string()))
                .expect("request"),
        )
        .await;
        assert_eq!(strict_response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn tool_execution_round_trip_is_preserved_across_requests() {
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let calls_for_handler = calls.clone();
        let upstream = Router::new().route(
            "/v1/chat/completions",
            post(move |Json(payload): Json<Value>| {
                let calls = calls_for_handler.clone();
                async move {
                    let call = calls.fetch_add(1, Ordering::SeqCst);
                    let messages = payload["messages"].as_array().expect("messages");
                    if call == 0 {
                        assert_eq!(messages.len(), 1);
                        assert_eq!(messages[0]["role"], "user");
                        assert_eq!(messages[0]["content"], "weather?");
                        Json(json!({
                            "choices": [{
                                "message": {
                                    "role": "assistant",
                                    "content": null,
                                    "tool_calls": [{
                                        "id":"call_weather_1",
                                        "type":"function",
                                        "function":{"name":"get_weather","arguments":"{\"city\":\"Shanghai\"}"}
                                    }]
                                },
                                "finish_reason": "tool_calls"
                            }]
                        }))
                    } else {
                        assert_eq!(messages.len(), 3);
                        assert_eq!(messages[1]["role"], "assistant");
                        assert_eq!(messages[1]["tool_calls"][0]["id"], "call_weather_1");
                        assert_eq!(messages[2]["role"], "tool");
                        assert_eq!(messages[2]["tool_call_id"], "call_weather_1");
                        assert_eq!(messages[2]["content"], "{\"temp\":26}");
                        Json(json!({
                            "choices": [{
                                "message": {
                                    "role": "assistant",
                                    "content": "26C"
                                },
                                "finish_reason": "stop"
                            }]
                        }))
                    }
                }
            }),
        );
        let upstream_base_url = spawn_upstream_with_router(upstream).await;
        let router = build_router(settings(upstream_base_url, false));

        let first = send(
            &router,
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header("authorization", "Bearer proxy-secret")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model":"gpt-5-codex",
                        "input":"weather?",
                        "tools":[{
                            "type":"function",
                            "name":"get_weather",
                            "parameters":{"type":"object","properties":{"city":{"type":"string"}}}
                        }]
                    })
                    .to_string(),
                ))
                .expect("request"),
        )
        .await;
        assert_eq!(first.status(), StatusCode::OK);
        let first_body = first
            .into_body()
            .collect()
            .await
            .expect("collect")
            .to_bytes();
        let first_payload: Value = serde_json::from_slice(&first_body).expect("json");
        let response_id = first_payload["id"].as_str().expect("id").to_owned();
        assert_eq!(first_payload["output"][0]["type"], "function_call");
        assert_eq!(first_payload["output"][0]["call_id"], "call_weather_1");

        let second = send(
            &router,
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header("authorization", "Bearer proxy-secret")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model":"gpt-5-codex",
                        "previous_response_id": response_id,
                        "input":[{
                            "type":"function_call_output",
                            "call_id":"call_weather_1",
                            "output":{"temp":26}
                        }]
                    })
                    .to_string(),
                ))
                .expect("request"),
        )
        .await;
        assert_eq!(second.status(), StatusCode::OK);
        let second_body = second
            .into_body()
            .collect()
            .await
            .expect("collect")
            .to_bytes();
        let second_payload: Value = serde_json::from_slice(&second_body).expect("json");
        assert_eq!(second_payload["output_text"], "26C");
    }

    #[tokio::test]
    async fn local_tool_execution_can_complete_non_stream_request() {
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let calls_for_handler = calls.clone();
        let upstream = Router::new().route(
            "/v1/chat/completions",
            post(move |Json(payload): Json<Value>| {
                let calls = calls_for_handler.clone();
                async move {
                    let call = calls.fetch_add(1, Ordering::SeqCst);
                    let messages = payload["messages"].as_array().expect("messages");
                    if call == 0 {
                        assert_eq!(messages.len(), 1);
                        Json(json!({
                            "choices": [{
                                "message": {
                                    "role": "assistant",
                                    "content": null,
                                    "tool_calls": [{
                                        "id":"call_echo_1",
                                        "type":"function",
                                        "function":{"name":"echo_json","arguments":"{\"city\":\"Shanghai\"}"}
                                    }]
                                },
                                "finish_reason": "tool_calls"
                            }]
                        }))
                    } else {
                        assert_eq!(messages.len(), 3);
                        assert_eq!(messages[2]["role"], "tool");
                        let tool_output: Value =
                            serde_json::from_str(messages[2]["content"].as_str().expect("content"))
                                .expect("tool output");
                        assert_eq!(tool_output["arguments"]["city"], "Shanghai");
                        Json(json!({
                            "choices": [{
                                "message": {
                                    "role": "assistant",
                                    "content": "tool-finished"
                                },
                                "finish_reason": "stop"
                            }],
                            "usage": {
                                "prompt_tokens": 4,
                                "completion_tokens": 2,
                                "total_tokens": 6
                            }
                        }))
                    }
                }
            }),
        );
        let upstream_base_url = spawn_upstream_with_router(upstream).await;
        let mut settings = settings(upstream_base_url, false);
        settings.local_tools.insert(
            "echo_json".to_owned(),
            crate::config::LocalToolSettings {
                command: "sh".to_owned(),
                args: vec!["-lc".to_owned(), "cat".to_owned()],
                env: Default::default(),
                workdir: None,
                timeout_seconds: 5.0,
                stdin_json: true,
                output_json: true,
            },
        );
        let router = build_router(settings);

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
                        "input":"run local tool",
                        "tools":[{
                            "type":"function",
                            "name":"echo_json",
                            "parameters":{"type":"object","properties":{"city":{"type":"string"}}}
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
        assert_eq!(payload["output_text"], "tool-finished");
        assert_eq!(payload["usage"]["total_tokens"], 6);
    }

    #[tokio::test]
    async fn local_tool_execution_can_complete_stream_request() {
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let calls_for_handler = calls.clone();
        let upstream = Router::new().route(
            "/v1/chat/completions",
            post(move |Json(_payload): Json<Value>| {
                let calls = calls_for_handler.clone();
                async move {
                    let call = calls.fetch_add(1, Ordering::SeqCst);
                    let stream = stream! {
                        if call == 0 {
                            yield Ok::<Event, Infallible>(Event::default().data(r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_echo_stream_1","function":{"name":"echo_json","arguments":"{\"city\":\"Shanghai\"}"}}]}}]}"#));
                            yield Ok::<Event, Infallible>(Event::default().data("[DONE]"));
                        } else {
                            yield Ok::<Event, Infallible>(Event::default().data(r#"{"choices":[{"delta":{"content":"stream-"}}]}"#));
                            yield Ok::<Event, Infallible>(Event::default().data(r#"{"choices":[{"delta":{"content":"done"}}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}"#));
                            yield Ok::<Event, Infallible>(Event::default().data("[DONE]"));
                        }
                    };
                    Sse::new(stream).into_response()
                }
            }),
        );
        let upstream_base_url = spawn_upstream_with_router(upstream).await;
        let mut settings = settings(upstream_base_url, false);
        settings.local_tools.insert(
            "echo_json".to_owned(),
            crate::config::LocalToolSettings {
                command: "sh".to_owned(),
                args: vec!["-lc".to_owned(), "cat".to_owned()],
                env: Default::default(),
                workdir: None,
                timeout_seconds: 5.0,
                stdin_json: true,
                output_json: true,
            },
        );
        let router = build_router(settings);

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
                        "input":"run local tool stream",
                        "stream": true,
                        "tools":[{
                            "type":"function",
                            "name":"echo_json",
                            "parameters":{"type":"object","properties":{"city":{"type":"string"}}}
                        }]
                    })
                    .to_string(),
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
        assert!(body.contains("event: response.created"));
        assert!(body.contains("event: response.output_text.done"));
        assert!(body.contains("stream-done"));
        assert!(body.contains("event: response.completed"));
        assert!(!body.contains("call_echo_stream_1"));
    }

    #[tokio::test]
    async fn streaming_failure_paths_are_reported() {
        let parse_router = Router::new().route(
            "/v1/chat/completions",
            post(|| async { "data: {not-json}\n\n" }),
        );
        let parse_router = build_router(settings(
            spawn_upstream_with_router(parse_router).await,
            false,
        ));
        let parse_response = send(
            &parse_router,
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header("authorization", "Bearer proxy-secret")
                .header("content-type", "application/json")
                .body(Body::from(json!({"input":"x","stream":true}).to_string()))
                .expect("request"),
        )
        .await;
        let parse_body = String::from_utf8(
            parse_response
                .into_body()
                .collect()
                .await
                .expect("collect")
                .to_bytes()
                .to_vec(),
        )
        .expect("utf8");
        assert!(parse_body.contains("event: response.failed"));

        let bad_json_router = Router::new().route(
            "/v1/chat/completions",
            post(|| async { Json(json!({"not_choices":true})) }),
        );
        let bad_json_router = build_router(settings(
            spawn_upstream_with_router(bad_json_router).await,
            false,
        ));
        let bad_json_response = send(
            &bad_json_router,
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header("authorization", "Bearer proxy-secret")
                .header("content-type", "application/json")
                .body(Body::from(json!({"input":"x"}).to_string()))
                .expect("request"),
        )
        .await;
        assert_eq!(bad_json_response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn internal_helpers_cover_remaining_branches() {
        let inflight = InflightRegistry::default();
        assert!(!inflight.cancel("missing").await);
        let token = inflight.start("resp_1".to_owned()).await;
        assert!(!token.load(Ordering::SeqCst));
        assert!(inflight.cancel("resp_1").await);
        assert!(token.load(Ordering::SeqCst));
        inflight.finish("resp_1").await;
        assert!(!inflight.cancel("resp_1").await);

        let mut sequence = 0;
        let _event = json_event("x", &mut sequence, json!({"type":"x"}));
        assert_eq!(sequence, 1);

        let turn = AssistantTurn {
            reasoning: "r".to_owned(),
            text: "t".to_owned(),
            tool_calls: vec![ToolCall {
                call_id: "call_1".to_owned(),
                name: "lookup".to_owned(),
                arguments: "{}".to_owned(),
            }],
        };
        let context = RequestContext {
            response_id: "resp_1".to_owned(),
            reasoning_id: "rs_1".to_owned(),
            message_id: "msg_1".to_owned(),
            created_at: 1,
            client_model: "m".to_owned(),
            upstream_model: "u".to_owned(),
            stream: true,
            store: true,
            parallel_tool_calls: true,
            instructions: None,
            metadata: json!({}),
            tool_choice: json!("auto"),
            tools: vec![],
            max_output_tokens: None,
        };
        let events = finalize_stream_items(&turn, &context, &mut 0);
        assert_eq!(events.len(), 7);

        let mut stream_state = StreamState::new();
        assert!(stream_state.has_message());
        let delta_events = apply_stream_delta(
            &mut stream_state,
            &json!({
                "choices": [{
                    "delta": {
                        "reasoning_summary": [{"text":"a"}],
                        "content": [{"text":"b"}],
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_1",
                            "function": {"name":"lookup","arguments":"{"}
                        },{
                            "index": 0,
                            "function": {"arguments":"}"}
                        }]
                    }
                }]
            }),
            &context,
            &mut 0,
        )
        .expect("delta");
        assert!(!delta_events.is_empty());
        let built_turn = stream_state.take_turn();
        assert_eq!(built_turn.reasoning, "a");
        assert_eq!(built_turn.text, "b");
        assert_eq!(built_turn.tool_calls[0].arguments, "{}");

        let no_choice = apply_stream_delta(&mut StreamState::new(), &json!({}), &context, &mut 0)
            .expect("empty");
        assert!(no_choice.is_empty());
        let no_delta = apply_stream_delta(
            &mut StreamState::new(),
            &json!({"choices":[{}]}),
            &context,
            &mut 0,
        )
        .expect("no delta");
        assert!(no_delta.is_empty());

        let assistant = assistant_turn_from_output(&[
            json!({"type":"reasoning","summary":[{"text":"r"}]}),
            json!({"type":"message","content":"hello"}),
            json!({"type":"function_call","call_id":"call_1","name":"lookup","arguments":"{}"}),
            json!({"type":"ignored"}),
        ])
        .expect("assistant");
        assert_eq!(assistant.reasoning, "r");
        assert_eq!(assistant.text, "hello");
        assert_eq!(assistant.tool_calls.len(), 1);
        assert!(
            assistant_turn_from_output(&[json!({"type":"function_call","name":"lookup"})]).is_err()
        );
        assert!(
            assistant_turn_from_output(&[json!({"type":"function_call","call_id":"call_1"})])
                .is_err()
        );

        let response = json!({
            "id":"resp_1",
            "created_at":1,
            "model":"m",
            "store":false,
            "parallel_tool_calls":false,
            "instructions":"i",
            "metadata":{"a":1},
            "tool_choice":"none",
            "tools":[{"type":"function"}],
            "max_output_tokens":12,
            "output":[
                {"id":"rs_1","type":"reasoning"},
                {"id":"msg_1","type":"message"}
            ]
        });
        let restored = context_from_response(&response).expect("context");
        assert_eq!(restored.reasoning_id, "rs_1");
        assert_eq!(restored.message_id, "msg_1");
        assert!(!restored.store);
        assert!(!restored.parallel_tool_calls);
        assert_eq!(restored.instructions.as_deref(), Some("i"));
        assert_eq!(restored.max_output_tokens, Some(12));

        let restored_defaults = context_from_response(&json!({
            "id":"resp_2",
            "created_at":2,
            "output":[]
        }))
        .expect("context");
        assert_eq!(restored_defaults.reasoning_id, "rs_cancelled");
        assert_eq!(restored_defaults.message_id, "msg_cancelled");
        assert_eq!(restored_defaults.tool_choice, json!("auto"));

        assert!(context_from_response(&json!({"created_at":1})).is_err());
        assert!(context_from_response(&json!({"id":"x"})).is_err());

        let mut parser = SseParser::default();
        assert_eq!(parser.push("data: first\n\n").len(), 1);
        assert!(parser.push("data: second").is_empty());
        assert_eq!(parser.push("\n\n").len(), 1);
        assert_eq!(parse_sse_event("event: x\n\n"), None);
        assert_eq!(
            parse_sse_event("data: a\ndata: b\n\n").as_deref(),
            Some("a\nb")
        );

        let store = ResponseStore::default();
        let translated = crate::translate::TranslatedRequest {
            upstream_payload: json!({}),
            context: RequestContext {
                store: false,
                ..context.clone()
            },
            input_items: vec![json!({"type":"message"})],
            request_messages: vec![json!({"role":"user","content":"hi"})],
        };
        store_final_response(&store, &translated, json!({"id":"resp_1"}))
            .await
            .expect("store");
        assert!(store.get("resp_1").await.expect("get").is_none());

        let translated = crate::translate::TranslatedRequest {
            upstream_payload: json!({}),
            context,
            input_items: vec![json!({"type":"message"})],
            request_messages: vec![json!({"role":"user","content":"hi"})],
        };
        store_final_response(&store, &translated, json!({"id":"resp_1"}))
            .await
            .expect("store");
        assert!(store.get("resp_1").await.expect("get").is_some());

        assert!(
            authorize(
                &settings("http://127.0.0.1".to_owned(), false),
                &HeaderMap::new()
            )
            .is_err()
        );
        let mut wrong = HeaderMap::new();
        wrong.insert("authorization", "Token nope".parse().expect("header"));
        assert!(authorize(&settings("http://127.0.0.1".to_owned(), false), &wrong).is_err());
    }

    #[tokio::test]
    async fn direct_handler_and_helper_edge_cases_are_covered() {
        let upstream_base_url = spawn_upstream().await;
        let mut strict = settings(upstream_base_url.clone(), false);
        strict.strict_protocol = true;
        let strict_router = build_router(strict);

        let supported_in_strict = send(
            &strict_router,
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header("authorization", "Bearer proxy-secret")
                .header("content-type", "application/json")
                .body(Body::from(json!({"input":"ok","store":false}).to_string()))
                .expect("request"),
        )
        .await;
        assert_eq!(supported_in_strict.status(), StatusCode::OK);

        let non_object = send(
            &strict_router,
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header("authorization", "Bearer proxy-secret")
                .header("content-type", "application/json")
                .body(Body::from("[]"))
                .expect("request"),
        )
        .await;
        assert_eq!(non_object.status(), StatusCode::BAD_REQUEST);

        let router = build_router(settings(upstream_base_url, false));
        let missing_delete = send(
            &router,
            Request::builder()
                .method("DELETE")
                .uri("/v1/responses/missing")
                .header("authorization", "Bearer proxy-secret")
                .body(Body::empty())
                .expect("request"),
        )
        .await;
        assert_eq!(missing_delete.status(), StatusCode::BAD_REQUEST);

        let app_state = AppState {
            settings: Arc::new(settings("http://127.0.0.1:1".to_owned(), false)),
            upstream: UpstreamClient::new(Arc::new(settings(
                "http://127.0.0.1:1".to_owned(),
                false,
            )))
            .expect("client"),
            tool_executor: ToolExecutor::new(Arc::new(settings(
                "http://127.0.0.1:1".to_owned(),
                false,
            ))),
            store: ResponseStore::default(),
            inflight: InflightRegistry::default(),
        };
        app_state
            .store
            .put(
                "resp_in_progress".to_owned(),
                StoredResponse {
                    response: json!({
                        "id":"resp_in_progress",
                        "created_at":1,
                        "status":"in_progress",
                        "model":"m",
                        "store":true,
                        "parallel_tool_calls":true,
                        "metadata":{},
                        "tool_choice":"auto",
                        "tools":[],
                        "output":[
                            {"id":"rs_1","type":"reasoning","summary":[{"text":"r"}]},
                            {"id":"msg_1","type":"message","content":"t"},
                            {"type":"function_call","call_id":"call_1","name":"lookup","arguments":"{}"}
                        ]
                    }),
                    input_items: vec![],
                    request_messages: vec![json!({"role":"user","content":"hi"})],
                },
            )
            .await
            .expect("put");
        let mut headers = HeaderMap::new();
        headers.insert(
            "authorization",
            "Bearer proxy-secret".parse().expect("header"),
        );
        let cancelled = cancel_response(
            State(app_state.clone()),
            Path("resp_in_progress".to_owned()),
            headers.clone(),
        )
        .await
        .expect("cancel");
        assert_eq!(cancelled.status(), StatusCode::OK);
        let stored = app_state
            .store
            .get("resp_in_progress")
            .await
            .expect("get")
            .expect("stored");
        assert_eq!(stored.response["status"], "cancelled");

        app_state
            .store
            .put(
                "resp_in_progress_empty".to_owned(),
                StoredResponse {
                    response: json!({
                        "id":"resp_in_progress_empty",
                        "created_at":1,
                        "status":"in_progress",
                        "output":[]
                    }),
                    input_items: vec![],
                    request_messages: vec![],
                },
            )
            .await
            .expect("put");
        let empty_cancelled = cancel_response(
            State(app_state.clone()),
            Path("resp_in_progress_empty".to_owned()),
            headers.clone(),
        )
        .await
        .expect("cancel");
        assert_eq!(empty_cancelled.status(), StatusCode::OK);

        let no_output_store = ResponseStore::default();
        no_output_store
            .put(
                "resp_prev".to_owned(),
                StoredResponse {
                    response: json!({"id":"resp_prev"}),
                    input_items: vec![],
                    request_messages: vec![json!({"role":"user","content":"hi"})],
                },
            )
            .await
            .expect("put");
        let previous = load_previous_messages(
            &no_output_store,
            json!({"previous_response_id":"resp_prev"})
                .as_object()
                .expect("object"),
        )
        .await
        .expect("previous");
        assert_eq!(previous.len(), 1);

        let mut wrong_headers = HeaderMap::new();
        wrong_headers.insert(
            "authorization",
            "Bearer definitely-wrong".parse().expect("header"),
        );
        assert!(
            authorize(
                &settings("http://127.0.0.1".to_owned(), false),
                &wrong_headers
            )
            .is_err()
        );

        let mut state = StreamState::new();
        let context = RequestContext {
            response_id: "resp_1".to_owned(),
            reasoning_id: "rs_1".to_owned(),
            message_id: "msg_1".to_owned(),
            created_at: 1,
            client_model: "m".to_owned(),
            upstream_model: "u".to_owned(),
            stream: true,
            store: true,
            parallel_tool_calls: true,
            instructions: None,
            metadata: json!({}),
            tool_choice: json!("auto"),
            tools: vec![],
            max_output_tokens: None,
        };
        let events = apply_stream_delta(
            &mut state,
            &json!({
                "choices": [{
                    "delta": {
                        "reasoning": "",
                        "content": 1,
                        "tool_calls": [{"id":"call_2"}]
                    }
                }]
            }),
            &context,
            &mut 0,
        )
        .expect("delta");
        assert!(events.is_empty());

        let events = apply_stream_delta(
            &mut state,
            &json!({
                "choices": [{
                    "delta": {
                        "reasoning": "x",
                        "tool_calls": [{
                            "id":"call_3",
                            "function":{"name":"lookup","arguments":{"x":1}}
                        },{
                            "id":"call_4",
                            "function":{"name":"noop"}
                        }]
                    }
                }]
            }),
            &context,
            &mut 0,
        )
        .expect("delta");
        assert!(!events.is_empty());

        let parsed = assistant_turn_from_output(&[json!({"type":"message"})]).expect("turn");
        assert_eq!(parsed.text, "");

        let restored = context_from_response(&json!({
            "id":"resp_3",
            "created_at":3,
            "output":[{"type":"message","id":"msg_3"}]
        }))
        .expect("context");
        assert_eq!(restored.reasoning_id, "rs_cancelled");
    }
}
