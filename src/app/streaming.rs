use std::{
    convert::Infallible,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use async_stream::stream;
use axum::response::{
    IntoResponse, Response,
    sse::{Event, KeepAlive, Sse},
};
use futures_util::StreamExt;
use serde_json::{Value, json};
use tracing::{info, warn};

use crate::{
    config::Settings,
    metrics::RequestMetrics,
    store::ResponseStore,
    stream::{
        SseParser, StreamRoundOutcome, StreamState, apply_stream_delta, collect_stream_turn,
        finalize_stream_items, json_event,
    },
    tools::{ToolExecutor, append_tool_outputs},
    translate::{
        AssistantTurn, build_cancelled_response, build_failed_response, build_in_progress_response,
        build_response, merge_usage, unix_timestamp, usage_from_upstream,
    },
    upstream::UpstreamClient,
};

use super::{
    InflightRegistry,
    helpers::{extract_request_messages, translated_with_request_messages},
    store_ext::store_final_response,
};
pub(crate) fn stream_response(
    store: ResponseStore,
    inflight: InflightRegistry,
    metrics: Arc<RequestMetrics>,
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

        for (hosted_index, hosted_item) in context.hosted_output_items.iter().enumerate() {
            yield Ok::<Event, Infallible>(json_event("response.output_item.added", &mut sequence_number, json!({
                "type": "response.output_item.added",
                "output_index": hosted_index,
                "item": hosted_item,
            })));
            yield Ok::<Event, Infallible>(json_event("response.output_item.done", &mut sequence_number, json!({
                "type": "response.output_item.done",
                "output_index": hosted_index,
                "item": hosted_item,
            })));
        }

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
            metrics.record_cancellation();
            let cancelled_response = build_cancelled_response(
                &context,
                &assistant_turn,
                state.usage.clone(),
                unix_timestamp(),
            );
            if let Err(err) = store_final_response(&store, &translated_for_store, cancelled_response.clone()).await {
                warn!(response_id = %response_id, "failed to store cancelled response: {err}");
            }
            let _ = inflight.finish(&response_id).await;
            yield Ok::<Event, Infallible>(json_event("response.completed", &mut sequence_number, json!({
                "type": "response.completed",
                "response": cancelled_response,
            })));
            return;
        }

        if let Some(message) = failed_message {
            metrics.record_failure();
            warn!("stream failed: {message}");
            let failed = build_failed_response(
                &context,
                &assistant_turn,
                unix_timestamp(),
                &message,
                "upstream_stream_error",
            );
            if let Err(err) = store_final_response(&store, &translated_for_store, failed.clone()).await {
                warn!(response_id = %response_id, "failed to store failed response: {err}");
            }
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
        metrics.record_completion(final_response.get("usage").unwrap_or(&json!({})));
        if let Err(err) = store_final_response(&store, &translated_for_store, final_response.clone()).await {
            warn!(response_id = %response_id, "failed to store final response: {err}");
        }
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

#[allow(clippy::too_many_arguments)]
pub(crate) fn stream_response_with_auto_tools(
    upstream: UpstreamClient,
    tool_executor: ToolExecutor,
    settings: Arc<Settings>,
    store: ResponseStore,
    inflight: InflightRegistry,
    metrics: Arc<RequestMetrics>,
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

        for (hosted_index, hosted_item) in context.hosted_output_items.iter().enumerate() {
            yield Ok::<Event, Infallible>(json_event("response.output_item.added", &mut sequence_number, json!({
                "type": "response.output_item.added",
                "output_index": hosted_index,
                "item": hosted_item,
            })));
            yield Ok::<Event, Infallible>(json_event("response.output_item.done", &mut sequence_number, json!({
                "type": "response.output_item.done",
                "output_index": hosted_index,
                "item": hosted_item,
            })));
        }

        let mut current_response = response;
        let mut current_payload = translated.upstream_payload.clone();
        let mut final_turn = AssistantTurn::default();
        let mut total_usage = usage_from_upstream(None);
        let mut failed_message: Option<String> = None;
        let mut cancelled = false;
        let max_tool_calls = context.max_tool_calls;
        let mut total_tool_calls: u32 = 0;
        let parallel = context.parallel_tool_calls;

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
                    merge_usage(&mut total_usage, &usage);
                    let supported: Vec<_> = turn
                        .tool_calls
                        .iter()
                        .filter(|tc| tool_executor.supports_tool(&tc.name))
                        .cloned()
                        .collect();
                    if supported.is_empty() {
                        final_turn = turn;
                        break;
                    }
                    if let Some(limit) = max_tool_calls {
                        total_tool_calls += supported.len() as u32;
                        if total_tool_calls > limit {
                            final_turn = turn;
                            break;
                        }
                    }

                    let outputs = match tool_executor.execute_calls(&supported, parallel).await {
                        Ok(Some(outputs)) => outputs,
                        Ok(None) => {
                            final_turn = turn;
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
                            break;
                        };
                        let Some(messages) = payload_map.get_mut("messages").and_then(Value::as_array_mut) else {
                            final_turn = turn;
                            break;
                        };
                        append_tool_outputs(
                            messages,
                            &supported,
                            &outputs,
                            Some(&turn.reasoning),
                        );
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
            metrics.record_cancellation();
            let cancelled_response = build_cancelled_response(
                &context,
                &final_turn,
                total_usage.clone(),
                unix_timestamp(),
            );
            let translated = translated_with_request_messages(&translated, extract_request_messages(&current_payload));
            if let Err(err) = store_final_response(&store, &translated, cancelled_response.clone()).await {
                warn!(response_id = %response_id, "failed to store cancelled response: {err}");
            }
            let _ = inflight.finish(&response_id).await;
            yield Ok::<Event, Infallible>(json_event("response.completed", &mut sequence_number, json!({
                "type": "response.completed",
                "response": cancelled_response,
            })));
            return;
        }

        if let Some(message) = failed_message {
            metrics.record_failure();
            let failed = build_failed_response(
                &context,
                &final_turn,
                unix_timestamp(),
                &message,
                "upstream_stream_error",
            );
            let translated = translated_with_request_messages(&translated, extract_request_messages(&current_payload));
            if let Err(err) = store_final_response(&store, &translated, failed.clone()).await {
                warn!(response_id = %response_id, "failed to store failed response: {err}");
            }
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
            total_usage,
            unix_timestamp(),
        );
        metrics.record_completion(final_response.get("usage").unwrap_or(&json!({})));
        let translated = translated_with_request_messages(&translated, extract_request_messages(&current_payload));
        if let Err(err) = store_final_response(&store, &translated, final_response.clone()).await {
            warn!(response_id = %response_id, "failed to store final response: {err}");
        }
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
