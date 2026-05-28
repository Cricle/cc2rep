use axum::{
    Json,
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use serde_json::{Value, json};
use tracing::{info, warn};

use crate::{
    app::AppState,
    error::ProxyError,
    protocol::analyze_protocol,
    translate::translate_request,
};

pub(crate) async fn healthz() -> Response {
    Json(json!({ "ok": true })).into_response()
}

pub(crate) async fn create_response(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: String,
) -> Result<Response, ProxyError> {
    authorize(&state.settings, &headers)?;

    let value: Value = serde_json::from_str(&body).map_err(|err| {
        ProxyError::bad_request(format!("invalid JSON body: {err}"))
    })?;

    let Value::Object(map) = &value else {
        return Err(ProxyError::bad_request("request body must be a JSON object"));
    };

    let report = analyze_protocol(map);
    if state.settings.strict_protocol
        && let Some(message) = report.strict_error() {
            return Err(ProxyError::bad_request(message));
        }
    let previous_messages = crate::app::load_previous_messages(&state.store, map).await?;
    let translated = translate_request(
        map,
        &state.settings,
        &report,
        &previous_messages,
        &state.capabilities,
    )?;

    if translated.context.store {
        state
            .store
            .put(
                translated.context.response_id.clone(),
                crate::store::StoredResponse {
                    response: crate::translate::build_in_progress_response(&translated.context),
                    input_items: translated.input_items.clone(),
                    request_messages: translated.request_messages.clone(),
                    inserted_at: std::time::Instant::now(),
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
            if crate::app::should_auto_execute_tools(&state.tool_executor, &translated) {
                crate::app::stream_response_with_auto_tools(
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
                crate::app::stream_response(
                    state.store.clone(),
                    state.inflight.clone(),
                    cancel_flag,
                    response,
                    translated,
                )
            },
        )
    } else {
        let execution = crate::app::execute_non_stream_turn(&state, &translated).await?;
        let payload = crate::translate::build_response(
            &translated.context,
            &execution.turn,
            execution.usage,
            crate::translate::unix_timestamp(),
        );
        let translated =
            crate::app::translated_with_request_messages(&translated, execution.request_messages);
        crate::app::store_final_response(&state.store, &translated, payload.clone()).await?;
        info!(
            response_id = %translated.context.response_id,
            "non-stream request completed"
        );
        state.inflight.finish(&translated.context.response_id).await;
        Ok(Json(payload).into_response())
    }
}

pub(crate) async fn get_response(
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

pub(crate) async fn list_input_items(
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

pub(crate) async fn delete_response(
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

pub(crate) async fn cancel_response(
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
    let context = crate::app::context_from_response(&stored.response)?;
    let turn = crate::app::assistant_turn_from_output(
        stored
            .response
            .get("output")
            .and_then(Value::as_array)
            .unwrap_or(&vec![]),
    )?;
    let usage = stored
        .response
        .get("usage")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let cancelled_response = crate::translate::build_cancelled_response(
        &context,
        &turn,
        usage,
        crate::translate::unix_timestamp(),
    );
    let translated = crate::app::translated_with_request_messages(
        &crate::translate::TranslatedRequest {
            context: context.clone(),
            upstream_payload: json!({}),
            input_items: stored.input_items.clone(),
            request_messages: stored.request_messages.clone(),
        },
        stored.request_messages.clone(),
    );
    if let Err(err) = crate::app::store_final_response(&state.store, &translated, cancelled_response.clone())
        .await
    {
        warn!(response_id = %response_id, "failed to store cancelled response: {err}");
    }
    info!(response_id = %response_id, "response cancelled");
    Ok((StatusCode::OK, Json(cancelled_response)).into_response())
}

pub(crate) fn authorize(
    settings: &crate::config::Settings,
    headers: &HeaderMap,
) -> Result<(), ProxyError> {
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
