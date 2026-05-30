use serde_json::Value;

use crate::{
    error::ProxyError,
    store::{ResponseStore, StoredResponse},
    translate::build_history_message,
};

use super::restore::assistant_turn_from_output;
pub(crate) async fn load_previous_messages(
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

pub(crate) async fn store_final_response(
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
                inserted_at: std::time::Instant::now(),
            },
        )
        .await
}

