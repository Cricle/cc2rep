use serde_json::Value;
pub(crate) fn extract_request_messages(payload: &Value) -> Vec<Value> {
    payload
        .get("messages")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

pub(crate) fn translated_with_request_messages(
    translated: &crate::translate::TranslatedRequest,
    request_messages: Vec<Value>,
) -> crate::translate::TranslatedRequest {
    let mut translated = translated.clone();
    translated.request_messages = request_messages;
    translated
}
