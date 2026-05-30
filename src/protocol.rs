use serde_json::{Map, Value, json};

const SUPPORTED_FIELDS: &[&str] = &[
    "frequency_penalty",
    "input",
    "include",
    "instructions",
    "max_output_tokens",
    "metadata",
    "model",
    "presence_penalty",
    "previous_response_id",
    "response_format",
    "stop",
    "stream",
    "temperature",
    "text",
    "tool_choice",
    "tools",
    "top_logprobs",
    "top_p",
    "truncation",
    "user",
];

const EMULATED_FIELDS: &[&str] = &["max_tool_calls", "parallel_tool_calls", "reasoning", "store"];
const UNSUPPORTED_FIELDS: &[&str] = &[
    "background",
    "prompt",
    "prompt_cache_key",
    "service_tier",
];

#[derive(Debug, Clone)]
pub struct ProtocolReport {
    pub supported_fields: Vec<String>,
    pub emulated_fields: Vec<String>,
    pub unsupported_fields: Vec<String>,
}

impl ProtocolReport {
    pub fn has_compatibility_notes(&self) -> bool {
        !(self.emulated_fields.is_empty() && self.unsupported_fields.is_empty())
    }

    pub fn strict_error(&self) -> Option<String> {
        let blocked = self.unsupported_fields.clone();
        if blocked.is_empty() {
            None
        } else {
            Some(format!(
                "Unsupported Responses API fields in strict mode: {}",
                blocked.join(", ")
            ))
        }
    }

    pub fn metadata_fragment(&self) -> Value {
        let mut compatibility = Map::new();
        compatibility.insert(
            "mode".to_owned(),
            Value::String("chat_completions_bridge".to_owned()),
        );
        compatibility.insert(
            "supported_fields".to_owned(),
            Value::Array(
                self.supported_fields
                    .iter()
                    .cloned()
                    .map(Value::String)
                    .collect(),
            ),
        );
        if !self.emulated_fields.is_empty() {
            compatibility.insert(
                "emulated_fields".to_owned(),
                Value::Array(
                    self.emulated_fields
                        .iter()
                        .cloned()
                        .map(Value::String)
                        .collect(),
                ),
            );
        }
        if !self.unsupported_fields.is_empty() {
            compatibility.insert(
                "unsupported_fields".to_owned(),
                Value::Array(
                    self.unsupported_fields
                        .iter()
                        .cloned()
                        .map(Value::String)
                        .collect(),
                ),
            );
        }
        json!({ "compatibility": compatibility })
    }
}

pub fn analyze_protocol(payload: &Map<String, Value>) -> ProtocolReport {
    let mut supported = Vec::new();
    let mut emulated = Vec::new();
    let mut unsupported = Vec::new();

    for key in payload.keys() {
        if SUPPORTED_FIELDS.contains(&key.as_str()) {
            supported.push(key.clone());
        } else if EMULATED_FIELDS.contains(&key.as_str()) {
            emulated.push(key.clone());
        } else if UNSUPPORTED_FIELDS.contains(&key.as_str()) {
            unsupported.push(key.clone());
        } else {
            unsupported.push(key.clone());
        }
    }

    supported.sort();
    emulated.sort();
    unsupported.sort();

    ProtocolReport {
        supported_fields: supported,
        emulated_fields: emulated,
        unsupported_fields: unsupported,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analyze_protocol_categorizes_and_sorts_fields() {
        let payload = serde_json::from_value::<Map<String, Value>>(json!({
            "zzz": 1,
            "input": "hi",
            "store": true,
            "parallel_tool_calls": true,
            "reasoning": {},
            "background": true,
            "model": "m"
        }))
        .expect("map");

        let report = analyze_protocol(&payload);
        assert_eq!(report.supported_fields, vec!["input", "model"]);
        assert_eq!(report.emulated_fields, vec!["parallel_tool_calls", "reasoning", "store"]);
        assert_eq!(report.unsupported_fields, vec!["background", "zzz"]);
        assert!(report.has_compatibility_notes());
    }

    #[test]
    fn strict_error_and_metadata_fragment_reflect_report_contents() {
        let payload = serde_json::from_value::<Map<String, Value>>(json!({
            "input": "hi",
            "prompt": "ignored",
            "unknown": true
        }))
        .expect("map");
        let report = analyze_protocol(&payload);

        let error = report.strict_error().expect("strict error");
        assert!(error.contains("prompt"));
        assert!(error.contains("unknown"));

        let fragment = report.metadata_fragment();
        assert_eq!(fragment["compatibility"]["mode"], "chat_completions_bridge");
        assert_eq!(fragment["compatibility"]["supported_fields"][0], "input");
        let unsupported = fragment["compatibility"]["unsupported_fields"].as_array().unwrap();
        assert!(unsupported.iter().any(|v| v.as_str() == Some("prompt")));
        assert!(unsupported.iter().any(|v| v.as_str() == Some("unknown")));
    }

    #[test]
    fn strict_error_is_none_without_blocked_fields() {
        let payload = serde_json::from_value::<Map<String, Value>>(json!({
            "input": "hi",
            "model": "m",
            "store": true
        }))
        .expect("map");
        let report = analyze_protocol(&payload);
        assert_eq!(report.strict_error(), None);
    }
}
