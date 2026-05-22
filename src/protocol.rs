use serde_json::{Map, Value, json};

const SUPPORTED_FIELDS: &[&str] = &[
    "frequency_penalty",
    "input",
    "instructions",
    "max_output_tokens",
    "metadata",
    "model",
    "presence_penalty",
    "response_format",
    "stop",
    "stream",
    "temperature",
    "text",
    "tool_choice",
    "tools",
    "top_p",
    "user",
];

const EMULATED_FIELDS: &[&str] = &["parallel_tool_calls", "store"];
const IGNORED_FIELDS: &[&str] = &[
    "background",
    "include",
    "max_tool_calls",
    "previous_response_id",
    "prompt",
    "prompt_cache_key",
    "reasoning",
    "service_tier",
    "top_logprobs",
    "truncation",
];

#[derive(Debug, Clone)]
pub struct ProtocolReport {
    pub supported_fields: Vec<String>,
    pub emulated_fields: Vec<String>,
    pub ignored_fields: Vec<String>,
    pub unsupported_fields: Vec<String>,
}

impl ProtocolReport {
    pub fn has_compatibility_notes(&self) -> bool {
        !(self.emulated_fields.is_empty()
            && self.ignored_fields.is_empty()
            && self.unsupported_fields.is_empty())
    }

    pub fn strict_error(&self) -> Option<String> {
        let mut blocked = self.ignored_fields.clone();
        blocked.extend(self.unsupported_fields.clone());
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
        if !self.ignored_fields.is_empty() {
            compatibility.insert(
                "ignored_fields".to_owned(),
                Value::Array(
                    self.ignored_fields
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
    let mut ignored = Vec::new();
    let mut unsupported = Vec::new();

    for key in payload.keys() {
        if SUPPORTED_FIELDS.contains(&key.as_str()) {
            supported.push(key.clone());
        } else if EMULATED_FIELDS.contains(&key.as_str()) {
            emulated.push(key.clone());
        } else if IGNORED_FIELDS.contains(&key.as_str()) {
            ignored.push(key.clone());
        } else {
            unsupported.push(key.clone());
        }
    }

    supported.sort();
    emulated.sort();
    ignored.sort();
    unsupported.sort();

    ProtocolReport {
        supported_fields: supported,
        emulated_fields: emulated,
        ignored_fields: ignored,
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
        assert_eq!(report.emulated_fields, vec!["parallel_tool_calls", "store"]);
        assert_eq!(report.ignored_fields, vec!["background", "reasoning"]);
        assert_eq!(report.unsupported_fields, vec!["zzz"]);
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
        assert_eq!(fragment["compatibility"]["ignored_fields"][0], "prompt");
        assert_eq!(
            fragment["compatibility"]["unsupported_fields"][0],
            "unknown"
        );
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
