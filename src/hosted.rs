use std::path::Path;

use serde_json::{Value, json};
use tracing::{info, warn};

use crate::{config::Settings, error::ProxyError};

const HOSTED_TOOL_TYPES: &[&str] = &[
    "web_search",
    "web_search_preview",
    "file_search",
    "computer_use",
    "computer_use_preview",
];

#[derive(Debug, Clone, Default)]
pub(crate) struct HostedToolContext {
    /// Output items to prepend to the response output (web_search_call, file_search_call, etc.)
    pub output_items: Vec<Value>,
    /// Messages to inject into upstream conversation (tool results)
    pub messages: Vec<Value>,
}

/// Execute hosted tools found in the request payload.
/// Returns a HostedToolContext with output items and messages to inject.
pub(crate) async fn execute_hosted_tools(
    payload: &serde_json::Map<String, Value>,
    settings: &Settings,
    http_client: &reqwest::Client,
) -> Result<HostedToolContext, ProxyError> {
    let tool_types = collect_hosted_tool_types(payload.get("tools"));
    if tool_types.is_empty() {
        return Ok(HostedToolContext::default());
    }

    let query = extract_query_text(payload.get("input"));
    if query.is_empty() {
        return Ok(HostedToolContext::default());
    }

    let mut ctx = HostedToolContext::default();

    if tool_types
        .iter()
        .any(|t| t == "web_search" || t == "web_search_preview")
        && let Some(ref url) = settings.web_search_url
    {
        match execute_web_search(&query, url, settings.web_search_max_results, http_client).await {
            Ok(section) => {
                if let Some(msg) = section.message {
                    ctx.messages.push(msg);
                }
                if let Some(item) = section.output_item {
                    ctx.output_items.push(item);
                }
            }
            Err(err) => {
                warn!("web_search failed: {err}");
                ctx.messages.push(json!({
                    "role": "system",
                    "content": format!("Web search for `{query}` failed: {err}. Continue without search results."),
                }));
            }
        }
    }

    if tool_types.iter().any(|t| t == "file_search") && !settings.file_search_paths.is_empty() {
        match execute_file_search(
            &query,
            &settings.file_search_paths,
            settings.file_search_max_results,
        ) {
            Ok(section) => {
                if let Some(msg) = section.message {
                    ctx.messages.push(msg);
                }
                if let Some(item) = section.output_item {
                    ctx.output_items.push(item);
                }
            }
            Err(err) => {
                warn!("file_search failed: {err}");
            }
        }
    }

    // computer_use: passthrough to local_tools if configured, otherwise skip
    // (handled by existing ToolExecutor auto-execution)

    Ok(ctx)
}

struct HostedToolSection {
    message: Option<Value>,
    output_item: Option<Value>,
}

/// Collect hosted tool types from the tools array.
fn collect_hosted_tool_types(tools: Option<&Value>) -> Vec<String> {
    let Some(Value::Array(tools)) = tools else {
        return Vec::new();
    };
    let mut found = Vec::new();
    for tool in tools {
        let Some(map) = tool.as_object() else {
            continue;
        };
        let tool_type = map
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("function");
        if HOSTED_TOOL_TYPES.contains(&tool_type) {
            found.push(tool_type.to_owned());
        }
        if tool_type == "namespace"
            && let Some(nested) = map.get("tools").or_else(|| map.get("items"))
        {
            found.extend(collect_hosted_tool_types(Some(nested)));
        }
    }
    found
}

/// Extract query text from the input field (string or message array).
fn extract_query_text(input: Option<&Value>) -> String {
    let Some(input) = input else {
        return String::new();
    };
    match input {
        Value::String(s) => s.clone(),
        Value::Array(items) => {
            let mut texts = Vec::new();
            for item in items {
                if let Some(content) = item.get("content") {
                    match content {
                        Value::String(s) => texts.push(s.as_str()),
                        Value::Array(parts) => {
                            for part in parts {
                                if let Some(text) = part.get("text").and_then(Value::as_str) {
                                    texts.push(text);
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            texts.join(" ")
        }
        _ => String::new(),
    }
}

// ============================================================
// Web Search (SearxNG)
// ============================================================

async fn execute_web_search(
    query: &str,
    searxng_url: &str,
    max_results: usize,
    client: &reqwest::Client,
) -> Result<HostedToolSection, ProxyError> {
    info!(query = %query, "executing web search");
    let response = client
        .get(searxng_url)
        .query(&[("q", query), ("format", "json")])
        .send()
        .await
        .map_err(|err| ProxyError::Transport(format!("web search request failed: {err}")))?;

    if !response.status().is_success() {
        return Err(ProxyError::Transport(format!(
            "web search returned HTTP {}",
            response.status()
        )));
    }

    let payload: Value = response
        .json()
        .await
        .map_err(|err| ProxyError::Transport(format!("web search response parse error: {err}")))?;

    let results = normalize_search_results(payload.get("results"), max_results);

    if results.is_empty() {
        return Ok(HostedToolSection {
            message: Some(json!({
                "role": "system",
                "content": format!("Web search results for `{query}`:\nNo results found."),
            })),
            output_item: Some(json!({
                "id": format!("ws_{}", uuid::Uuid::new_v4().as_simple()),
                "type": "web_search_call",
                "status": "completed",
                "action": {"type": "search", "query": query},
            })),
        });
    }

    let results_text = results
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let title = r.get("title").and_then(Value::as_str).unwrap_or("");
            let url = r.get("url").and_then(Value::as_str).unwrap_or("");
            let snippet = r.get("snippet").and_then(Value::as_str).unwrap_or("");
            format!("{}. {}\n   {}\n   {}", i + 1, title, url, snippet)
        })
        .collect::<Vec<_>>()
        .join("\n\n");

    let annotations: Vec<Value> = results
        .iter()
        .filter_map(|r| {
            let url = r.get("url").and_then(Value::as_str)?;
            if url.is_empty() {
                return None;
            }
            Some(json!({
                "type": "url_citation",
                "url": url,
                "title": r.get("title").and_then(Value::as_str).unwrap_or(""),
            }))
        })
        .collect();

    let mut output_item = json!({
        "id": format!("ws_{}", uuid::Uuid::new_v4().as_simple()),
        "type": "web_search_call",
        "status": "completed",
        "action": {"type": "search", "query": query},
    });
    if !annotations.is_empty() {
        output_item["annotations"] = json!(annotations);
    }

    Ok(HostedToolSection {
        message: Some(json!({
            "role": "system",
            "content": format!("Web search results for `{query}`:\n{results_text}"),
        })),
        output_item: Some(output_item),
    })
}

fn normalize_search_results(results: Option<&Value>, max_results: usize) -> Vec<Value> {
    let Some(Value::Array(results)) = results else {
        return Vec::new();
    };
    results
        .iter()
        .take(max_results)
        .filter_map(|r| {
            let map = r.as_object()?;
            let title = map.get("title").and_then(Value::as_str).unwrap_or("");
            let url = map.get("url").and_then(Value::as_str).unwrap_or("");
            let snippet = map
                .get("content")
                .or_else(|| map.get("snippet"))
                .and_then(Value::as_str)
                .unwrap_or("");
            if title.is_empty() && url.is_empty() {
                return None;
            }
            Some(json!({
                "title": title,
                "url": url,
                "snippet": snippet,
            }))
        })
        .collect()
}

// ============================================================
// File Search (local)
// ============================================================

fn execute_file_search(
    query: &str,
    search_paths: &[String],
    max_results: usize,
) -> Result<HostedToolSection, ProxyError> {
    info!(query = %query, paths = ?search_paths, "executing file search");
    let keywords: Vec<String> = query
        .split_whitespace()
        .filter(|w| w.len() > 1)
        .map(|w| w.to_lowercase())
        .collect();

    if keywords.is_empty() {
        return Ok(HostedToolSection {
            message: None,
            output_item: None,
        });
    }

    let mut matches: Vec<Value> = Vec::new();
    for search_path in search_paths {
        let path = Path::new(search_path);
        if !path.exists() {
            continue;
        }
        search_directory(path, &keywords, &mut matches, max_results);
        if matches.len() >= max_results {
            break;
        }
    }

    matches.truncate(max_results);

    if matches.is_empty() {
        return Ok(HostedToolSection {
            message: Some(json!({
                "role": "system",
                "content": format!("File search for `{query}`:\nNo matching files found."),
            })),
            output_item: Some(json!({
                "id": format!("fs_{}", uuid::Uuid::new_v4().as_simple()),
                "type": "file_search_call",
                "status": "completed",
                "query": query,
            })),
        });
    }

    let results_text = matches
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let path = m.get("path").and_then(Value::as_str).unwrap_or("");
            let snippet = m.get("snippet").and_then(Value::as_str).unwrap_or("");
            format!("{}. {}\n   {}", i + 1, path, snippet)
        })
        .collect::<Vec<_>>()
        .join("\n\n");

    Ok(HostedToolSection {
        message: Some(json!({
            "role": "system",
            "content": format!("File search results for `{query}`:\n{results_text}"),
        })),
        output_item: Some(json!({
            "id": format!("fs_{}", uuid::Uuid::new_v4().as_simple()),
            "type": "file_search_call",
            "status": "completed",
            "query": query,
            "results": matches,
        })),
    })
}

fn search_directory(dir: &Path, keywords: &[String], matches: &mut Vec<Value>, limit: usize) {
    if matches.len() >= limit {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        if matches.len() >= limit {
            return;
        }
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name.starts_with('.') || name == "node_modules" || name == "target" {
                continue;
            }
            search_directory(&path, keywords, matches, limit);
        } else if path.is_file() {
            let file_name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_lowercase();
            let path_str = path.to_string_lossy().to_lowercase();

            // Check filename match
            let name_match = keywords
                .iter()
                .any(|kw| file_name.contains(kw) || path_str.contains(kw));

            // Check content match (only for text files, max 1MB)
            let content_match = if !name_match {
                match_content(&path, keywords)
            } else {
                None
            };

            if name_match || content_match.is_some() {
                let snippet = content_match.unwrap_or_default();
                matches.push(json!({
                    "path": path.to_string_lossy(),
                    "snippet": snippet,
                }));
            }
        }
    }
}

fn match_content(path: &Path, keywords: &[String]) -> Option<String> {
    // Skip large files and binary files
    let metadata = std::fs::metadata(path).ok()?;
    if metadata.len() > 1_000_000 {
        return None;
    }
    let content = std::fs::read_to_string(path).ok()?;
    let lower = content.to_lowercase();
    for keyword in keywords {
        if let Some(pos) = lower.find(keyword.as_str()) {
            let start = pos.saturating_sub(80);
            let end = (pos + keyword.len() + 80).min(content.len());
            // Find valid UTF-8 boundaries
            let start = content.floor_char_boundary(start);
            let end = content.ceil_char_boundary(end);
            let snippet = content[start..end].replace('\n', " ");
            return Some(format!("...{snippet}..."));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collect_hosted_tool_types_finds_hosted_tools() {
        let tools = json!([
            {"type": "function", "function": {"name": "test"}},
            {"type": "web_search"},
            {"type": "file_search"},
            {"type": "namespace", "tools": [
                {"type": "computer_use"},
            ]},
        ]);
        let types = collect_hosted_tool_types(Some(&tools));
        assert!(types.contains(&"web_search".to_owned()));
        assert!(types.contains(&"file_search".to_owned()));
        assert!(types.contains(&"computer_use".to_owned()));
        assert!(!types.iter().any(|t| t == "function"));
    }

    #[test]
    fn extract_query_text_from_string() {
        assert_eq!(
            extract_query_text(Some(&json!("hello world"))),
            "hello world"
        );
    }

    #[test]
    fn extract_query_text_from_messages() {
        let input = json!([
            {"role": "user", "content": "What is Rust?"},
            {"role": "assistant", "content": "Rust is a language."},
            {"role": "user", "content": "Tell me more"},
        ]);
        let query = extract_query_text(Some(&input));
        assert!(query.contains("What is Rust?"));
        assert!(query.contains("Tell me more"));
    }

    #[test]
    fn extract_query_text_from_content_parts() {
        let input = json!([{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "search for cats"},
            ]
        }]);
        assert_eq!(extract_query_text(Some(&input)), "search for cats");
    }

    #[test]
    fn normalize_search_results_handles_empty() {
        assert!(normalize_search_results(None, 5).is_empty());
        assert!(normalize_search_results(Some(&json!([])), 5).is_empty());
    }

    #[test]
    fn normalize_search_results_extracts_fields() {
        let results = json!([
            {"title": "Rust", "url": "https://rust-lang.org", "content": "Systems language"},
            {"title": "", "url": "", "content": "empty"},
        ]);
        let normalized = normalize_search_results(Some(&results), 10);
        assert_eq!(normalized.len(), 1);
        assert_eq!(normalized[0]["title"], "Rust");
        assert_eq!(normalized[0]["snippet"], "Systems language");
    }

    #[test]
    fn search_directory_finds_matching_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("hello.txt"), "world").unwrap();
        std::fs::write(dir.path().join("other.md"), "nothing").unwrap();

        let mut matches = Vec::new();
        search_directory(dir.path(), &["hello".to_owned()], &mut matches, 10);
        assert_eq!(matches.len(), 1);
        assert!(matches[0]["path"].as_str().unwrap().contains("hello.txt"));
    }

    #[test]
    fn match_content_finds_keyword() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "The quick brown fox jumps over the lazy dog").unwrap();

        let result = match_content(&path, &["brown".to_owned()]);
        assert!(result.is_some());
        assert!(result.unwrap().contains("brown"));
    }
}
