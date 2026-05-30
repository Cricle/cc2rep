use serde_json::Value;

use crate::{
    error::ProxyError,
    tools::{ToolExecutor, append_tool_outputs},
    translate::{
        merge_usage, parse_assistant_turn_from_response, usage_from_upstream,
    },
};

use super::{AppState, NonStreamExecution};
pub(crate) async fn execute_non_stream_turn(
    state: &AppState,
    translated: &crate::translate::TranslatedRequest,
) -> Result<NonStreamExecution, ProxyError> {
    let mut upstream_payload = translated.upstream_payload.clone();
    let mut last_upstream = state.upstream.chat_json(&upstream_payload).await?;
    let mut turn = parse_assistant_turn_from_response(&last_upstream)?;
    let mut total_usage = usage_from_upstream(last_upstream.get("usage"));

    if !state.tool_executor.has_local_tools() {
        return Ok(NonStreamExecution {
            turn,
            usage: total_usage,
            request_messages: translated.request_messages.clone(),
        });
    }

    let max_tool_calls = translated.context.max_tool_calls;
    let mut total_tool_calls: u32 = 0;

    for _ in 0..state.settings.max_auto_tool_rounds {
        let supported: Vec<_> = turn
            .tool_calls
            .iter()
            .filter(|tc| state.tool_executor.supports_tool(&tc.name))
            .cloned()
            .collect();
        if supported.is_empty() {
            break;
        }
        if let Some(limit) = max_tool_calls {
            total_tool_calls += supported.len() as u32;
            if total_tool_calls > limit {
                break;
            }
        }

        let Some(outputs) = state
            .tool_executor
            .execute_calls(&supported, translated.context.parallel_tool_calls)
            .await?
        else {
            break;
        };
        {
            let Some(payload_map) = upstream_payload.as_object_mut() else {
                break;
            };
            let Some(messages) = payload_map
                .get_mut("messages")
                .and_then(Value::as_array_mut)
            else {
                break;
            };
            append_tool_outputs(messages, &supported, &outputs, Some(&turn.reasoning));
        }

        last_upstream = state.upstream.chat_json(&upstream_payload).await?;
        merge_usage(
            &mut total_usage,
            &usage_from_upstream(last_upstream.get("usage")),
        );
        let next_turn = parse_assistant_turn_from_response(&last_upstream)?;
        if next_turn.tool_calls.is_empty() {
            return Ok(NonStreamExecution {
                turn: next_turn,
                usage: total_usage,
                request_messages: super::extract_request_messages(&upstream_payload),
            });
        }
        turn = next_turn;
    }

    Ok(NonStreamExecution {
        turn,
        usage: total_usage,
        request_messages: super::extract_request_messages(&upstream_payload),
    })
}

pub(crate) fn should_auto_execute_tools(
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

