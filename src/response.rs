use serde_json::{Value, json};

use crate::translate::{AssistantTurn, RequestContext, ToolCall, function_item_id};

pub fn build_response(
    context: &RequestContext,
    turn: &AssistantTurn,
    usage: Value,
    completed_at: i64,
) -> Value {
    build_response_with_status(context, turn, usage, completed_at, "completed", Value::Null)
}

pub fn build_cancelled_response(
    context: &RequestContext,
    turn: &AssistantTurn,
    usage: Value,
    completed_at: i64,
) -> Value {
    build_response_with_status(
        context,
        turn,
        usage,
        completed_at,
        "cancelled",
        json!({"reason":"cancelled"}),
    )
}

pub fn build_failed_response(
    context: &RequestContext,
    turn: &AssistantTurn,
    completed_at: i64,
    message: &str,
    code: &str,
) -> Value {
    json!({
        "id": context.response_id,
        "object": "response",
        "created_at": context.created_at,
        "completed_at": completed_at,
        "status": "failed",
        "error": {
            "message": message,
            "type": "api_connection_error",
            "code": code,
        },
        "incomplete_details": { "reason": "error" },
        "instructions": context.instructions,
        "max_output_tokens": context.max_output_tokens,
        "metadata": context.metadata,
        "model": context.client_model,
        "output": build_output_items(context, turn, "incomplete"),
        "output_text": turn.text,
        "parallel_tool_calls": context.parallel_tool_calls,
        "previous_response_id": context.previous_response_id,
        "store": context.store,
        "temperature": context.temperature,
        "top_p": context.top_p,
        "truncation": context.truncation,
        "text": { "format": { "type": "text" } },
        "tool_choice": context.tool_choice,
        "tools": context.tools,
        "usage": empty_usage(),
    })
}

pub fn build_in_progress_response(context: &RequestContext) -> Value {
    json!({
        "id": context.response_id,
        "object": "response",
        "created_at": context.created_at,
        "completed_at": null,
        "status": "in_progress",
        "error": null,
        "incomplete_details": null,
        "instructions": context.instructions,
        "max_output_tokens": context.max_output_tokens,
        "metadata": context.metadata,
        "model": context.client_model,
        "output": [],
        "output_text": "",
        "parallel_tool_calls": context.parallel_tool_calls,
        "previous_response_id": context.previous_response_id,
        "store": context.store,
        "temperature": context.temperature,
        "top_p": context.top_p,
        "truncation": context.truncation,
        "text": { "format": { "type": "text" } },
        "tool_choice": context.tool_choice,
        "tools": context.tools,
        "usage": empty_usage(),
    })
}

pub fn build_reasoning_item(context: &RequestContext, status: &str, reasoning: &str) -> Value {
    json!({
        "id": context.reasoning_id,
        "type": "reasoning",
        "status": status,
        "summary": [{
            "type": "summary_text",
            "text": reasoning,
        }],
    })
}

pub fn build_message_item(context: &RequestContext, status: &str, text: &str) -> Value {
    json!({
        "id": context.message_id,
        "type": "message",
        "status": status,
        "role": "assistant",
        "phase": "final_answer",
        "content": [{
            "type": "output_text",
            "text": text,
            "annotations": [],
            "logprobs": [],
        }],
    })
}

pub fn build_function_call_item(tool_call: &ToolCall, status: &str) -> Value {
    json!({
        "id": function_item_id(&tool_call.call_id),
        "type": "function_call",
        "call_id": tool_call.call_id,
        "name": tool_call.name,
        "arguments": tool_call.arguments,
        "status": status,
    })
}

pub fn build_content_part(text: &str) -> Value {
    json!({
        "type": "output_text",
        "text": text,
        "annotations": [],
        "logprobs": [],
    })
}

fn build_response_with_status(
    context: &RequestContext,
    turn: &AssistantTurn,
    usage: Value,
    completed_at: i64,
    status: &str,
    incomplete_details: Value,
) -> Value {
    json!({
        "id": context.response_id,
        "object": "response",
        "created_at": context.created_at,
        "completed_at": completed_at,
        "status": status,
        "error": null,
        "incomplete_details": incomplete_details,
        "instructions": context.instructions,
        "max_output_tokens": context.max_output_tokens,
        "metadata": context.metadata,
        "model": context.client_model,
        "output": build_output_items(context, turn, status),
        "output_text": turn.text,
        "parallel_tool_calls": context.parallel_tool_calls,
        "previous_response_id": context.previous_response_id,
        "store": context.store,
        "temperature": context.temperature,
        "top_p": context.top_p,
        "truncation": context.truncation,
        "text": { "format": { "type": "text" } },
        "tool_choice": context.tool_choice,
        "tools": context.tools,
        "usage": usage,
    })
}

fn build_output_items(context: &RequestContext, turn: &AssistantTurn, status: &str) -> Vec<Value> {
    let mut output = Vec::new();
    output.extend(context.hosted_output_items.iter().cloned());
    if !context.skip_reasoning_output && !turn.reasoning.is_empty() {
        output.push(build_reasoning_item(context, status, &turn.reasoning));
    }
    if !turn.text.is_empty() || turn.tool_calls.is_empty() {
        output.push(build_message_item(context, status, &turn.text));
    }
    output.extend(
        turn.tool_calls
            .iter()
            .map(|tool_call| build_function_call_item(tool_call, status)),
    );
    output
}

fn empty_usage() -> Value {
    json!({
        "input_tokens": 0,
        "input_tokens_details": { "cached_tokens": 0 },
        "output_tokens": 0,
        "output_tokens_details": { "reasoning_tokens": 0 },
        "total_tokens": 0,
    })
}
