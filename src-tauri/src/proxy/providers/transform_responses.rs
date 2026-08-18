use crate::proxy::{
    error::ProxyError,
    json_canonical::canonical_json_string,
    tool_media::{
        strip_and_clamp_media_from_tool_value, ToolMediaScope, TOOL_RESULT_MEDIA_ATTACHED_MARKER,
    },
};
use aho_corasick::AhoCorasick;
use serde_json::{json, Value};
use std::collections::{HashMap, VecDeque};

use super::web_search_bridge::{
    citation_titles_by_exact_url, fold_anthropic_web_search_history, map_anthropic_web_search_tool,
    render_anthropic_citation, render_anthropic_web_search_blocks, WebSearchRequestPolicy,
    WEB_SEARCH_SOURCES_INCLUDE,
};

pub(crate) const RESPONSES_PROXY_THINKING_SIGNATURE: &str = "cc-switch-responses-summary-v1";
const RESPONSES_CITATION_INDEX_PREFIX: &str = "ccs-cit1:";
const RESPONSES_REASONING_PREFIX: &str = "cc-switch-responses-reasoning-v1:";

pub(crate) const TOOL_RESULT_ERROR_MARKER: &str = "[cc-switch:tool-result-error]";

pub(crate) fn responses_reasoning_redacted_block(item: &Value) -> Option<Value> {
    let encrypted_content = item
        .get("encrypted_content")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())?;
    let sidecar = json!({
        "id": item.get("id").and_then(Value::as_str).unwrap_or(""),
        "encrypted_content": encrypted_content
    });
    Some(json!({
        "type": "redacted_thinking",
        "data": format!(
            "{RESPONSES_REASONING_PREFIX}{}",
            serde_json::to_string(&sidecar).ok()?
        )
    }))
}

fn responses_reasoning_from_redacted_block(block: &Value) -> Option<Value> {
    let payload = block
        .get("data")
        .and_then(Value::as_str)?
        .strip_prefix(RESPONSES_REASONING_PREFIX)?;
    let sidecar: Value = serde_json::from_str(payload).ok()?;
    let encrypted_content = sidecar
        .get("encrypted_content")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())?;
    let mut item = json!({
        "type": "reasoning",
        "summary": [],
        "encrypted_content": encrypted_content
    });
    if let Some(id) = sidecar
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
    {
        item["id"] = json!(id);
    }
    Some(item)
}

fn anthropic_image_to_responses_part(block: &Value) -> Option<Value> {
    let source = block.get("source")?;
    match source.get("type").and_then(Value::as_str) {
        Some("url") => source
            .get("url")
            .and_then(Value::as_str)
            .filter(|url| url.starts_with("http://") || url.starts_with("https://"))
            .map(|url| json!({"type":"input_image","image_url":url})),
        Some("base64") | None => {
            let data = source.get("data").and_then(Value::as_str)?;
            if data.is_empty() {
                return None;
            }
            let media_type = source
                .get("media_type")
                .and_then(Value::as_str)
                .unwrap_or("image/png");
            Some(json!({
                "type":"input_image",
                "image_url":format!("data:{media_type};base64,{data}")
            }))
        }
        _ => None,
    }
}

fn anthropic_document_to_responses_part(block: &Value) -> Option<Value> {
    let source = block.get("source")?;
    let filename = block
        .get("title")
        .or_else(|| block.get("filename"))
        .and_then(Value::as_str)
        .unwrap_or("document.pdf");
    match source.get("type").and_then(Value::as_str) {
        Some("url") => source
            .get("url")
            .and_then(Value::as_str)
            .filter(|url| url.starts_with("http://") || url.starts_with("https://"))
            .map(|url| json!({"type":"input_file","file_url":url,"filename":filename})),
        Some("base64") => {
            let data = source.get("data").and_then(Value::as_str)?;
            if data.is_empty() {
                return None;
            }
            let media_type = source
                .get("media_type")
                .and_then(Value::as_str)
                .unwrap_or("application/pdf");
            Some(json!({
                "type":"input_file",
                "file_data":format!("data:{media_type};base64,{data}"),
                "filename":filename
            }))
        }
        _ => None,
    }
}

fn anthropic_tool_result_to_responses_output(block: &Value) -> Value {
    let is_error = block.get("is_error").and_then(Value::as_bool) == Some(true);
    let content = block.get("content");

    if !is_error {
        if let Some(text @ Value::String(_)) = content {
            if let Some(output) = alternate_image_tool_result_to_responses(text) {
                return Value::Array(output);
            }
            return text.clone();
        }
    }

    let mut output = Vec::new();
    if is_error {
        output.push(json!({"type":"input_text","text":TOOL_RESULT_ERROR_MARKER}));
    }

    match content {
        Some(Value::String(text)) => {
            if let Some(mut alternate) =
                alternate_image_tool_result_to_responses(&Value::String(text.clone()))
            {
                output.append(&mut alternate);
            } else {
                output.push(json!({"type":"input_text","text":text}));
            }
        }
        Some(Value::Array(blocks)) => {
            for part in blocks {
                match part.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        if let Some(text) = part.get("text").and_then(Value::as_str) {
                            output.push(json!({"type":"input_text","text":text}));
                        }
                    }
                    Some("image") => {
                        if let Some(image) = anthropic_image_to_responses_part(part) {
                            output.push(image);
                        } else if let Some(mut alternate) =
                            alternate_image_tool_result_to_responses(part)
                        {
                            output.append(&mut alternate);
                        } else {
                            output.push(json!({
                                "type":"input_text",
                                "text":canonical_json_string(part)
                            }));
                        }
                    }
                    Some("document") => {
                        if let Some(file) = anthropic_document_to_responses_part(part) {
                            output.push(file);
                        } else {
                            output.push(json!({
                                "type":"input_text",
                                "text":canonical_json_string(part)
                            }));
                        }
                    }
                    _ => {
                        if let Some(mut alternate) = alternate_image_tool_result_to_responses(part)
                        {
                            output.append(&mut alternate);
                        } else {
                            output.push(json!({
                                "type":"input_text",
                                "text":canonical_json_string(part)
                            }));
                        }
                    }
                }
            }
        }
        Some(value) => {
            if let Some(mut alternate) = alternate_image_tool_result_to_responses(value) {
                output.append(&mut alternate);
            } else {
                output.push(json!({
                    "type":"input_text",
                    "text":canonical_json_string(value)
                }));
            }
        }
        None => {}
    }

    Value::Array(output)
}

fn alternate_image_tool_result_to_responses(value: &Value) -> Option<Vec<Value>> {
    let mut cleaned = value.clone();
    let replacement_block = json!({
        "type":"input_text",
        "text":TOOL_RESULT_MEDIA_ATTACHED_MARKER
    });
    let mut chat_media_parts = Vec::new();
    let replaced = strip_and_clamp_media_from_tool_value(
        &mut cleaned,
        &mut chat_media_parts,
        ToolMediaScope::ImagesOnly,
        &replacement_block,
        TOOL_RESULT_MEDIA_ATTACHED_MARKER,
    );
    if replaced == 0 {
        return None;
    }

    let mut output = Vec::new();
    append_sanitized_responses_tool_value(&cleaned, &mut output);
    output.extend(
        chat_media_parts
            .iter()
            .filter_map(responses_image_from_chat_media),
    );
    Some(output)
}

fn append_sanitized_responses_tool_value(value: &Value, output: &mut Vec<Value>) {
    match value {
        Value::String(text) if !text.is_empty() => {
            output.push(json!({"type":"input_text","text":text}));
        }
        Value::Array(parts) => {
            for part in parts {
                match part.get("type").and_then(Value::as_str) {
                    Some("input_text" | "output_text" | "text") => {
                        if let Some(text) = part.get("text").and_then(Value::as_str) {
                            output.push(json!({"type":"input_text","text":text}));
                        }
                    }
                    _ => output.push(json!({
                        "type":"input_text",
                        "text":canonical_json_string(part)
                    })),
                }
            }
        }
        Value::Object(object)
            if matches!(
                object.get("type").and_then(Value::as_str),
                Some("input_text" | "output_text" | "text")
            ) =>
        {
            if let Some(text) = object.get("text").and_then(Value::as_str) {
                output.push(json!({"type":"input_text","text":text}));
            }
        }
        Value::Null | Value::String(_) => {}
        other => output.push(json!({
            "type":"input_text",
            "text":canonical_json_string(other)
        })),
    }
}

fn responses_image_from_chat_media(part: &Value) -> Option<Value> {
    let image_url = part
        .pointer("/image_url/url")
        .and_then(Value::as_str)
        .filter(|url| !url.trim().is_empty())?;
    let mut image = json!({
        "type":"input_image",
        "image_url":image_url
    });
    if let Some(detail) = part.pointer("/image_url/detail") {
        image["detail"] = detail.clone();
    }
    Some(image)
}

pub(crate) fn sanitize_anthropic_tool_use_input(name: &str, input: Value) -> Value {
    if name != "Read" {
        return input;
    }

    match input {
        Value::Object(mut object) => {
            if matches!(object.get("pages"), Some(Value::String(value)) if value.is_empty()) {
                object.remove("pages");
            }
            Value::Object(object)
        }
        other => other,
    }
}

pub(crate) fn sanitize_anthropic_tool_use_input_json(name: &str, raw: &str) -> String {
    if name != "Read" || raw.is_empty() {
        return raw.to_string();
    }

    let Ok(input) = serde_json::from_str::<Value>(raw) else {
        return raw.to_string();
    };

    serde_json::to_string(&sanitize_anthropic_tool_use_input(name, input))
        .unwrap_or_else(|_| raw.to_string())
}

fn decode_responses_citation_index(citation: &Value) -> Option<(u64, u64)> {
    let encoded = citation.get("encrypted_index").and_then(Value::as_str)?;
    let indices = encoded.strip_prefix(RESPONSES_CITATION_INDEX_PREFIX)?;
    let (start, end) = indices.split_once(':')?;
    let start = start.parse().ok()?;
    let end = end.parse().ok()?;
    (end >= start).then_some((start, end))
}

pub fn anthropic_to_responses(
    body: Value,
    cache_key: Option<&str>,
    is_codex_oauth: bool,
    codex_fast_mode: bool,
) -> Result<Value, ProxyError> {
    let mut result = json!({});
    let web_search_policy = WebSearchRequestPolicy::from_anthropic_request(&body)?;
    let uses_hosted_web_search = web_search_policy.is_some();
    let mut include_reasoning = is_codex_oauth;

    if let Some(model) = body.get("model").and_then(|m| m.as_str()) {
        result["model"] = json!(model);
    }

    if let Some(system) = body.get("system") {
        let instructions = if let Some(text) = system.as_str() {
            super::transform::strip_leading_anthropic_billing_header(text).to_string()
        } else if let Some(arr) = system.as_array() {
            arr.iter()
                .filter_map(|msg| msg.get("text").and_then(|t| t.as_str()))
                .map(super::transform::strip_leading_anthropic_billing_header)
                .filter(|text| !text.is_empty())
                .map(str::to_string)
                .collect::<Vec<_>>()
                .join("\n\n")
        } else {
            String::new()
        };

        if !instructions.is_empty() {
            result["instructions"] = json!(instructions);
        }
    }

    if let Some(msgs) = body.get("messages").and_then(|m| m.as_array()) {
        result["input"] = json!(convert_messages_to_input(msgs, uses_hosted_web_search)?);
    }

    if let Some(v) = body.get("max_tokens") {
        result["max_output_tokens"] = v.clone();
    }
    if let Some(v) = body.get("temperature") {
        result["temperature"] = v.clone();
    }
    if let Some(v) = body.get("top_p") {
        result["top_p"] = v.clone();
    }
    // Use the Responses streaming transport for every bridged request. Some
    // Responses-compatible relays only implement the streaming wire contract.
    // The Claude handler preserves the client-facing mode: streaming Claude
    // requests are converted incrementally, while non-streaming requests are
    // buffered and folded back into one Anthropic message response.
    result["stream"] = json!(true);

    if let Some(model_name) = body.get("model").and_then(|m| m.as_str()) {
        if super::transform::supports_reasoning_effort(model_name) {
            if uses_hosted_web_search {
                include_reasoning = true;
            }
            if let Some(effort) = super::transform::resolve_reasoning_effort(&body) {
                result["reasoning"] = json!({ "effort": effort });
            }
        }
    }

    if let Some(tools) = body.get("tools").and_then(|t| t.as_array()) {
        let mut response_tools = Vec::new();
        for tool in tools {
            let tool_type = tool.get("type").and_then(Value::as_str);
            if tool_type == Some("BatchTool") {
                continue;
            }
            if tool_type
                .is_some_and(|value| value == "web_search" || value.starts_with("web_search_"))
            {
                response_tools.push(map_anthropic_web_search_tool(tool)?);
            } else {
                response_tools.push(json!({
                    "type": "function",
                    "name": tool.get("name").and_then(|n| n.as_str()).unwrap_or(""),
                    "description": tool.get("description"),
                    "parameters": super::transform::clean_schema(
                        tool.get("input_schema").cloned().unwrap_or(json!({}))
                    )
                }));
            }
        }

        if !response_tools.is_empty() {
            result["tools"] = json!(response_tools);
        }
    }

    if uses_hosted_web_search {
        let mut includes = body
            .get("include")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        if !includes
            .iter()
            .any(|value| value.as_str() == Some(WEB_SEARCH_SOURCES_INCLUDE))
        {
            includes.push(json!(WEB_SEARCH_SOURCES_INCLUDE));
        }
        result["include"] = json!(includes);
    }
    // Anthropic max_uses counts searches only. Responses max_tool_calls counts all
    // hosted-tool actions (including open_page/find_in_page), so it is not mapped.

    if include_reasoning {
        const REASONING_MARKER: &str = "reasoning.encrypted_content";
        let mut includes: Vec<Value> = result
            .get("include")
            .and_then(Value::as_array)
            .cloned()
            .or_else(|| body.get("include").and_then(Value::as_array).cloned())
            .unwrap_or_default();
        if !includes
            .iter()
            .any(|value| value.as_str() == Some(REASONING_MARKER))
        {
            includes.push(json!(REASONING_MARKER));
        }
        result["include"] = json!(includes);
    }

    if let Some(v) = body.get("tool_choice") {
        result["tool_choice"] = map_tool_choice_to_responses(v, uses_hosted_web_search);
        if uses_hosted_web_search
            && v.get("disable_parallel_tool_use").and_then(Value::as_bool) == Some(true)
        {
            result["parallel_tool_calls"] = json!(false);
        }
    }

    if let Some(key) = cache_key {
        result["prompt_cache_key"] = json!(key);
    }

    if is_codex_oauth {
        result["store"] = json!(false);
        if codex_fast_mode {
            result["service_tier"] = json!("priority");
        }

        if let Some(obj) = result.as_object_mut() {
            obj.remove("max_output_tokens");
            obj.remove("temperature");
            obj.remove("top_p");
            obj.entry("instructions".to_string()).or_insert(json!(""));
            obj.entry("tools".to_string()).or_insert(json!([]));
            obj.entry("parallel_tool_calls".to_string())
                .or_insert(json!(false));
        }
    }

    Ok(result)
}

fn map_tool_choice_to_responses(tool_choice: &Value, uses_hosted_web_search: bool) -> Value {
    match tool_choice {
        Value::String(_) => tool_choice.clone(),
        Value::Object(obj) => match obj.get("type").and_then(|t| t.as_str()) {
            Some("any") => json!("required"),
            Some("auto") => json!("auto"),
            Some("none") => json!("none"),
            Some("tool") => {
                let name = obj.get("name").and_then(|n| n.as_str()).unwrap_or("");
                if uses_hosted_web_search && name == "web_search" {
                    return json!({ "type": "web_search" });
                }
                json!({
                    "type": "function",
                    "name": name
                })
            }
            _ => tool_choice.clone(),
        },
        _ => tool_choice.clone(),
    }
}

pub(crate) fn map_responses_stop_reason(
    status: Option<&str>,
    has_tool_use: bool,
    incomplete_reason: Option<&str>,
) -> Option<&'static str> {
    status.map(|s| match s {
        "completed" => {
            if has_tool_use {
                "tool_use"
            } else {
                "end_turn"
            }
        }
        "incomplete" => {
            if matches!(
                incomplete_reason,
                Some("max_output_tokens") | Some("max_tokens")
            ) || incomplete_reason.is_none()
            {
                "max_tokens"
            } else {
                "end_turn"
            }
        }
        _ => "end_turn",
    })
}

pub(crate) fn build_anthropic_usage_from_responses(usage: Option<&Value>) -> Value {
    let u = match usage {
        Some(v) if !v.is_null() => v,
        _ => {
            return json!({
                "input_tokens": 0,
                "output_tokens": 0
            })
        }
    };

    let input = u
        .get("input_tokens")
        .and_then(|v| v.as_u64())
        .or_else(|| u.get("prompt_tokens").and_then(|v| v.as_u64()))
        .unwrap_or(0);
    let output = u
        .get("output_tokens")
        .and_then(|v| v.as_u64())
        .or_else(|| u.get("completion_tokens").and_then(|v| v.as_u64()))
        .unwrap_or(0);

    let mut result = json!({
        "input_tokens": input,
        "output_tokens": output
    });

    if let Some(cached) = u
        .pointer("/input_tokens_details/cached_tokens")
        .and_then(|v| v.as_u64())
    {
        result["cache_read_input_tokens"] = json!(cached);
    }
    if let Some(cached) = u
        .pointer("/prompt_tokens_details/cached_tokens")
        .and_then(|v| v.as_u64())
    {
        if result.get("cache_read_input_tokens").is_none() {
            result["cache_read_input_tokens"] = json!(cached);
        }
    }

    let nested_cache_write = u
        .pointer("/input_tokens_details/cache_write_tokens")
        .and_then(|v| v.as_u64())
        .or_else(|| {
            u.pointer("/prompt_tokens_details/cache_write_tokens")
                .and_then(|v| v.as_u64())
        });
    if let Some(cache_write) = nested_cache_write {
        result["cache_creation_input_tokens"] = json!(cache_write);
    }

    if let Some(v) = u.get("cache_read_input_tokens") {
        result["cache_read_input_tokens"] = v.clone();
    }
    if let Some(v) = u.get("cache_creation_input_tokens") {
        result["cache_creation_input_tokens"] = v.clone();
    }

    // OpenAI/Responses input (prompt_tokens/input_tokens) is cache-inclusive; Anthropic
    // input_tokens is fresh. This mapping is claude-billed only (Codex passthrough uses
    // from_codex_response_*), so subtract cache_read + cache_creation to avoid counting
    // cached tokens both as input and in the cache buckets. Buckets are mutually exclusive:
    // input + cache_read + cache_creation == upstream input. Covers non-streaming and
    // streaming (streaming_responses).
    let cached = result
        .get("cache_read_input_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let cache_creation = result
        .get("cache_creation_input_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    if cached > 0 || cache_creation > 0 {
        result["input_tokens"] = json!(input.saturating_sub(cached).saturating_sub(cache_creation));
    }

    result
}

pub(crate) fn add_web_search_usage(mut usage: Value, web_search_requests: u64) -> Value {
    if web_search_requests > 0 {
        usage["server_tool_use"] = json!({
            "web_search_requests": web_search_requests
        });
    }
    usage
}

fn responses_annotations_from_anthropic_text(block: &Value, text: &str) -> Vec<Value> {
    let citations: Vec<&Value> = block
        .get("citations")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|citation| {
            citation.get("type").and_then(Value::as_str) == Some("web_search_result_location")
        })
        .collect();

    let mut pattern_index: HashMap<String, usize> = HashMap::new();
    let mut patterns = Vec::new();
    let mut span_limits = Vec::new();
    for citation in &citations {
        let mut counted_patterns = std::collections::HashSet::new();
        for pattern in [
            citation.get("cited_text").and_then(Value::as_str),
            citation.get("url").and_then(Value::as_str),
        ]
        .into_iter()
        .flatten()
        .filter(|pattern| !pattern.is_empty())
        {
            let index = if let Some(index) = pattern_index.get(pattern).copied() {
                index
            } else {
                let index = patterns.len();
                patterns.push(pattern.to_string());
                pattern_index.insert(pattern.to_string(), index);
                span_limits.push(0usize);
                index
            };
            if counted_patterns.insert(index) {
                span_limits[index] += 1;
            }
        }
    }

    let mut spans: Vec<VecDeque<(u64, u64)>> = vec![VecDeque::new(); patterns.len()];
    if !patterns.is_empty() {
        if let Ok(matcher) = AhoCorasick::new(&patterns) {
            let mut char_index_by_byte: HashMap<usize, u64> = text
                .char_indices()
                .enumerate()
                .map(|(char_index, (byte_index, _))| (byte_index, char_index as u64))
                .collect();
            char_index_by_byte.insert(text.len(), text.chars().count() as u64);
            for matched in matcher.find_overlapping_iter(text) {
                let pattern_index = matched.pattern().as_usize();
                if spans[pattern_index].len() >= span_limits[pattern_index] {
                    continue;
                }
                if let (Some(start), Some(end)) = (
                    char_index_by_byte.get(&matched.start()),
                    char_index_by_byte.get(&matched.end()),
                ) {
                    spans[pattern_index].push_back((*start, *end));
                }
            }
        }
    }

    citations
        .into_iter()
        .filter_map(|citation| {
            let url = citation
                .get("url")
                .and_then(Value::as_str)
                .filter(|url| !url.trim().is_empty())?;
            let title = citation
                .get("title")
                .and_then(Value::as_str)
                .filter(|title| !title.trim().is_empty())
                .unwrap_or(url);
            let (start_index, end_index) = decode_responses_citation_index(citation)
                .unwrap_or_else(|| {
                    let cited_text = citation
                        .get("cited_text")
                        .and_then(Value::as_str)
                        .filter(|cited_text| !cited_text.is_empty());
                    cited_text
                        .and_then(|cited_text| pattern_index.get(cited_text).copied())
                        .and_then(|index| spans[index].pop_front())
                        .or_else(|| {
                            pattern_index
                                .get(url)
                                .copied()
                                .and_then(|index| spans[index].pop_front())
                        })
                        .unwrap_or((0, 0))
                });
            Some(json!({
                "type": "url_citation",
                "start_index": start_index,
                "end_index": end_index,
                "title": title,
                "url": url
            }))
        })
        .collect()
}

fn convert_messages_to_input(
    messages: &[Value],
    preserve_responses_reasoning: bool,
) -> Result<Vec<Value>, ProxyError> {
    let mut input = Vec::new();

    for msg in messages {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");
        let content = msg.get("content");

        match content {
            Some(Value::String(text)) => {
                let content_type = if role == "assistant" {
                    "output_text"
                } else {
                    "input_text"
                };
                input.push(json!({
                    "role": role,
                    "content": [{ "type": content_type, "text": text }]
                }));
            }
            Some(Value::Array(blocks)) => {
                let mut message_content = Vec::new();
                let mut block_index = 0usize;
                while block_index < blocks.len() {
                    let block = &blocks[block_index];
                    let block_type = block.get("type").and_then(|t| t.as_str()).unwrap_or("");

                    match block_type {
                        "text" => {
                            if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                                let content_type = if role == "assistant" {
                                    "output_text"
                                } else {
                                    "input_text"
                                };
                                let mut text_part = json!({
                                    "type": content_type,
                                    "text": text
                                });
                                if role == "assistant" && preserve_responses_reasoning {
                                    let annotations =
                                        responses_annotations_from_anthropic_text(block, text);
                                    if !annotations.is_empty() {
                                        text_part["annotations"] = json!(annotations);
                                    }
                                }
                                message_content.push(text_part);
                            }
                        }
                        "image" => {
                            if let Some(image) = anthropic_image_to_responses_part(block) {
                                message_content.push(image);
                            } else {
                                log::warn!(
                                    "[Responses] Unsupported or invalid Anthropic image block"
                                );
                            }
                        }
                        "document" => {
                            if let Some(file) = anthropic_document_to_responses_part(block) {
                                message_content.push(file);
                            } else {
                                log::warn!(
                                    "[Responses] Unsupported or invalid Anthropic document block"
                                );
                            }
                        }
                        "tool_use" => {
                            if !message_content.is_empty() {
                                input.push(json!({
                                    "role": role,
                                    "content": message_content.clone()
                                }));
                                message_content.clear();
                            }

                            let id = block.get("id").and_then(|i| i.as_str()).unwrap_or("");
                            let name = block.get("name").and_then(|n| n.as_str()).unwrap_or("");
                            let arguments = block.get("input").cloned().unwrap_or(json!({}));

                            input.push(json!({
                                "type": "function_call",
                                "call_id": id,
                                "name": name,
                                "arguments": canonical_json_string(&arguments)
                            }));
                        }
                        "tool_result" => {
                            if !message_content.is_empty() {
                                input.push(json!({
                                    "role": role,
                                    "content": message_content.clone()
                                }));
                                message_content.clear();
                            }

                            let call_id = block
                                .get("tool_use_id")
                                .and_then(|i| i.as_str())
                                .unwrap_or("");
                            let output = anthropic_tool_result_to_responses_output(block);

                            input.push(json!({
                                "type": "function_call_output",
                                "call_id": call_id,
                                "output": output
                            }));
                        }
                        "server_tool_use" => {
                            if !preserve_responses_reasoning {
                                block_index += 1;
                                continue;
                            }
                            if block.get("name").and_then(Value::as_str) != Some("web_search") {
                                block_index += 1;
                                continue;
                            }
                            if !message_content.is_empty() {
                                input.push(json!({
                                    "role": role,
                                    "content": message_content.clone()
                                }));
                                message_content.clear();
                            }
                            let (web_search, next_index) =
                                fold_anthropic_web_search_history(blocks, block_index)?
                                    .ok_or_else(|| {
                                        ProxyError::InvalidRequest(
                                            "unsupported server_tool_use block".to_string(),
                                        )
                                    })?;
                            input.push(web_search);
                            block_index = next_index;
                            continue;
                        }
                        "web_search_tool_result" => {
                            if preserve_responses_reasoning {
                                return Err(ProxyError::InvalidRequest(
                                    "orphan web_search_tool_result block".to_string(),
                                ));
                            }
                        }
                        "redacted_thinking" => {
                            if preserve_responses_reasoning {
                                if let Some(reasoning) =
                                    responses_reasoning_from_redacted_block(block)
                                {
                                    if !message_content.is_empty() {
                                        input.push(json!({
                                            "role": role,
                                            "content": message_content.clone()
                                        }));
                                        message_content.clear();
                                    }
                                    input.push(reasoning);
                                }
                            }
                        }
                        "thinking" => {}
                        _ => {}
                    }
                    block_index += 1;
                }

                if !message_content.is_empty() {
                    input.push(json!({
                        "role": role,
                        "content": message_content
                    }));
                }
            }
            _ => {
                input.push(json!({ "role": role }));
            }
        }
    }

    Ok(input)
}

pub fn responses_to_anthropic(body: Value) -> Result<Value, ProxyError> {
    responses_to_anthropic_impl(body, false)
}

pub(crate) fn responses_to_anthropic_web_search(body: Value) -> Result<Value, ProxyError> {
    responses_to_anthropic_impl(body, true)
}

fn responses_to_anthropic_impl(
    body: Value,
    preserve_responses_reasoning: bool,
) -> Result<Value, ProxyError> {
    let output = body
        .get("output")
        .and_then(|o| o.as_array())
        .ok_or_else(|| ProxyError::TransformError("No output in response".to_string()))?;

    let mut content = Vec::new();
    let mut has_tool_use = false;
    let web_search_titles = preserve_responses_reasoning
        .then(|| citation_titles_by_exact_url(output))
        .unwrap_or_default();

    for item in output {
        let item_type = item.get("type").and_then(|t| t.as_str()).unwrap_or("");

        match item_type {
            "message" => {
                if let Some(msg_content) = item.get("content").and_then(|c| c.as_array()) {
                    for block in msg_content {
                        let block_type = block.get("type").and_then(|t| t.as_str()).unwrap_or("");
                        if block_type == "output_text" {
                            if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                                if !text.is_empty() {
                                    let mut citations = Vec::new();
                                    if preserve_responses_reasoning {
                                        let scalar_boundaries =
                                            super::web_search_bridge::unicode_scalar_boundaries(
                                                text,
                                            );
                                        for annotation in block
                                            .get("annotations")
                                            .and_then(Value::as_array)
                                            .into_iter()
                                            .flatten()
                                        {
                                            if let Some(citation) = render_anthropic_citation(
                                                annotation,
                                                text,
                                                &scalar_boundaries,
                                            )? {
                                                citations.push(citation);
                                            }
                                        }
                                    }
                                    let mut text_block = json!({"type": "text", "text": text});
                                    if !citations.is_empty() {
                                        text_block["citations"] = json!(citations);
                                    }
                                    content.push(text_block);
                                }
                            }
                        } else if block_type == "refusal" {
                            if let Some(refusal) = block.get("refusal").and_then(|t| t.as_str()) {
                                if !refusal.is_empty() {
                                    content.push(json!({"type": "text", "text": refusal}));
                                }
                            }
                        }
                    }
                }
            }
            "function_call" => {
                let call_id = item.get("call_id").and_then(|i| i.as_str()).unwrap_or("");
                let name = item.get("name").and_then(|n| n.as_str()).unwrap_or("");
                let args_str = item
                    .get("arguments")
                    .and_then(|a| a.as_str())
                    .unwrap_or("{}");
                let input: Value = serde_json::from_str(args_str).unwrap_or(json!({}));
                let input = sanitize_anthropic_tool_use_input(name, input);

                content.push(json!({
                    "type": "tool_use",
                    "id": call_id,
                    "name": name,
                    "input": input
                }));
                has_tool_use = true;
            }
            "web_search_call" => {
                if preserve_responses_reasoning {
                    let [server_tool_use, result] =
                        render_anthropic_web_search_blocks(item, &web_search_titles)?;
                    content.push(server_tool_use);
                    content.push(result);
                }
            }
            "reasoning" => {
                let summary_text: String = item
                    .get("summary")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(|part| {
                        (part.get("type").and_then(Value::as_str) == Some("summary_text"))
                            .then(|| part.get("text").and_then(Value::as_str))
                            .flatten()
                    })
                    .collect();
                let reasoning_text: String = if preserve_responses_reasoning {
                    item.get("content")
                        .and_then(Value::as_array)
                        .into_iter()
                        .flatten()
                        .filter_map(|part| {
                            (part.get("type").and_then(Value::as_str) == Some("reasoning_text"))
                                .then(|| part.get("text").and_then(Value::as_str))
                                .flatten()
                        })
                        .collect()
                } else {
                    String::new()
                };
                let thinking_text = format!("{summary_text}{reasoning_text}");
                if !thinking_text.is_empty() {
                    let mut thinking = json!({
                        "type": "thinking", "thinking": thinking_text
                    });
                    if preserve_responses_reasoning {
                        thinking["signature"] = json!(RESPONSES_PROXY_THINKING_SIGNATURE);
                    }
                    content.push(thinking);
                }
                if preserve_responses_reasoning {
                    if let Some(redacted) = responses_reasoning_redacted_block(item) {
                        content.push(redacted);
                    }
                }
            }
            _ => {}
        }
    }

    let web_search_requests = if preserve_responses_reasoning {
        output
            .iter()
            .filter(|item| {
                item.get("type").and_then(Value::as_str) == Some("web_search_call")
                    && item.get("status").and_then(Value::as_str) == Some("completed")
                    && item.pointer("/action/type").and_then(Value::as_str) == Some("search")
            })
            .count() as u64
    } else {
        0
    };

    Ok(json!({
        "id": body.get("id").and_then(|i| i.as_str()).unwrap_or(""),
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": body.get("model").and_then(|m| m.as_str()).unwrap_or(""),
        "stop_reason": map_responses_stop_reason(
            body.get("status").and_then(|s| s.as_str()),
            has_tool_use,
            body.pointer("/incomplete_details/reason")
                .and_then(|r| r.as_str()),
        ),
        "stop_sequence": null,
        "usage": add_web_search_usage(
            build_anthropic_usage_from_responses(body.get("usage")),
            web_search_requests
        )
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anthropic_to_responses_removes_billing_header_from_system_string() {
        let input = json!({
            "model": "gpt-5",
            "system": "x-anthropic-billing-header: cc_version=2.1.120.cf9; cc_entrypoint=cli; cch=543cf;\nYou are helpful.",
            "messages": [{"role": "user", "content": "Hello"}]
        });

        let result =
            anthropic_to_responses(input, None, false, false).expect("transform responses");

        assert_eq!(result["instructions"], json!("You are helpful."));
    }

    #[test]
    fn anthropic_to_responses_removes_billing_header_from_system_array() {
        let input = json!({
            "model": "gpt-5",
            "system": [{
                "type": "text",
                "text": "x-anthropic-billing-header: cc_version=2.1.120.cf9; cc_entrypoint=cli; cch=543cf;\nProject instructions"
            }],
            "messages": [{"role": "user", "content": "Hello"}]
        });

        let result =
            anthropic_to_responses(input, None, false, false).expect("transform responses");

        assert_eq!(result["instructions"], json!("Project instructions"));
    }

    #[test]
    fn anthropic_to_responses_omits_empty_billing_header_system_block() {
        let input = json!({
            "model": "gpt-5",
            "system": [{
                "type": "text",
                "text": "x-anthropic-billing-header: cc_version=2.1.120.cf9; cc_entrypoint=cli; cch=543cf;"
            }],
            "messages": [{"role": "user", "content": "Hello"}]
        });

        let result =
            anthropic_to_responses(input, None, false, false).expect("transform responses");

        assert!(result.get("instructions").is_none());
    }

    #[test]
    fn anthropic_to_responses_keeps_non_leading_billing_header_text() {
        let input = json!({
            "model": "gpt-5",
            "system": "Keep this literal:\nx-anthropic-billing-header: example",
            "messages": [{"role": "user", "content": "Hello"}]
        });

        let result =
            anthropic_to_responses(input, None, false, false).expect("transform responses");

        assert_eq!(
            result["instructions"],
            json!("Keep this literal:\nx-anthropic-billing-header: example")
        );
    }

    #[test]
    fn anthropic_to_responses_codex_oauth_sets_required_contract_fields() {
        let input = json!({
            "model": "gpt-5-codex",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}]
        });

        let result = anthropic_to_responses(input, None, true, true).expect("transform responses");

        assert_eq!(result["store"], json!(false));
        assert_eq!(result["service_tier"], json!("priority"));
        assert_eq!(result["include"], json!(["reasoning.encrypted_content"]));
        assert_eq!(result["instructions"], json!(""));
        assert_eq!(result["tools"], json!([]));
        assert_eq!(result["parallel_tool_calls"], json!(false));
        assert_eq!(result["stream"], json!(true));
    }

    #[test]
    fn anthropic_to_responses_codex_oauth_strips_unsupported_fields() {
        let input = json!({
            "model": "gpt-5-codex",
            "max_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.9,
            "messages": [{"role": "user", "content": "Hello"}]
        });

        let result = anthropic_to_responses(input, None, true, true).expect("transform responses");

        assert!(result.get("max_output_tokens").is_none());
        assert!(result.get("temperature").is_none());
        assert!(result.get("top_p").is_none());
    }

    #[test]
    fn anthropic_to_responses_non_codex_keeps_openai_fields() {
        let input = json!({
            "model": "gpt-5-codex",
            "max_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.9,
            "messages": [{"role": "user", "content": "Hello"}]
        });

        let result =
            anthropic_to_responses(input, None, false, false).expect("transform responses");

        assert_eq!(result["max_output_tokens"], json!(1024));
        assert_eq!(result["temperature"], json!(0.7));
        assert_eq!(result["top_p"], json!(0.9));
        assert_eq!(result["stream"], json!(true));
        assert!(result.get("store").is_none());
        assert!(result.get("service_tier").is_none());
    }

    #[test]
    fn anthropic_to_responses_tool_result_preserves_blocks_and_error() {
        let input = json!({
            "model":"gpt-5",
            "messages":[{"role":"user","content":[{
                "type":"tool_result",
                "tool_use_id":"call_1",
                "is_error":true,
                "content":[
                    {"type":"text","text":"command failed"},
                    {"type":"image","source":{"type":"url","url":"https://example.com/error.png"}},
                    {"type":"document","title":"trace.pdf","source":{"type":"base64","media_type":"application/pdf","data":"JVBERi0="}}
                ]
            }]}]
        });

        let result = anthropic_to_responses(input.clone(), None, false, false).unwrap();
        let output = result["input"][0]["output"].as_array().unwrap();

        assert_eq!(output[0]["text"], TOOL_RESULT_ERROR_MARKER);
        assert_eq!(
            output[1],
            json!({"type":"input_text","text":"command failed"})
        );
        assert_eq!(output[2]["image_url"], "https://example.com/error.png");
        assert_eq!(output[3]["type"], "input_file");
        assert_eq!(output[3]["filename"], "trace.pdf");
    }

    #[test]
    fn anthropic_to_responses_url_image_and_document() {
        let input = json!({
            "model":"gpt-5",
            "messages":[{"role":"user","content":[
                {"type":"image","source":{"type":"url","url":"https://example.com/a.png"}},
                {"type":"document","title":"manual.pdf","source":{"type":"url","url":"https://example.com/manual.pdf"}}
            ]}]
        });

        let result = anthropic_to_responses(input.clone(), None, false, false).unwrap();
        let content = result["input"][0]["content"].as_array().unwrap();

        assert_eq!(content[0]["type"], "input_image");
        assert_eq!(content[0]["image_url"], "https://example.com/a.png");
        assert_eq!(content[1]["type"], "input_file");
        assert_eq!(content[1]["file_url"], "https://example.com/manual.pdf");
        assert_eq!(content[1]["filename"], "manual.pdf");
    }

    #[test]
    fn anthropic_to_responses_converts_mcp_tool_image() {
        let input = json!({
            "model":"gpt-5",
            "messages":[{"role":"user","content":[{
                "type":"tool_result",
                "tool_use_id":"call_1",
                "content":[{
                    "type":"image",
                    "mimeType":"image/webp",
                    "data":"MCP_RESPONSES_IMAGE_SENTINEL"
                }]
            }]}]
        });

        let result = anthropic_to_responses(input, None, false, false).unwrap();
        let output = result["input"][0]["output"].as_array().unwrap();

        assert_eq!(output[0]["type"], "input_text");
        assert!(!output[0]["text"]
            .as_str()
            .unwrap()
            .contains("MCP_RESPONSES_IMAGE_SENTINEL"));
        assert_eq!(output[1]["type"], "input_image");
        assert_eq!(
            output[1]["image_url"],
            "data:image/webp;base64,MCP_RESPONSES_IMAGE_SENTINEL"
        );
    }

    #[test]
    fn anthropic_to_responses_converts_json_string_tool_image() {
        let residual_base64 = "A".repeat(20_000);
        let encoded = json!({
            "content":[
                {
                    "type":"image_url",
                    "image_url":{"url":"data:image/png;base64,STRING_RESPONSES_SENTINEL"}
                },
                {"type":"video","data":residual_base64}
            ]
        })
        .to_string();
        let input = json!({
            "model":"gpt-5",
            "messages":[{"role":"user","content":[{
                "type":"tool_result",
                "tool_use_id":"call_1",
                "content":encoded
            }]}]
        });

        let result = anthropic_to_responses(input, None, false, false).unwrap();
        let output = result["input"][0]["output"].as_array().unwrap();
        let image = output
            .iter()
            .find(|part| part["type"] == "input_image")
            .expect("stringified image must stay a Responses image");

        assert_eq!(
            image["image_url"],
            "data:image/png;base64,STRING_RESPONSES_SENTINEL"
        );
        assert!(output
            .iter()
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .all(|text| !text.contains("STRING_RESPONSES_SENTINEL")));
        let serialized = result.to_string();
        assert!(serialized.contains("[cc-switch: omitted 20000 bytes]"));
        assert!(!serialized.contains(&"A".repeat(64)));
    }

    #[test]
    fn anthropic_to_responses_maps_reasoning_effort_for_gpt5_models() {
        let input = json!({
            "model": "gpt-5.4",
            "max_tokens": 1024,
            "thinking": {"type": "adaptive"},
            "messages": [{"role": "user", "content": "Hello"}]
        });

        let result =
            anthropic_to_responses(input, None, false, false).expect("transform responses");

        assert_eq!(result["reasoning"]["effort"], json!("xhigh"));
        assert!(result.get("include").is_none());
    }

    #[test]
    fn anthropic_to_responses_codex_oauth_fast_mode_can_be_disabled() {
        let input = json!({
            "model": "gpt-5-codex",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}]
        });

        let result = anthropic_to_responses(input, None, true, false).expect("transform responses");

        assert_eq!(result["store"], json!(false));
        assert_eq!(result["include"], json!(["reasoning.encrypted_content"]));
        assert!(result.get("service_tier").is_none());
    }

    #[test]
    fn responses_to_anthropic_read_tool_drops_empty_pages() {
        let input = json!({
            "id": "resp_read",
            "status": "completed",
            "model": "gpt-5.4",
            "output": [{
                "type": "function_call",
                "call_id": "call_read",
                "name": "Read",
                "arguments": "{\"file_path\":\"/tmp/demo.py\",\"limit\":2000,\"offset\":0,\"pages\":\"\"}"
            }]
        });

        let result = responses_to_anthropic(input).expect("transform responses");

        assert_eq!(result["content"][0]["type"], "tool_use");
        assert_eq!(result["content"][0]["name"], "Read");
        assert!(result["content"][0]["input"].get("pages").is_none());
    }

    #[test]
    fn responses_to_anthropic_read_tool_preserves_whitespace_pages() {
        let input = json!({
            "id": "resp_read",
            "status": "completed",
            "model": "gpt-5.4",
            "output": [{
                "type": "function_call",
                "call_id": "call_read",
                "name": "Read",
                "arguments": "{\"file_path\":\"/tmp/demo.py\",\"pages\":\" \"}"
            }]
        });

        let result = responses_to_anthropic(input).expect("transform responses");

        assert_eq!(
            result["content"][0]["input"],
            json!({"file_path": "/tmp/demo.py", "pages": " "})
        );
    }

    #[test]
    fn responses_to_anthropic_read_tool_preserves_non_empty_pages() {
        let input = json!({
            "id": "resp_read",
            "status": "completed",
            "model": "gpt-5.4",
            "output": [{
                "type": "function_call",
                "call_id": "call_read",
                "name": "Read",
                "arguments": "{\"file_path\":\"/tmp/example.pdf\",\"pages\":\"1-3\"}"
            }]
        });

        let result = responses_to_anthropic(input).expect("transform responses");

        assert_eq!(
            result["content"][0]["input"],
            json!({"file_path": "/tmp/example.pdf", "pages": "1-3"})
        );
    }

    #[test]
    fn responses_to_anthropic_does_not_sanitize_non_read_tool_pages() {
        let input = json!({
            "id": "resp_1",
            "status": "completed",
            "model": "gpt-5.4",
            "output": [{
                "type": "function_call",
                "call_id": "call_other",
                "name": "OtherTool",
                "arguments": "{\"pages\":\"\"}"
            }]
        });

        let result = responses_to_anthropic(input).expect("transform responses");

        assert_eq!(result["content"][0]["input"], json!({"pages": ""}));
    }

    #[test]
    fn responses_usage_uses_openai_field_name_fallbacks() {
        let result = build_anthropic_usage_from_responses(Some(&json!({
            "prompt_tokens": 120,
            "completion_tokens": 45
        })));

        assert_eq!(result["input_tokens"], json!(120));
        assert_eq!(result["output_tokens"], json!(45));
    }

    #[test]
    fn responses_usage_prefers_anthropic_field_names() {
        let result = build_anthropic_usage_from_responses(Some(&json!({
            "input_tokens": 100,
            "prompt_tokens": 120,
            "output_tokens": 50,
            "completion_tokens": 45
        })));

        assert_eq!(result["input_tokens"], json!(100));
        assert_eq!(result["output_tokens"], json!(50));
    }

    #[test]
    fn responses_usage_subtracts_cache_from_input() {
        // input_tokens is cache-inclusive on the wire; Anthropic input_tokens must be
        // fresh input (this path is billed as claude, which does not subtract cache again).
        let result = build_anthropic_usage_from_responses(Some(&json!({
            "input_tokens": 100,
            "output_tokens": 50,
            "input_tokens_details": {"cached_tokens": 80}
        })));

        assert_eq!(result["input_tokens"], json!(20));
        assert_eq!(result["output_tokens"], json!(50));
        assert_eq!(result["cache_read_input_tokens"], json!(80));
    }

    #[test]
    fn responses_usage_maps_nested_cache_write_tokens() {
        let result = build_anthropic_usage_from_responses(Some(&json!({
            "input_tokens": 100,
            "output_tokens": 10,
            "input_tokens_details": {
                "cached_tokens": 30,
                "cache_write_tokens": 20
            }
        })));

        assert_eq!(result["input_tokens"], json!(50));
        assert_eq!(result["cache_read_input_tokens"], json!(30));
        assert_eq!(result["cache_creation_input_tokens"], json!(20));
    }

    #[test]
    fn anthropic_to_responses_defaults_missing_tool_schema_type() {
        let input = json!({
            "model": "gpt-4o",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Weather?"}],
            "tools": [{
                "name": "get_weather",
                "description": "Get weather info",
                "input_schema": {"properties": {"location": {"type": "string"}}}
            }]
        });

        let result = anthropic_to_responses(input, None, false, false).unwrap();
        let parameters = &result["tools"][0]["parameters"];

        assert_eq!(parameters["type"], json!("object"));
        assert_eq!(
            parameters["properties"]["location"]["type"],
            json!("string")
        );
    }

    #[test]
    fn anthropic_to_responses_defaults_empty_tool_schema() {
        let input = json!({
            "model": "gpt-4o",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Do work"}],
            "tools": [{"name": "do_work", "input_schema": {}}]
        });

        let result = anthropic_to_responses(input, None, false, false).unwrap();
        let parameters = &result["tools"][0]["parameters"];

        assert_eq!(parameters, &json!({"type": "object", "properties": {}}));
    }

    #[test]
    fn anthropic_to_responses_tool_arguments_are_canonical() {
        let input = json!({
            "model": "gpt-5.4",
            "messages": [{
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "tool",
                    "input": {"b": 2, "a": 1}
                }]
            }]
        });

        let result =
            anthropic_to_responses(input, None, false, false).expect("transform responses");

        assert_eq!(result["input"][0]["arguments"], "{\"a\":1,\"b\":2}");
    }

    #[test]
    fn anthropic_to_responses_maps_hosted_web_search_and_preserves_function_tools() {
        let input = json!({
            "model": "gpt-5.6-luna",
            "messages": [{"role": "user", "content": "Find the repository"}],
            "tools": [
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": 8,
                    "allowed_domains": ["github.com"],
                    "blocked_domains": [],
                    "user_location": {"type": "approximate", "country": "CN"}
                },
                {
                    "name": "Read",
                    "description": "Read a file",
                    "input_schema": {
                        "type": "object",
                        "properties": {"file_path": {"type": "string"}}
                    }
                }
            ],
            "tool_choice": {"type": "auto"}
        });

        let result = anthropic_to_responses(input.clone(), None, false, false).unwrap();

        assert_eq!(
            result["tools"][0],
            json!({
                "type": "web_search",
                "filters": {
                    "allowed_domains": ["github.com"]
                },
                "user_location": {"type": "approximate", "country": "CN"}
            })
        );
        assert_eq!(result["tools"][1]["type"], "function");
        assert_eq!(result["tools"][1]["name"], "Read");
        let includes = result["include"].as_array().unwrap();
        assert!(includes.contains(&json!(WEB_SEARCH_SOURCES_INCLUDE)));
        assert!(includes.contains(&json!("reasoning.encrypted_content")));
        assert_eq!(result["tool_choice"], "auto");
        assert!(result.get("max_tool_calls").is_none());

        let official = anthropic_to_responses(input, None, true, false).unwrap();
        assert!(official.get("max_tool_calls").is_none());
    }

    #[test]
    fn anthropic_to_responses_forces_hosted_web_search_choice() {
        let input = json!({
            "model": "gpt-5.6-luna",
            "messages": [{"role": "user", "content": "Search"}],
            "tools": [{"type": "web_search_20250305", "name": "web_search"}],
            "tool_choice": {"type": "tool", "name": "web_search"}
        });

        let result = anthropic_to_responses(input, None, false, false).unwrap();

        assert_eq!(result["tool_choice"], json!({"type": "web_search"}));
    }

    #[test]
    fn anthropic_to_responses_merges_web_search_and_codex_includes() {
        let input = json!({
            "model": "gpt-5.6-luna",
            "messages": [{"role": "user", "content": "Search"}],
            "tools": [{"type": "web_search_20250305", "name": "web_search"}]
        });

        let result = anthropic_to_responses(input, None, true, false).unwrap();
        let includes = result["include"].as_array().unwrap();

        assert!(includes.contains(&json!(WEB_SEARCH_SOURCES_INCLUDE)));
        assert!(includes.contains(&json!("reasoning.encrypted_content")));
    }

    #[test]
    fn responses_to_anthropic_maps_web_search_results_and_citations() {
        let input = json!({
            "id": "resp_search",
            "status": "completed",
            "model": "gpt-5.6-luna",
            "output": [
                {
                    "id": "ws_1",
                    "type": "web_search_call",
                    "status": "completed",
                    "action": {
                        "type": "search",
                        "query": "SaladDay cc-switch-cli",
                        "sources": [{"type": "url", "url": "https://github.com/SaladDay/cc-switch-cli"}]
                    }
                },
                {
                    "type": "message",
                    "content": [{
                        "type": "output_text",
                        "text": "GitHub result",
                        "annotations": [{
                            "type": "url_citation",
                            "start_index": 0,
                            "end_index": 6,
                            "title": "SaladDay cc-switch-cli",
                            "url": "https://github.com/SaladDay/cc-switch-cli?utm_source=openai"
                        }]
                    }]
                }
            ]
        });

        let result = responses_to_anthropic_web_search(input).unwrap();

        assert_eq!(result["content"][0]["type"], "server_tool_use");
        assert_eq!(result["content"][0]["id"], "ws_1");
        assert_eq!(
            result["content"][0]["input"],
            json!({"query": "SaladDay cc-switch-cli"})
        );
        assert_eq!(result["content"][1]["type"], "web_search_tool_result");
        assert_eq!(
            result["content"][1]["content"][0]["title"],
            "https://github.com/SaladDay/cc-switch-cli"
        );
        assert_eq!(
            result["content"][1]["content"][0]["url"],
            "https://github.com/SaladDay/cc-switch-cli"
        );
        assert_eq!(
            result["content"][2]["citations"][0]["type"],
            "web_search_result_location"
        );
        assert_eq!(result["content"][2]["citations"][0]["cited_text"], "GitHub");
        assert_eq!(
            result["content"][2]["citations"][0]["encrypted_index"],
            "ccs-cit1:0:6"
        );
        assert_eq!(result["usage"]["server_tool_use"]["web_search_requests"], 1);
        assert_eq!(result["stop_reason"], "end_turn");
    }

    #[test]
    fn anthropic_to_responses_preserves_hosted_search_history() {
        let input = json!({
            "model": "gpt-5.6-luna",
            "tools": [{"type": "web_search_20250305", "name": "web_search"}],
            "messages": [
                {"role": "user", "content": "Find the repo"},
                {"role": "assistant", "content": [
                    {
                        "type": "server_tool_use",
                        "id": "ws_1",
                        "name": "web_search",
                        "input": {"query": "SaladDay cc-switch-cli"}
                    },
                    {
                        "type": "web_search_tool_result",
                        "tool_use_id": "ws_1",
                        "content": [{
                            "type": "web_search_result",
                            "url": "https://github.com/SaladDay/cc-switch-cli",
                            "title": "SaladDay cc-switch-cli",
                            "encrypted_content": "ccs-ws1:9caa7767eea9deb1b062b630e6c79a2d2f2c3b5c9ad236bc6a1ff8b77b978b78"
                        }]
                    },
                    {
                        "type": "text",
                        "text": "GitHub result",
                        "citations": []
                    }
                ]},
                {"role": "user", "content": "Which organization owns it?"}
            ]
        });

        let result = anthropic_to_responses(input, None, false, false).unwrap();
        let items = result["input"].as_array().unwrap();
        let search_call = items
            .iter()
            .find(|item| item["type"] == "web_search_call")
            .expect("preserved web search call");
        assert_eq!(search_call["id"], "ws_1");
        assert_eq!(search_call["action"]["query"], "SaladDay cc-switch-cli");
        assert_eq!(
            search_call["action"]["sources"][0]["url"],
            "https://github.com/SaladDay/cc-switch-cli"
        );
        assert!(items
            .iter()
            .all(|item| item.pointer("/content/0/annotations").is_none()));
    }

    #[test]
    fn anthropic_citation_with_opaque_index_uses_cited_text_span() {
        let text = "The project released v5.10.1.";
        let annotations = responses_annotations_from_anthropic_text(
            &json!({
                "citations": [{
                    "type": "web_search_result_location",
                    "url": "https://example.com/release",
                    "title": "Release",
                    "encrypted_index": "anthropic-opaque-index",
                    "cited_text": "released v5.10.1"
                }]
            }),
            text,
        );
        assert_eq!(annotations[0]["start_index"], 12);
        assert_eq!(annotations[0]["end_index"], 28);
    }

    #[test]
    fn repeated_opaque_citations_consume_successive_text_spans() {
        let annotations = responses_annotations_from_anthropic_text(
            &json!({
                "citations": [
                    {
                        "type": "web_search_result_location",
                        "url": "https://example.com/first",
                        "encrypted_index": "opaque-1",
                        "cited_text": "foo"
                    },
                    {
                        "type": "web_search_result_location",
                        "url": "https://example.com/second",
                        "encrypted_index": "opaque-2",
                        "cited_text": "foo"
                    }
                ]
            }),
            "foo then foo",
        );
        assert_eq!(annotations[0]["start_index"], 0);
        assert_eq!(annotations[0]["end_index"], 3);
        assert_eq!(annotations[1]["start_index"], 9);
        assert_eq!(annotations[1]["end_index"], 12);
    }

    #[test]
    fn hosted_search_round_trip_preserves_actions_and_counts_only_search() {
        let response = json!({
            "id": "resp_actions",
            "status": "completed",
            "model": "gpt-5.6-sol",
            "output": [
                {
                    "id": "ws_search",
                    "type": "web_search_call",
                    "status": "completed",
                    "action": {
                        "type": "search",
                        "query": "release",
                        "sources": [{
                            "type": "url",
                            "url": "https://example.com/releases",
                            "title": "Releases"
                        }]
                    }
                },
                {
                    "id": "ws_open",
                    "type": "web_search_call",
                    "status": "completed",
                    "action": {
                        "type": "open_page",
                        "url": "https://example.com/releases"
                    }
                },
                {
                    "id": "ws_find",
                    "type": "web_search_call",
                    "status": "completed",
                    "action": {
                        "type": "find_in_page",
                        "url": "https://example.com/releases",
                        "pattern": "v5.10.1"
                    }
                }
            ]
        });

        let anthropic = responses_to_anthropic_web_search(response).unwrap();
        assert_eq!(
            anthropic["usage"]["server_tool_use"]["web_search_requests"],
            1
        );
        assert_eq!(anthropic["content"][1]["content"][0]["title"], "Releases");

        let history = anthropic_to_responses(
            json!({
                "model": "gpt-5.6-sol",
                "messages": [{"role": "assistant", "content": anthropic["content"]}],
                "tools": [{"type": "web_search_20250305", "name": "web_search"}]
            }),
            None,
            false,
            false,
        )
        .unwrap();
        let actions: Vec<&Value> = history["input"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|item| item["type"] == "web_search_call")
            .map(|item| &item["action"])
            .collect();

        assert_eq!(actions.len(), 3);
        assert_eq!(actions[0]["type"], "search");
        assert_eq!(actions[1]["type"], "open_page");
        assert_eq!(actions[1]["url"], "https://example.com/releases");
        assert_eq!(actions[2]["type"], "find_in_page");
        assert_eq!(actions[2]["pattern"], "v5.10.1");
    }

    #[test]
    fn hosted_search_results_use_stable_non_payload_digests() {
        let item = json!({
            "id": "ws_many",
            "type": "web_search_call",
            "action": {
                "type": "search",
                "query": "many sources",
                "sources": [
                    {"type": "url", "url": "https://example.com/1"},
                    {"type": "url", "url": "https://example.com/2"},
                    {"type": "url", "url": "https://example.com/3"}
                ]
            }
        });

        let [_, result] = render_anthropic_web_search_blocks(
            &json!({
                "id": item["id"],
                "type": item["type"],
                "status": "completed",
                "action": item["action"]
            }),
            &HashMap::new(),
        )
        .unwrap();
        let digests = result["content"].as_array().unwrap();
        assert_eq!(digests.len(), 3);
        assert!(digests.iter().all(|result| result["encrypted_content"]
            .as_str()
            .is_some_and(|value| value.starts_with("ccs-ws1:") && value.len() == 72)));
        assert_ne!(
            digests[0]["encrypted_content"],
            digests[1]["encrypted_content"]
        );
    }

    #[test]
    fn hosted_search_indexes_many_citation_titles_by_url() {
        let sources: Vec<Value> = (0..128)
            .map(|index| {
                json!({
                    "type": "url",
                    "url": format!("https://example.com/page?id={index}")
                })
            })
            .collect();
        let annotations: Vec<Value> = (0..128)
            .rev()
            .map(|index| {
                json!({
                    "type": "url_citation",
                    "url": format!("https://example.com/page?id={index}"),
                    "title": format!("Page {index}")
                })
            })
            .collect();
        let item = json!({
            "type": "web_search_call",
            "action": {"type": "search", "query": "pages", "sources": sources}
        });
        let output = vec![json!({
            "type": "message",
            "content": [{"type": "output_text", "text": "pages", "annotations": annotations}]
        })];

        let titles = citation_titles_by_exact_url(&output);
        let [_, result] = render_anthropic_web_search_blocks(
            &json!({
                "id": "ws_many_titles",
                "type": "web_search_call",
                "status": "completed",
                "action": item["action"]
            }),
            &titles,
        )
        .unwrap();
        let results = result["content"].as_array().unwrap();

        assert_eq!(results.len(), 128);
        for (index, result) in results.iter().enumerate() {
            assert_eq!(result["title"], format!("Page {index}"));
        }
    }

    #[test]
    fn encrypted_reasoning_round_trips_through_redacted_thinking() {
        let response = json!({
            "id": "resp_reasoning",
            "status": "completed",
            "output": [{
                "id": "rs_1",
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "Need a tool."}],
                "encrypted_content": "opaque-encrypted-reasoning"
            }]
        });

        let anthropic = responses_to_anthropic_web_search(response).unwrap();
        assert_eq!(anthropic["content"][0]["type"], "thinking");
        assert_eq!(
            anthropic["content"][0]["signature"],
            RESPONSES_PROXY_THINKING_SIGNATURE
        );
        assert_eq!(anthropic["content"][1]["type"], "redacted_thinking");

        let replay = anthropic_to_responses(
            json!({
                "model": "gpt-5.6-sol",
                "messages": [{"role": "assistant", "content": anthropic["content"]}],
                "tools": [{"type": "web_search_20250305", "name": "web_search"}]
            }),
            None,
            false,
            false,
        )
        .unwrap();
        let reasoning = replay["input"]
            .as_array()
            .unwrap()
            .iter()
            .find(|item| item["type"] == "reasoning")
            .expect("replayed reasoning item");
        assert_eq!(reasoning["id"], "rs_1");
        assert_eq!(reasoning["encrypted_content"], "opaque-encrypted-reasoning");
        assert_eq!(reasoning["summary"], json!([]));
    }

    #[test]
    fn responses_reasoning_content_maps_to_anthropic_thinking() {
        let response = json!({
            "id": "resp_reasoning_text",
            "status": "completed",
            "output": [{
                "id": "rs_text",
                "type": "reasoning",
                "summary": [],
                "content": [{"type": "reasoning_text", "text": "Raw reasoning."}]
            }]
        });

        let anthropic = responses_to_anthropic_web_search(response).unwrap();
        assert_eq!(anthropic["content"][0]["type"], "thinking");
        assert_eq!(anthropic["content"][0]["thinking"], "Raw reasoning.");
        assert_eq!(
            anthropic["content"][0]["signature"],
            RESPONSES_PROXY_THINKING_SIGNATURE
        );
    }

    #[test]
    fn ordinary_reasoning_response_keeps_baseline_summary_only_shape() {
        let response = json!({
            "id": "resp_ordinary_reasoning",
            "status": "completed",
            "output": [{
                "id": "rs_ordinary",
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "Visible summary."}],
                "content": [{"type": "reasoning_text", "text": "Raw reasoning."}],
                "encrypted_content": "opaque-encrypted-reasoning"
            }]
        });

        let anthropic = responses_to_anthropic(response).unwrap();
        assert_eq!(anthropic["content"].as_array().unwrap().len(), 1);
        assert_eq!(anthropic["content"][0]["type"], "thinking");
        assert_eq!(anthropic["content"][0]["thinking"], "Visible summary.");
        assert!(anthropic["content"][0].get("signature").is_none());
    }

    #[test]
    fn ordinary_disable_parallel_tool_use_keeps_baseline_wire_contract() {
        let transformed = anthropic_to_responses(
            json!({
                "model": "gpt-5.6-sol",
                "messages": [{"role": "user", "content": "use a tool"}],
                "tools": [{"name": "Bash", "input_schema": {"type": "object"}}],
                "tool_choice": {"type": "auto", "disable_parallel_tool_use": true}
            }),
            None,
            false,
            false,
        )
        .unwrap();

        assert_eq!(transformed["tool_choice"], "auto");
        assert!(transformed.get("parallel_tool_calls").is_none());
    }

    #[test]
    fn ordinary_request_ignores_unrelated_server_tool_history() {
        let transformed = anthropic_to_responses(
            json!({
                "model": "gpt-5.6-sol",
                "messages": [{
                    "role": "assistant",
                    "content": [
                        {
                            "type": "server_tool_use",
                            "id": "fetch_1",
                            "name": "web_fetch",
                            "input": {"url": "https://example.com"}
                        },
                        {"type": "text", "text": "kept"}
                    ]
                }]
            }),
            None,
            false,
            false,
        )
        .unwrap();

        assert_eq!(transformed["input"][0]["content"][0]["text"], "kept");
    }

    #[test]
    fn hosted_web_search_request_ignores_unrelated_server_tool_history() {
        let transformed = anthropic_to_responses(
            json!({
                "model": "gpt-5.6-sol",
                "tools": [{"type": "web_search_20250305", "name": "web_search"}],
                "messages": [{
                    "role": "assistant",
                    "content": [
                        {
                            "type": "server_tool_use",
                            "id": "fetch_1",
                            "name": "web_fetch",
                            "input": {"url": "https://example.com"}
                        },
                        {"type": "text", "text": "kept"}
                    ]
                }]
            }),
            None,
            false,
            false,
        )
        .unwrap();

        assert_eq!(transformed["input"][0]["content"][0]["text"], "kept");
    }

    #[test]
    fn ordinary_response_does_not_enable_web_search_citations() {
        let transformed = responses_to_anthropic(json!({
            "id": "resp_ordinary",
            "status": "completed",
            "output": [{
                "type": "message",
                "content": [{
                    "type": "output_text",
                    "text": "plain",
                    "annotations": [{
                        "type": "url_citation",
                        "url": "https://example.com",
                        "title": "Example",
                        "start_index": 99,
                        "end_index": 100
                    }]
                }]
            }]
        }))
        .unwrap();

        assert_eq!(
            transformed["content"][0],
            json!({"type": "text", "text": "plain"})
        );
        assert!(transformed["usage"].get("server_tool_use").is_none());
    }

    #[test]
    fn ordinary_request_does_not_replay_assistant_citations() {
        let transformed = anthropic_to_responses(
            json!({
                "model": "gpt-5.6-sol",
                "messages": [{
                    "role": "assistant",
                    "content": [{
                        "type": "text",
                        "text": "plain",
                        "citations": [{
                            "type": "web_search_result_location",
                            "url": "https://example.com",
                            "title": "Example",
                            "cited_text": "plain",
                            "encrypted_index": "opaque"
                        }]
                    }]
                }]
            }),
            None,
            false,
            false,
        )
        .unwrap();

        assert!(transformed["input"][0]["content"][0]
            .get("annotations")
            .is_none());
    }

    #[test]
    fn hosted_web_search_disable_parallel_tool_use_maps_to_responses() {
        let transformed = anthropic_to_responses(
            json!({
                "model": "gpt-5.6-sol",
                "messages": [{"role": "user", "content": "search"}],
                "tools": [{
                    "type": "web_search_20250305",
                    "name": "web_search"
                }],
                "tool_choice": {"type": "auto", "disable_parallel_tool_use": true}
            }),
            None,
            false,
            false,
        )
        .unwrap();

        assert_eq!(transformed["tool_choice"], "auto");
        assert_eq!(transformed["parallel_tool_calls"], false);
    }

    #[test]
    fn hosted_search_without_sources_round_trips_supported_action_fields() {
        let response = json!({
            "id": "resp_empty_search",
            "status": "completed",
            "output": [{
                "id": "ws_empty",
                "type": "web_search_call",
                "status": "completed",
                "action": {
                    "type": "search",
                    "query": "a OR b",
                    "queries": ["a", "b"],
                    "sources": []
                }
            }]
        });

        let anthropic = responses_to_anthropic_web_search(response).unwrap();
        assert_eq!(anthropic["content"][1]["content"], json!([]));
        let replay = anthropic_to_responses(
            json!({
                "model": "gpt-5.6-sol",
                "tools": [{"type": "web_search_20250305", "name": "web_search"}],
                "messages": [{"role": "assistant", "content": anthropic["content"]}]
            }),
            None,
            false,
            false,
        )
        .unwrap();
        let action = &replay["input"][0]["action"];
        assert_eq!(action["type"], "search");
        assert!(action.get("queries").is_none());
        assert_eq!(action["query"], "a OR b");
        assert_eq!(action["sources"], json!([]));
    }

    #[test]
    fn failed_hosted_search_fails_the_response_conversion() {
        let result = responses_to_anthropic_web_search(json!({
            "id": "resp_failed_search",
            "status": "completed",
            "output": [{
                "id": "ws_failed",
                "type": "web_search_call",
                "status": "failed",
                "error": {"code": "too_many_requests"},
                "action": {"type": "search", "query": "q"}
            }]
        }));
        assert!(matches!(result, Err(ProxyError::TransformError(_))));
    }
}
