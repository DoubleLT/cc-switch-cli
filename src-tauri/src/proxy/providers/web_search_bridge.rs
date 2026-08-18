use bytes::Bytes;
use futures::{Stream, StreamExt};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::proxy::{
    error::ProxyError,
    json_canonical::canonical_json_string,
    response::StreamCompletion,
    sse::{append_utf8_safe, strip_sse_field, take_sse_block},
};

pub(crate) const WEB_SEARCH_SOURCES_INCLUDE: &str = "web_search_call.action.sources";
const WEB_SEARCH_TOOL_TYPE: &str = "web_search_20250305";
const WEB_SEARCH_TOOL_NAME: &str = "web_search";
const RESULT_DIGEST_PREFIX: &str = "ccs-ws1:";
const CITATION_INDEX_PREFIX: &str = "ccs-cit1:";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WebSearchRequestPolicy {
    max_uses: Option<u64>,
}

impl WebSearchRequestPolicy {
    pub fn from_anthropic_request(body: &Value) -> Result<Option<Self>, ProxyError> {
        let mut policy = None;
        for tool in body
            .get("tools")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let tool_type = tool.get("type").and_then(Value::as_str);
            let looks_like_web_search = tool_type
                .is_some_and(|value| value == "web_search" || value.starts_with("web_search_"));
            if !looks_like_web_search {
                continue;
            }
            map_anthropic_web_search_tool(tool)?;
            if policy.is_some() {
                return Err(invalid("multiple hosted WebSearch tools are not supported"));
            }
            let max_uses = match tool.get("max_uses") {
                None => None,
                Some(value) => Some(
                    value
                        .as_u64()
                        .filter(|value| *value > 0)
                        .ok_or_else(|| invalid("WebSearch max_uses must be a positive integer"))?,
                ),
            };
            policy = Some(Self { max_uses });
        }
        Ok(policy)
    }

    fn permits_search_count(&self, count: u64) -> bool {
        self.max_uses.is_none_or(|limit| count <= limit)
    }

    pub(crate) fn validate_buffered_response(&self, body: &Value) -> Result<(), ProxyError> {
        let output = body
            .get("output")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                ProxyError::TransformError("WebSearch response output must be an array".to_string())
            })?;
        validate_web_search_output(output, self).map_err(ProxyError::TransformError)?;
        Ok(())
    }
}

pub(crate) fn map_anthropic_web_search_tool(tool: &Value) -> Result<Value, ProxyError> {
    validate_allowed_block_fields(
        tool,
        &[
            "type",
            "name",
            "max_uses",
            "allowed_domains",
            "blocked_domains",
            "user_location",
            "cache_control",
        ],
        "WebSearch tool",
    )?;
    validate_ephemeral_cache_control(tool.get("cache_control"))?;
    if tool.get("type").and_then(Value::as_str) != Some(WEB_SEARCH_TOOL_TYPE)
        || tool.get("name").and_then(Value::as_str) != Some(WEB_SEARCH_TOOL_NAME)
    {
        return Err(invalid(
            "only type=web_search_20250305,name=web_search is supported",
        ));
    }

    if let Some(domains) = tool.get("blocked_domains") {
        let domains = string_array(domains, "WebSearch blocked_domains")?;
        if !domains.is_empty() {
            return Err(invalid(
                "non-empty WebSearch blocked_domains is not supported",
            ));
        }
    }

    let mut mapped = json!({"type": "web_search"});
    if let Some(domains) = tool.get("allowed_domains") {
        let domains = string_array(domains, "WebSearch allowed_domains")?;
        if !domains.is_empty() {
            mapped["filters"] = json!({"allowed_domains": domains});
        }
    }
    if let Some(location) = tool.get("user_location") {
        mapped["user_location"] = map_user_location(location)?;
    }
    Ok(mapped)
}

fn map_user_location(location: &Value) -> Result<Value, ProxyError> {
    let object = location
        .as_object()
        .ok_or_else(|| invalid("WebSearch user_location must be an object"))?;
    reject_unknown_fields(
        object,
        &["type", "city", "country", "region", "timezone"],
        "WebSearch user_location",
    )?;
    if object.get("type").and_then(Value::as_str) != Some("approximate") {
        return Err(invalid("WebSearch user_location.type must be approximate"));
    }
    for field in ["city", "country", "region", "timezone"] {
        if object
            .get(field)
            .is_some_and(|value| !value.is_null() && !value.is_string())
        {
            return Err(invalid(format!(
                "WebSearch user_location.{field} must be a string or null"
            )));
        }
    }
    Ok(location.clone())
}

pub(crate) fn fold_anthropic_web_search_history(
    blocks: &[Value],
    index: usize,
) -> Result<Option<(Value, usize)>, ProxyError> {
    let Some(server) = blocks.get(index) else {
        return Ok(None);
    };
    if server.get("type").and_then(Value::as_str) != Some("server_tool_use") {
        return Ok(None);
    }
    if server.get("name").and_then(Value::as_str) != Some(WEB_SEARCH_TOOL_NAME) {
        return Err(invalid("unsupported server_tool_use block"));
    }
    validate_allowed_block_fields(
        server,
        &["type", "id", "name", "input", "cache_control"],
        "server_tool_use",
    )?;
    validate_ephemeral_cache_control(server.get("cache_control"))?;

    let id = non_empty_string(server.get("id"), "server_tool_use.id")?;
    let input = server
        .get("input")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid("server_tool_use.input must be an object"))?;
    let result = blocks
        .get(index + 1)
        .ok_or_else(|| invalid("server_tool_use must be followed by web_search_tool_result"))?;
    if result.get("type").and_then(Value::as_str) != Some("web_search_tool_result") {
        return Err(invalid(
            "server_tool_use must be followed by web_search_tool_result",
        ));
    }
    validate_allowed_block_fields(
        result,
        &["type", "tool_use_id", "content", "cache_control"],
        "web_search_tool_result",
    )?;
    validate_ephemeral_cache_control(result.get("cache_control"))?;
    if result.get("tool_use_id").and_then(Value::as_str) != Some(id) {
        return Err(invalid("WebSearch history pair IDs do not match"));
    }

    let content = result
        .get("content")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("web_search_tool_result.content must be an array"))?;
    let action = fold_action(id, input, content)?;
    Ok(Some((
        json!({
            "type": "web_search_call",
            "id": id,
            "status": "completed",
            "action": action
        }),
        index + 2,
    )))
}

fn fold_action(
    item_id: &str,
    input: &Map<String, Value>,
    content: &[Value],
) -> Result<Value, ProxyError> {
    let action_type = input
        .get("action")
        .and_then(Value::as_str)
        .unwrap_or("search");
    match action_type {
        "search" => {
            reject_unknown_fields(input, &["query"], "WebSearch search input")?;
            let query = non_empty_string(input.get("query"), "WebSearch search query")?;
            let sources = validate_result_content(content, item_id)?
                .into_iter()
                .map(|url| json!({"type": "url", "url": url}))
                .collect::<Vec<_>>();
            Ok(json!({"type": "search", "query": query, "sources": sources}))
        }
        "open_page" => {
            reject_unknown_fields(
                input,
                &["query", "action", "url"],
                "WebSearch open_page input",
            )?;
            let url = non_empty_string(input.get("url"), "WebSearch open_page url")?;
            if input.get("query").and_then(Value::as_str) != Some(url) {
                return Err(invalid("WebSearch open_page query must equal url"));
            }
            let urls = validate_result_content(content, item_id)?;
            if urls.as_slice() != [url] {
                return Err(invalid(
                    "WebSearch open_page result must contain its URL once",
                ));
            }
            Ok(json!({"type": "open_page", "url": url}))
        }
        "find_in_page" => {
            reject_unknown_fields(
                input,
                &["query", "action", "url", "pattern"],
                "WebSearch find_in_page input",
            )?;
            let url = non_empty_string(input.get("url"), "WebSearch find_in_page url")?;
            let pattern = non_empty_string(input.get("pattern"), "WebSearch find_in_page pattern")?;
            let expected_query = format!("{pattern} ({url})");
            if input.get("query").and_then(Value::as_str) != Some(expected_query.as_str()) {
                return Err(invalid(
                    "WebSearch find_in_page query is not in canonical form",
                ));
            }
            let urls = validate_result_content(content, item_id)?;
            if urls.as_slice() != [url] {
                return Err(invalid(
                    "WebSearch find_in_page result must contain its URL once",
                ));
            }
            Ok(json!({"type": "find_in_page", "url": url, "pattern": pattern}))
        }
        _ => Err(invalid("unsupported WebSearch history action")),
    }
}

fn validate_result_content<'a>(
    content: &'a [Value],
    item_id: &str,
) -> Result<Vec<&'a str>, ProxyError> {
    let mut urls = Vec::with_capacity(content.len());
    for (ordinal, result) in content.iter().enumerate() {
        validate_allowed_block_fields(
            result,
            &["type", "url", "title", "encrypted_content"],
            "web_search_result",
        )?;
        if result.get("type").and_then(Value::as_str) != Some("web_search_result") {
            return Err(invalid("unsupported web_search_tool_result content"));
        }
        let url = non_empty_string(result.get("url"), "web_search_result.url")?;
        non_empty_string(result.get("title"), "web_search_result.title")?;
        let encrypted = non_empty_string(
            result.get("encrypted_content"),
            "web_search_result.encrypted_content",
        )?;
        if encrypted != result_digest(item_id, ordinal, url) {
            return Err(invalid(
                "web_search_result.encrypted_content does not match its canonical result",
            ));
        }
        urls.push(url);
    }
    Ok(urls)
}

#[derive(Debug)]
pub struct WebSearchBufferedResponse {
    added_indices: BTreeSet<u64>,
    done_items: BTreeMap<u64, Value>,
    terminal: Option<Value>,
}

impl WebSearchBufferedResponse {
    pub fn new() -> Self {
        Self {
            added_indices: BTreeSet::new(),
            done_items: BTreeMap::new(),
            terminal: None,
        }
    }

    pub fn ingest_responses_event(&mut self, event_name: &str, data: &Value) -> Result<(), String> {
        if self.terminal.is_some() {
            return Err("received a Responses event after response.completed".to_string());
        }
        match event_name {
            "response.output_item.added" => {
                let index = event_output_index(data)?;
                self.added_indices.insert(index);
            }
            "response.output_item.done" => {
                let index = event_output_index(data)?;
                let item = data
                    .get("item")
                    .cloned()
                    .ok_or_else(|| "response.output_item.done is missing item".to_string())?;
                non_empty_event_string(item.get("id"), "done item id")?;
                non_empty_event_string(item.get("type"), "done item type")?;
                if let Some(previous) = self.done_items.get(&index) {
                    if previous != &item {
                        return Err(format!("output_index {index} has conflicting done items"));
                    }
                } else {
                    self.done_items.insert(index, item);
                }
            }
            "response.completed" => {
                let mut response = data.get("response").unwrap_or(data).clone();
                if response.get("status").and_then(Value::as_str) != Some("completed") {
                    return Err("response.completed did not carry completed status".to_string());
                }
                if response.get("error").is_some_and(|error| !error.is_null()) {
                    return Err("response.completed carried a non-null error".to_string());
                }
                if let Some(output) = response.get("output").and_then(Value::as_array) {
                    self.added_indices.extend(0..output.len() as u64);
                }
                // output_item.done is the authority. Avoid retaining a duplicate
                // terminal output snapshot while waiting for transport EOF.
                response["output"] = json!([]);
                self.terminal = Some(response);
            }
            "response.failed" | "response.cancelled" | "response.incomplete" => {
                return Err(format!("upstream ended with {event_name}"));
            }
            "error" | "response.error" => {
                return Err("upstream emitted a Responses error event".to_string());
            }
            _ => {}
        }
        Ok(())
    }

    pub(crate) fn seal_responses(self, policy: &WebSearchRequestPolicy) -> Result<Value, String> {
        let mut terminal = self
            .terminal
            .ok_or_else(|| "WebSearch stream ended without response.completed".to_string())?;
        let missing: Vec<u64> = self
            .added_indices
            .difference(&self.done_items.keys().copied().collect())
            .copied()
            .collect();
        if !missing.is_empty() {
            return Err(format!(
                "WebSearch stream ended without output_item.done for indices {missing:?}"
            ));
        }

        let output: Vec<Value> = self.done_items.into_values().collect();
        validate_web_search_output(&output, policy)?;
        terminal["output"] = Value::Array(output);
        Ok(terminal)
    }

    pub fn seal(self, policy: &WebSearchRequestPolicy) -> Result<Value, String> {
        let terminal = self.seal_responses(policy)?;
        super::transform_responses::responses_to_anthropic_web_search(terminal)
            .map_err(|error| error.to_string())
    }
}

fn validate_web_search_output(
    output: &[Value],
    policy: &WebSearchRequestPolicy,
) -> Result<(), String> {
    let searches = output
        .iter()
        .filter(|item| {
            item.get("type").and_then(Value::as_str) == Some("web_search_call")
                && item.pointer("/action/type").and_then(Value::as_str) == Some("search")
                && item.get("status").and_then(Value::as_str) == Some("completed")
        })
        .count() as u64;
    if !policy.permits_search_count(searches) {
        return Err("WebSearch max_uses exceeded".to_string());
    }
    for item in output
        .iter()
        .filter(|item| item.get("type").and_then(Value::as_str) == Some("web_search_call"))
    {
        if item.get("status").and_then(Value::as_str) != Some("completed") {
            return Err("WebSearch item did not complete successfully".to_string());
        }
    }
    Ok(())
}

fn event_output_index(data: &Value) -> Result<u64, String> {
    data.get("output_index")
        .and_then(Value::as_u64)
        .ok_or_else(|| "Responses item event is missing output_index".to_string())
}

pub(crate) fn render_anthropic_web_search_blocks(
    item: &Value,
    titles: &HashMap<String, String>,
) -> Result<[Value; 2], ProxyError> {
    if item.get("status").and_then(Value::as_str) != Some("completed") {
        return Err(ProxyError::TransformError(
            "WebSearch item did not complete successfully".to_string(),
        ));
    }
    let id = non_empty_string(item.get("id"), "WebSearch item id")?;
    let action = item
        .get("action")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid("WebSearch item action must be an object"))?;
    let (input, source_urls): (Value, Vec<(&str, Option<&str>)>) =
        match action.get("type").and_then(Value::as_str) {
            Some("search") => {
                let query = non_empty_string(action.get("query"), "WebSearch search query")?;
                let sources = match action.get("sources") {
                    None => Vec::new(),
                    Some(value) => value
                        .as_array()
                        .ok_or_else(|| invalid("WebSearch search sources must be an array"))?
                        .iter()
                        .map(|source| {
                            if source.get("type").and_then(Value::as_str) != Some("url") {
                                return Err(invalid("WebSearch source type must be url"));
                            }
                            let url = non_empty_string(source.get("url"), "WebSearch source url")?;
                            let title = source
                                .get("title")
                                .and_then(Value::as_str)
                                .filter(|title| !title.trim().is_empty());
                            Ok((url, title))
                        })
                        .collect::<Result<Vec<_>, ProxyError>>()?,
                };
                (json!({"query": query}), sources)
            }
            Some("open_page") => {
                let url = non_empty_string(action.get("url"), "WebSearch open_page url")?;
                (
                    json!({"query": url, "action": "open_page", "url": url}),
                    vec![(url, None)],
                )
            }
            Some("find_in_page") => {
                let url = non_empty_string(action.get("url"), "WebSearch find_in_page url")?;
                let pattern =
                    non_empty_string(action.get("pattern"), "WebSearch find_in_page pattern")?;
                (
                    json!({
                        "query": format!("{pattern} ({url})"),
                        "action": "find_in_page",
                        "url": url,
                        "pattern": pattern
                    }),
                    vec![(url, None)],
                )
            }
            _ => return Err(invalid("unsupported WebSearch action")),
        };

    let results = source_urls
        .into_iter()
        .enumerate()
        .map(|(ordinal, (url, source_title))| {
            let title = source_title
                .map(str::to_string)
                .or_else(|| titles.get(url).cloned())
                .unwrap_or_else(|| url.to_string());
            json!({
                "type": "web_search_result",
                "url": url,
                "title": title,
                "encrypted_content": result_digest(id, ordinal, url)
            })
        })
        .collect::<Vec<_>>();

    Ok([
        json!({
            "type": "server_tool_use",
            "id": id,
            "name": WEB_SEARCH_TOOL_NAME,
            "input": input
        }),
        json!({
            "type": "web_search_tool_result",
            "tool_use_id": id,
            "content": results
        }),
    ])
}

pub(crate) fn citation_titles_by_exact_url(output: &[Value]) -> HashMap<String, String> {
    let mut titles = HashMap::new();
    for item in output {
        for part in item
            .get("content")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            for annotation in part
                .get("annotations")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                if annotation.get("type").and_then(Value::as_str) != Some("url_citation") {
                    continue;
                }
                let Some(url) = annotation.get("url").and_then(Value::as_str) else {
                    continue;
                };
                let Some(title) = annotation
                    .get("title")
                    .and_then(Value::as_str)
                    .filter(|title| !title.trim().is_empty())
                else {
                    continue;
                };
                titles
                    .entry(url.to_string())
                    .or_insert_with(|| title.to_string());
            }
        }
    }
    titles
}

pub(crate) fn render_anthropic_citation(
    annotation: &Value,
    text: &str,
    scalar_boundaries: &[usize],
) -> Result<Option<Value>, ProxyError> {
    if annotation.get("type").and_then(Value::as_str) != Some("url_citation") {
        return Ok(None);
    }
    let url = non_empty_string(annotation.get("url"), "citation url")?;
    let title = annotation
        .get("title")
        .and_then(Value::as_str)
        .filter(|title| !title.trim().is_empty())
        .unwrap_or(url);
    let start = annotation
        .get("start_index")
        .and_then(Value::as_u64)
        .ok_or_else(|| invalid("citation start_index must be a non-negative integer"))?;
    let end = annotation
        .get("end_index")
        .and_then(Value::as_u64)
        .ok_or_else(|| invalid("citation end_index must be a non-negative integer"))?;
    let start_usize =
        usize::try_from(start).map_err(|_| invalid("citation start_index overflow"))?;
    let end_usize = usize::try_from(end).map_err(|_| invalid("citation end_index overflow"))?;
    if start_usize > end_usize || end_usize >= scalar_boundaries.len() {
        return Err(invalid("citation offsets are outside the owning text"));
    }
    let cited_text = &text[scalar_boundaries[start_usize]..scalar_boundaries[end_usize]];
    Ok(Some(json!({
        "type": "web_search_result_location",
        "url": url,
        "title": title,
        "cited_text": cited_text,
        "encrypted_index": format!("{CITATION_INDEX_PREFIX}{start}:{end}")
    })))
}

pub(crate) fn unicode_scalar_boundaries(text: &str) -> Vec<usize> {
    text.char_indices()
        .map(|(byte_index, _)| byte_index)
        .chain(std::iter::once(text.len()))
        .collect()
}

fn result_digest(item_id: &str, ordinal: usize, url: &str) -> String {
    let mut digest = Sha256::new();
    digest.update(item_id.as_bytes());
    digest.update([0]);
    digest.update(ordinal.to_string().as_bytes());
    digest.update([0]);
    digest.update(url.as_bytes());
    format!("{RESULT_DIGEST_PREFIX}{:x}", digest.finalize())
}

pub fn create_buffered_anthropic_sse_stream(
    stream: impl Stream<Item = Result<Bytes, std::io::Error>> + Send + 'static,
    completion: StreamCompletion,
    policy: WebSearchRequestPolicy,
) -> impl Stream<Item = Result<Bytes, std::io::Error>> + Send {
    async_stream::stream! {
        let mut accumulator = WebSearchBufferedResponse::new();
        let mut buffer = String::new();
        let mut utf8_remainder = Vec::new();
        let mut failure = None;
        tokio::pin!(stream);

        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(bytes) => append_utf8_safe(&mut buffer, &mut utf8_remainder, &bytes),
                Err(error) => {
                    failure = Some(error.to_string());
                    break;
                }
            }
            if failure.is_some() {
                buffer.clear();
                utf8_remainder.clear();
                continue;
            }
            while let Some(block) = take_sse_block(&mut buffer) {
                if block.trim().is_empty() {
                    continue;
                }
                match parse_responses_sse_block(&block) {
                    Ok(None) => {}
                    Ok(Some((event, data))) => {
                        if let Err(error) = accumulator.ingest_responses_event(&event, &data) {
                            failure = Some(error);
                            buffer.clear();
                            break;
                        }
                    }
                    Err(error) => {
                        failure = Some(error);
                        buffer.clear();
                        break;
                    }
                }
            }
        }

        if failure.is_none() && (!utf8_remainder.is_empty() || !buffer.trim().is_empty()) {
            failure = Some("WebSearch stream ended with an incomplete SSE frame".to_string());
        }
        let message = if let Some(error) = failure {
            Err(error)
        } else {
            accumulator.seal(&policy)
        };
        match message {
            Ok(message) => {
                for event in anthropic_message_sse(&message) {
                    yield Ok(event);
                }
                completion.record_success();
            }
            Err(error) => {
                completion.record_error(error.clone());
                yield Ok(anthropic_sse("error", &json!({
                    "type": "error",
                    "error": {"type": "upstream_error", "message": bounded_message(&error)}
                })));
            }
        }
    }
}

fn parse_responses_sse_block(block: &str) -> Result<Option<(String, Value)>, String> {
    let mut event = None;
    let mut data = Vec::new();
    for line in block.lines() {
        if let Some(value) = strip_sse_field(line, "event") {
            event = Some(value.trim().to_string());
        } else if let Some(value) = strip_sse_field(line, "data") {
            data.push(value.to_string());
        }
    }
    if data.is_empty() {
        return Ok(None);
    }
    let data = data.join("\n");
    if data.trim() == "[DONE]" {
        return Ok(None);
    }
    let value: Value = serde_json::from_str(&data)
        .map_err(|error| format!("invalid Responses SSE JSON: {error}"))?;
    let event = event
        .filter(|event| !event.is_empty())
        .or_else(|| {
            value
                .get("type")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .ok_or_else(|| "Responses SSE event is missing its type".to_string())?;
    Ok(Some((event, value)))
}

fn anthropic_message_sse(message: &Value) -> Vec<Bytes> {
    let usage = message.get("usage").cloned().unwrap_or_else(|| {
        json!({
            "input_tokens": 0,
            "output_tokens": 0
        })
    });
    let mut events = vec![anthropic_sse(
        "message_start",
        &json!({
            "type": "message_start",
            "message": {
                "id": message.get("id").and_then(Value::as_str).unwrap_or("msg_responses_proxy"),
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": message.get("model").and_then(Value::as_str).unwrap_or("unknown"),
                "stop_reason": null,
                "stop_sequence": null,
                "usage": usage
            }
        }),
    )];

    for (index, block) in message
        .get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
    {
        let index = index as u64;
        let block_type = block.get("type").and_then(Value::as_str).unwrap_or("");
        let start = match block_type {
            "text" => {
                let mut start = json!({"type": "text", "text": ""});
                if block.get("citations").is_some() {
                    start["citations"] = json!([]);
                }
                start
            }
            "thinking" => json!({"type": "thinking", "thinking": "", "signature": ""}),
            "tool_use" | "server_tool_use" => json!({
                "type": block_type,
                "id": block.get("id"),
                "name": block.get("name"),
                "input": {}
            }),
            _ => block.clone(),
        };
        events.push(anthropic_sse(
            "content_block_start",
            &json!({
                "type": "content_block_start", "index": index, "content_block": start
            }),
        ));
        match block_type {
            "text" => {
                if let Some(text) = block
                    .get("text")
                    .and_then(Value::as_str)
                    .filter(|v| !v.is_empty())
                {
                    events.push(anthropic_sse(
                        "content_block_delta",
                        &json!({
                            "type": "content_block_delta", "index": index,
                            "delta": {"type": "text_delta", "text": text}
                        }),
                    ));
                }
                for citation in block
                    .get("citations")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    events.push(anthropic_sse(
                        "content_block_delta",
                        &json!({
                            "type": "content_block_delta", "index": index,
                            "delta": {"type": "citations_delta", "citation": citation}
                        }),
                    ));
                }
            }
            "thinking" => {
                if let Some(thinking) = block
                    .get("thinking")
                    .and_then(Value::as_str)
                    .filter(|v| !v.is_empty())
                {
                    events.push(anthropic_sse(
                        "content_block_delta",
                        &json!({
                            "type": "content_block_delta", "index": index,
                            "delta": {"type": "thinking_delta", "thinking": thinking}
                        }),
                    ));
                }
                if let Some(signature) = block
                    .get("signature")
                    .and_then(Value::as_str)
                    .filter(|v| !v.is_empty())
                {
                    events.push(anthropic_sse(
                        "content_block_delta",
                        &json!({
                            "type": "content_block_delta", "index": index,
                            "delta": {"type": "signature_delta", "signature": signature}
                        }),
                    ));
                }
            }
            "tool_use" | "server_tool_use" => {
                let input = block.get("input").cloned().unwrap_or_else(|| json!({}));
                events.push(anthropic_sse("content_block_delta", &json!({
                    "type": "content_block_delta", "index": index,
                    "delta": {"type": "input_json_delta", "partial_json": canonical_json_string(&input)}
                })));
            }
            _ => {}
        }
        events.push(anthropic_sse(
            "content_block_stop",
            &json!({
                "type": "content_block_stop", "index": index
            }),
        ));
    }

    events.push(anthropic_sse(
        "message_delta",
        &json!({
            "type": "message_delta",
            "delta": {
                "stop_reason": message.get("stop_reason").cloned().unwrap_or(Value::Null),
                "stop_sequence": message.get("stop_sequence").cloned().unwrap_or(Value::Null)
            },
            "usage": usage
        }),
    ));
    events.push(anthropic_sse(
        "message_stop",
        &json!({"type": "message_stop"}),
    ));
    events
}

fn anthropic_sse(event: &str, payload: &Value) -> Bytes {
    Bytes::from(format!(
        "event: {event}\ndata: {}\n\n",
        serde_json::to_string(payload).unwrap_or_default()
    ))
}

fn bounded_message(message: &str) -> String {
    let normalized = message.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.chars().count() <= 240 {
        normalized
    } else {
        format!("{}...", normalized.chars().take(240).collect::<String>())
    }
}

fn string_array(value: &Value, name: &str) -> Result<Vec<String>, ProxyError> {
    value
        .as_array()
        .ok_or_else(|| invalid(format!("{name} must be an array")))?
        .iter()
        .map(|value| {
            value
                .as_str()
                .filter(|value| !value.trim().is_empty())
                .map(str::to_string)
                .ok_or_else(|| invalid(format!("{name} entries must be non-empty strings")))
        })
        .collect()
}

fn validate_allowed_block_fields(
    value: &Value,
    allowed: &[&str],
    name: &str,
) -> Result<(), ProxyError> {
    let object = value
        .as_object()
        .ok_or_else(|| invalid(format!("{name} must be an object")))?;
    reject_unknown_fields(object, allowed, name)
}

fn reject_unknown_fields(
    object: &Map<String, Value>,
    allowed: &[&str],
    name: &str,
) -> Result<(), ProxyError> {
    if let Some(field) = object
        .keys()
        .find(|field| !allowed.contains(&field.as_str()))
    {
        return Err(invalid(format!(
            "{name} contains unsupported field {field}"
        )));
    }
    Ok(())
}

fn validate_ephemeral_cache_control(value: Option<&Value>) -> Result<(), ProxyError> {
    let Some(value) = value else {
        return Ok(());
    };
    if value != &json!({"type": "ephemeral"}) {
        return Err(invalid("only cache_control={type:ephemeral} is supported"));
    }
    Ok(())
}

fn non_empty_string<'a>(value: Option<&'a Value>, name: &str) -> Result<&'a str, ProxyError> {
    value
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| invalid(format!("{name} must be a non-empty string")))
}

fn non_empty_event_string<'a>(value: Option<&'a Value>, name: &str) -> Result<&'a str, String> {
    value
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| format!("{name} must be a non-empty string"))
}

fn invalid(message: impl Into<String>) -> ProxyError {
    ProxyError::InvalidRequest(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::{stream, StreamExt};

    fn policy(max_uses: u64) -> WebSearchRequestPolicy {
        WebSearchRequestPolicy {
            max_uses: Some(max_uses),
        }
    }

    async fn collect_buffered(
        input: String,
        policy: WebSearchRequestPolicy,
    ) -> (String, StreamCompletion) {
        let completion = StreamCompletion::default();
        let converted = create_buffered_anthropic_sse_stream(
            stream::iter(vec![Ok::<_, std::io::Error>(Bytes::from(input))]),
            completion.clone(),
            policy,
        );
        let output = converted
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .map(|chunk| String::from_utf8_lossy(&chunk.unwrap()).into_owned())
            .collect();
        (output, completion)
    }

    fn anthropic_payloads(output: &str) -> Vec<Value> {
        output
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter_map(|data| serde_json::from_str(data).ok())
            .collect()
    }

    #[test]
    fn request_mapping_preserves_supported_fields_and_omits_max_tool_calls() {
        let request = json!({
            "tools": [{
                "type": "web_search_20250305",
                "name": "web_search",
                "allowed_domains": ["example.com"],
                "blocked_domains": [],
                "max_uses": 2,
                "user_location": {"type": "approximate", "country": "CN"}
            }]
        });
        let policy = WebSearchRequestPolicy::from_anthropic_request(&request)
            .unwrap()
            .unwrap();
        assert!(policy.permits_search_count(2));
        assert!(!policy.permits_search_count(3));
        assert_eq!(
            map_anthropic_web_search_tool(&request["tools"][0]).unwrap(),
            json!({
                "type": "web_search",
                "filters": {"allowed_domains": ["example.com"]},
                "user_location": {"type": "approximate", "country": "CN"}
            })
        );
    }

    #[test]
    fn request_mapping_rejects_unknown_version_and_blocked_domains() {
        for tool in [
            json!({"type":"web_search_20990101","name":"web_search"}),
            json!({
                "type":"web_search_20250305","name":"web_search",
                "blocked_domains":["example.com"]
            }),
            json!({
                "type":"web_search_20250305","name":"web_search",
                "blocked_domains":"example.com"
            }),
            json!({
                "type":"web_search_20250305","name":"web_search",
                "future_restrictive_field": true
            }),
        ] {
            assert!(map_anthropic_web_search_tool(&tool).is_err());
        }
    }

    #[test]
    fn canonical_history_pair_folds_without_titles_or_encrypted_payload() {
        let blocks = vec![
            json!({
                "type":"server_tool_use","id":"ws_1","name":"web_search",
                "input":{"query":"rust"}
            }),
            json!({
                "type":"web_search_tool_result","tool_use_id":"ws_1",
                "content":[{
                    "type":"web_search_result","url":"https://example.com",
                    "title":"Example",
                    "encrypted_content":result_digest("ws_1", 0, "https://example.com")
                }]
            }),
        ];
        let (item, next) = fold_anthropic_web_search_history(&blocks, 0)
            .unwrap()
            .unwrap();
        assert_eq!(next, 2);
        assert_eq!(
            item,
            json!({
                "type":"web_search_call","id":"ws_1","status":"completed",
                "action":{
                    "type":"search","query":"rust",
                    "sources":[{"type":"url","url":"https://example.com"}]
                }
            })
        );

        let mut tampered = blocks;
        tampered[1]["content"][0]["url"] = json!("https://example.com/tampered");
        assert!(fold_anthropic_web_search_history(&tampered, 0).is_err());
    }

    #[test]
    fn unicode_citation_uses_scalar_offsets_and_exact_slice() {
        let text = "A😊中B";
        let citation = render_anthropic_citation(
            &json!({
                "type":"url_citation","url":"https://example.com","title":"Example",
                "start_index":1,"end_index":3
            }),
            text,
            &unicode_scalar_boundaries(text),
        )
        .unwrap()
        .unwrap();
        assert_eq!(citation["cited_text"], "😊中");
        assert_eq!(citation["encrypted_index"], "ccs-cit1:1:3");
    }

    #[test]
    fn result_digest_is_stable_and_contains_no_source_payload() {
        let digest = result_digest("ws_1", 0, "https://example.com");
        assert_eq!(digest, result_digest("ws_1", 0, "https://example.com"));
        assert_eq!(digest.len(), RESULT_DIGEST_PREFIX.len() + 64);
        assert!(digest
            .strip_prefix(RESULT_DIGEST_PREFIX)
            .unwrap()
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)));
        assert!(!digest.contains("example.com"));
    }

    #[test]
    fn ordinary_request_does_not_enable_buffered_mode() {
        let request = json!({
            "tools": [
                {"name":"Bash","input_schema":{"type":"object"}},
                {"name":"web_search","input_schema":{"type":"object"}}
            ]
        });
        assert_eq!(
            WebSearchRequestPolicy::from_anthropic_request(&request).unwrap(),
            None
        );
    }

    #[tokio::test]
    async fn buffered_mode_renders_search_open_find_late_title_and_unicode_citation() {
        let input = concat!(
            "event: response.output_item.done\n",
            "data: {\"type\":\"response.output_item.done\",\"output_index\":1,\"item\":{\"id\":\"ws_search\",\"type\":\"web_search_call\",\"status\":\"completed\",\"action\":{\"type\":\"search\",\"query\":\"rust\",\"sources\":[{\"type\":\"url\",\"url\":\"https://example.com\"}]}}}\n\n",
            "event: response.output_item.done\n",
            "data: {\"type\":\"response.output_item.done\",\"output_index\":3,\"item\":{\"id\":\"ws_open\",\"type\":\"web_search_call\",\"status\":\"completed\",\"action\":{\"type\":\"open_page\",\"url\":\"https://example.com\"}}}\n\n",
            "event: response.output_item.done\n",
            "data: {\"type\":\"response.output_item.done\",\"output_index\":5,\"item\":{\"id\":\"ws_find\",\"type\":\"web_search_call\",\"status\":\"completed\",\"action\":{\"type\":\"find_in_page\",\"url\":\"https://example.com\",\"pattern\":\"Rust\"}}}\n\n",
            "event: response.output_item.done\n",
            "data: {\"type\":\"response.output_item.done\",\"output_index\":6,\"item\":{\"id\":\"msg_1\",\"type\":\"message\",\"status\":\"completed\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"A😊中B\",\"annotations\":[{\"type\":\"url_citation\",\"url\":\"https://example.com\",\"title\":\"Late title\",\"start_index\":1,\"end_index\":3}]}]}}\n\n",
            "event: response.completed\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-5.6-sol\",\"status\":\"completed\",\"output\":[],\"usage\":{\"input_tokens\":9,\"output_tokens\":4}}}\n\n"
        )
        .to_string();
        let (output, completion) = collect_buffered(input, policy(1)).await;
        let payloads = anthropic_payloads(&output);
        let starts = payloads
            .iter()
            .filter_map(|payload| {
                payload
                    .pointer("/content_block/type")
                    .and_then(Value::as_str)
            })
            .collect::<Vec<_>>();
        assert_eq!(
            starts,
            vec![
                "server_tool_use",
                "web_search_tool_result",
                "server_tool_use",
                "web_search_tool_result",
                "server_tool_use",
                "web_search_tool_result",
                "text"
            ]
        );
        let first_result = payloads
            .iter()
            .find(|payload| {
                payload
                    .pointer("/content_block/tool_use_id")
                    .and_then(Value::as_str)
                    == Some("ws_search")
            })
            .unwrap();
        assert_eq!(
            first_result["content_block"]["content"][0]["title"],
            "Late title"
        );
        let citation = payloads
            .iter()
            .find_map(|payload| payload.pointer("/delta/citation"))
            .unwrap();
        assert_eq!(citation["cited_text"], "😊中");
        let terminal = payloads
            .iter()
            .find(|payload| payload.get("type").and_then(Value::as_str) == Some("message_delta"))
            .unwrap();
        assert_eq!(terminal["delta"]["stop_reason"], "end_turn");
        assert_eq!(
            terminal["usage"]["server_tool_use"]["web_search_requests"],
            1
        );
        assert_eq!(completion.outcome(), Some(Ok(())));
    }

    #[tokio::test]
    async fn missing_search_sources_renders_an_empty_result_array() {
        let input = concat!(
            "event: response.output_item.done\n",
            "data: {\"output_index\":0,\"item\":{\"id\":\"ws_empty\",\"type\":\"web_search_call\",\"status\":\"completed\",\"action\":{\"type\":\"search\",\"query\":\"nothing\"}}}\n\n",
            "event: response.completed\n",
            "data: {\"response\":{\"id\":\"resp_empty\",\"model\":\"gpt-5.6-sol\",\"status\":\"completed\",\"output\":[]}}\n\n"
        )
        .to_string();
        let (output, completion) = collect_buffered(input, policy(1)).await;
        let result = anthropic_payloads(&output)
            .into_iter()
            .find(|payload| {
                payload
                    .pointer("/content_block/type")
                    .and_then(Value::as_str)
                    == Some("web_search_tool_result")
            })
            .unwrap();
        assert_eq!(result["content_block"]["content"], json!([]));
        assert_eq!(completion.outcome(), Some(Ok(())));
    }

    #[tokio::test]
    async fn max_uses_excess_fails_the_buffered_response() {
        let search = |index: u64| {
            format!(
                "event: response.output_item.done\ndata: {{\"output_index\":{index},\"item\":{{\"id\":\"ws_{index}\",\"type\":\"web_search_call\",\"status\":\"completed\",\"action\":{{\"type\":\"search\",\"query\":\"q{index}\",\"sources\":[]}}}}}}\n\n"
            )
        };
        let input = format!(
            "{}{}event: response.completed\ndata: {{\"response\":{{\"status\":\"completed\",\"output\":[]}}}}\n\n",
            search(0),
            search(1)
        );
        let (output, completion) = collect_buffered(input, policy(1)).await;
        assert!(output.contains("WebSearch max_uses exceeded"));
        assert!(matches!(completion.outcome(), Some(Err(_))));
        assert!(!output.contains("content_block_start"));
    }

    #[tokio::test]
    async fn completed_event_does_not_flush_before_transport_eof() {
        let input = concat!(
            "event: response.output_item.done\n",
            "data: {\"output_index\":0,\"item\":{\"id\":\"ws_1\",\"type\":\"web_search_call\",\"status\":\"completed\",\"action\":{\"type\":\"search\",\"query\":\"q\",\"sources\":[]}}}\n\n",
            "event: response.completed\n",
            "data: {\"response\":{\"status\":\"completed\",\"output\":[]}}\n\n"
        );
        let upstream = stream::iter(vec![Ok::<_, std::io::Error>(Bytes::from(input))])
            .chain(stream::pending());
        let completion = StreamCompletion::default();
        let converted = create_buffered_anthropic_sse_stream(upstream, completion, policy(1));
        tokio::pin!(converted);
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(50), converted.next())
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn error_event_does_not_flush_before_transport_eof() {
        let input = concat!(
            "event: error\n",
            "data: {\"type\":\"error\",\"error\":{\"type\":\"server_error\"}}\n\n"
        );
        let upstream = stream::iter(vec![Ok::<_, std::io::Error>(Bytes::from(input))])
            .chain(stream::pending());
        let completion = StreamCompletion::default();
        let converted = create_buffered_anthropic_sse_stream(upstream, completion, policy(1));
        tokio::pin!(converted);

        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(50), converted.next())
                .await
                .is_err()
        );
    }

    #[test]
    fn completed_response_with_error_is_rejected() {
        let mut accumulator = WebSearchBufferedResponse::new();
        let error = accumulator
            .ingest_responses_event(
                "response.completed",
                &json!({
                    "response": {
                        "status": "completed",
                        "output": [],
                        "error": {"type": "server_error"}
                    }
                }),
            )
            .unwrap_err();
        assert!(error.contains("non-null error"));
    }
}
