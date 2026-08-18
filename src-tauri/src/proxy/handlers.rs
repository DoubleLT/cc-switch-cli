use axum::{
    extract::State,
    http::{HeaderMap, StatusCode, Uri},
    response::{IntoResponse, Response},
    Json,
};
use bytes::Bytes;
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    time::{Duration, Instant},
};

use crate::{app_config::AppType, provider::Provider};

use super::{
    error::ProxyError,
    forwarder::{ForwardOptions, RequestForwarder},
    handler_context::HandlerContext,
    metrics::estimate_tokens_from_value,
    providers::{
        web_search_bridge::{WebSearchBufferedResponse, WebSearchRequestPolicy},
        ClaudeAdapter, ProviderAdapter,
    },
    response::{
        build_anthropic_stream_response, build_buffered_codex_anthropic_response_with_context,
        build_buffered_codex_chat_response, build_buffered_codex_chat_response_with_context,
        build_buffered_json_response, build_buffered_passthrough_response,
        build_codex_anthropic_response_with_context,
        build_codex_anthropic_stream_response_with_context, build_codex_chat_error_response,
        build_codex_chat_response_with_context, build_codex_chat_stream_response_with_context,
        build_json_response, build_passthrough_response, is_sse_response, PreparedResponse,
    },
    response_handler::{proxy_error_response, ResponseHandler, SuccessSyncInfo},
    server::ProxyServerState,
    sse::{strip_sse_field, take_sse_block},
    types::RectifierConfig,
    usage::{log_error_request, RequestLogContext, UsageLogPolicy},
};

pub async fn health_check() -> impl IntoResponse {
    (StatusCode::OK, Json(json!({ "ok": true })))
}

pub async fn get_status(State(state): State<ProxyServerState>) -> impl IntoResponse {
    Json(state.snapshot_status().await)
}

/// Return the active cc-switch-managed Codex model catalog.
///
/// Codex probes `/models` or `/v1/models` and expects its native catalog
/// envelope with a top-level `models` field.
pub async fn handle_models() -> Result<Json<Value>, ProxyError> {
    let generated_path = crate::codex_config::get_codex_model_catalog_path();
    let active_catalog_path = match crate::codex_config::read_codex_config_text() {
        Ok(config_text) => {
            crate::codex_config::resolve_cc_switch_catalog_path(&config_text, &generated_path)
        }
        Err(_) => None,
    };

    let catalog = if let Some(catalog_path) =
        active_catalog_path.as_ref().filter(|path| path.exists())
    {
        let text = std::fs::read_to_string(catalog_path).unwrap_or_default();
        serde_json::from_str(&text).unwrap_or(json!({ "models": [] }))
    } else {
        if active_catalog_path.is_none() {
            log::debug!(
                "[models] stale guard: catalog not served because config.toml does not reference the cc-switch catalog"
            );
        }
        json!({ "models": [] })
    };

    Ok(Json(catalog))
}

pub async fn handle_messages(
    State(state): State<ProxyServerState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    handle_claude_request(state, headers, body).await
}

pub async fn handle_chat_completions(
    State(state): State<ProxyServerState>,
    uri: Uri,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    handle_passthrough_request(
        state,
        headers,
        body,
        AppType::Codex,
        endpoint_with_query(&uri, "/chat/completions"),
    )
    .await
}

pub async fn handle_responses(
    State(state): State<ProxyServerState>,
    uri: Uri,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    handle_passthrough_request(
        state,
        headers,
        body,
        AppType::Codex,
        endpoint_with_query(&uri, "/responses"),
    )
    .await
}

pub async fn handle_responses_compact(
    State(state): State<ProxyServerState>,
    uri: Uri,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    handle_passthrough_request(
        state,
        headers,
        body,
        AppType::Codex,
        endpoint_with_query(&uri, "/responses/compact"),
    )
    .await
}

fn endpoint_with_query(uri: &Uri, endpoint: &str) -> String {
    match uri.query() {
        Some(query) => format!("{endpoint}?{query}"),
        None => endpoint.to_string(),
    }
}

pub async fn handle_gemini(
    State(state): State<ProxyServerState>,
    uri: Uri,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let endpoint = uri
        .path_and_query()
        .map(|value| value.as_str().to_string())
        .unwrap_or_else(|| uri.path().to_string());
    let endpoint = endpoint
        .strip_prefix("/gemini")
        .unwrap_or(endpoint.as_str())
        .to_string();
    handle_passthrough_request(state, headers, body, AppType::Gemini, endpoint).await
}

async fn handle_claude_request(
    state: ProxyServerState,
    headers: HeaderMap,
    body: Value,
) -> Response {
    state
        .record_estimated_input_tokens(estimate_tokens_from_value(&body))
        .await;
    let context = match HandlerContext::load(&state, AppType::Claude, &headers, &body).await {
        Ok(context) => context,
        Err(error) => {
            state.record_request_error(&error).await;
            return proxy_error_response(error);
        }
    };

    let forwarder = match RequestForwarder::new(context.provider_router.clone()) {
        Ok(forwarder) => forwarder
            .with_optimizer_config(context.optimizer_config.clone())
            .with_copilot_optimizer_config(context.copilot_optimizer_config.clone())
            .with_session(context.session_id.clone(), context.session_client_provided)
            .with_gemini_shadow(context.state.gemini_shadow.clone()),
        Err(error) => {
            context.state.record_request_error(&error).await;
            return proxy_error_response(error);
        }
    };
    forwarder.prewarm_provider_clients(&context.app_type, context.providers());

    let is_stream = body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let web_search_policy = WebSearchRequestPolicy::from_anthropic_request(&body)
        .ok()
        .flatten();
    let tool_schema_hints =
        super::providers::transform_gemini::extract_anthropic_tool_schema_hints(&body);
    let tool_schema_hints = (!tool_schema_hints.is_empty()).then_some(tool_schema_hints);
    let adapter = ClaudeAdapter::new();

    if is_stream {
        let first_byte_timeout = context.streaming_first_byte_timeout();
        let request_started_at = Instant::now();
        let options = ForwardOptions {
            max_retries: context.app_proxy.max_retries,
            request_timeout: first_byte_timeout,
            bypass_circuit_breaker: !context.app_proxy.auto_failover_enabled,
        };
        let forward_result = match forwarder
            .forward_response_detailed(
                &context.app_type,
                "/v1/messages",
                body,
                &headers,
                context.providers().to_vec(),
                options,
                context.rectifier_config.clone(),
            )
            .await
        {
            Ok(response) => response,
            Err(failure) => {
                let super::forwarder::ForwardFailure { provider, error } = failure;
                if let Some(provider) = provider.or_else(|| context.primary_provider().cloned()) {
                    let request_log = RequestLogContext::from_handler(
                        &context,
                        provider,
                        true,
                        UsageLogPolicy::Passthrough,
                    );
                    log_error_request(&context.state, &request_log, &error).await;
                }
                context.state.record_request_error(&error).await;
                return proxy_error_response(error);
            }
        };

        let api_format = super::providers::get_claude_api_format(&forward_result.provider);
        let request_log = RequestLogContext::from_handler(
            &context,
            forward_result.provider.clone(),
            true,
            if adapter.needs_transform(&forward_result.provider) {
                UsageLogPolicy::Transformed
            } else {
                UsageLogPolicy::Passthrough
            },
        );
        let response = forward_result.response;
        let status = response.status();
        let success_sync = status.is_success().then(|| SuccessSyncInfo {
            app_type: context.app_type.clone(),
            provider: forward_result.provider.clone(),
            current_provider_id_at_start: context.current_provider_id_at_start.clone(),
        });
        let first_byte_timeout = remaining_timeout(first_byte_timeout, request_started_at);
        let idle_timeout = context.streaming_idle_timeout();
        let response_result = match response {
            super::forwarder::StreamingResponse::Live(response)
                if adapter.needs_transform(&forward_result.provider) =>
            {
                let upstream_is_sse = is_sse_response(&response);
                if should_use_claude_transform_streaming(
                    is_stream,
                    upstream_is_sse,
                    api_format,
                    forward_result.provider.is_codex_oauth(),
                ) {
                    build_anthropic_stream_response(
                        response,
                        first_byte_timeout,
                        idle_timeout,
                        api_format,
                        Some(context.state.gemini_shadow.clone()),
                        Some(forward_result.provider.id.clone()),
                        Some(context.session_id.clone()),
                        tool_schema_hints.clone(),
                        (api_format == "openai_responses")
                            .then(|| web_search_policy.clone())
                            .flatten(),
                    )
                } else {
                    build_json_response(response, first_byte_timeout, |body| {
                        adapter.transform_response_with_web_search(
                            body,
                            (api_format == "openai_responses")
                                .then_some(web_search_policy.as_ref())
                                .flatten(),
                        )
                    })
                    .await
                }
            }
            super::forwarder::StreamingResponse::Live(response) => {
                build_passthrough_response(response, first_byte_timeout, idle_timeout).await
            }
            super::forwarder::StreamingResponse::Buffered(response) => {
                if adapter.needs_transform(&forward_result.provider) {
                    build_buffered_json_response(status, &response.headers, response.body, |body| {
                        if api_format == "gemini_native" {
                            super::providers::transform_gemini_response_for_provider(
                                body,
                                &forward_result.provider,
                                Some(&context.session_id),
                                Some(context.state.gemini_shadow.as_ref()),
                                tool_schema_hints.as_ref(),
                            )
                        } else {
                            adapter.transform_response_with_web_search(
                                body,
                                (api_format == "openai_responses")
                                    .then_some(web_search_policy.as_ref())
                                    .flatten(),
                            )
                        }
                    })
                } else {
                    build_buffered_passthrough_response(status, &response.headers, response.body)
                }
            }
        };

        return ResponseHandler::finish_streaming(
            &context.state,
            response_result,
            status,
            success_sync,
            Some(request_log),
        )
        .await;
    }

    let options = ForwardOptions {
        max_retries: context.app_proxy.max_retries,
        request_timeout: context.non_streaming_timeout(),
        bypass_circuit_breaker: !context.app_proxy.auto_failover_enabled,
    };

    let forward_result = match forwarder
        .forward_buffered_response_detailed(
            &context.app_type,
            "/v1/messages",
            body,
            &headers,
            context.providers().to_vec(),
            options,
            context.rectifier_config.clone(),
        )
        .await
    {
        Ok(response) => response,
        Err(failure) => {
            let super::forwarder::ForwardFailure { provider, error } = failure;
            if let Some(provider) = provider.or_else(|| context.primary_provider().cloned()) {
                let request_log = RequestLogContext::from_handler(
                    &context,
                    provider,
                    false,
                    UsageLogPolicy::Passthrough,
                );
                log_error_request(&context.state, &request_log, &error).await;
            }
            context.state.record_request_error(&error).await;
            return proxy_error_response(error);
        }
    };

    let provider = &forward_result.provider;
    let request_log = RequestLogContext::from_handler(
        &context,
        provider.clone(),
        false,
        if adapter.needs_transform(provider) {
            UsageLogPolicy::Transformed
        } else {
            UsageLogPolicy::Passthrough
        },
    );
    let response = forward_result.response;
    let status = response.status;
    let success_sync = status.is_success().then(|| SuccessSyncInfo {
        app_type: context.app_type.clone(),
        provider: provider.clone(),
        current_provider_id_at_start: context.current_provider_id_at_start.clone(),
    });
    let api_format = super::providers::get_claude_api_format(provider);
    let response_result = if adapter.needs_transform(provider) {
        build_buffered_claude_transform_response(
            status,
            &response.headers,
            response.body,
            api_format == "openai_responses",
            (api_format == "openai_responses")
                .then_some(web_search_policy.as_ref())
                .flatten(),
            |body| {
                if api_format == "gemini_native" {
                    super::providers::transform_gemini_response_for_provider(
                        body,
                        provider,
                        Some(&context.session_id),
                        Some(context.state.gemini_shadow.as_ref()),
                        tool_schema_hints.as_ref(),
                    )
                } else {
                    adapter.transform_response_with_web_search(
                        body,
                        (api_format == "openai_responses")
                            .then_some(web_search_policy.as_ref())
                            .flatten(),
                    )
                }
            },
        )
    } else {
        build_buffered_passthrough_response(status, &response.headers, response.body)
    };

    ResponseHandler::finish_buffered(
        &context.state,
        response_result,
        status,
        success_sync,
        Some(request_log),
    )
    .await
}

fn build_buffered_claude_transform_response<F>(
    status: reqwest::StatusCode,
    headers: &reqwest::header::HeaderMap,
    body: Bytes,
    aggregate_responses_sse: bool,
    web_search_policy: Option<&WebSearchRequestPolicy>,
    transform: F,
) -> Result<PreparedResponse, ProxyError>
where
    F: FnOnce(Value) -> Result<Value, ProxyError>,
{
    if status.is_success()
        && aggregate_responses_sse
        && serde_json::from_slice::<Value>(&body).is_err()
        && buffered_body_looks_like_sse(&body)
    {
        let body = if let Some(policy) = web_search_policy {
            responses_sse_to_web_search_response_value(&String::from_utf8_lossy(&body), policy)?
        } else {
            responses_sse_to_response_value(&String::from_utf8_lossy(&body))?
        };
        let body = serde_json::to_vec(&body).map_err(|error| {
            ProxyError::RequestFailed(format!("serialize aggregated upstream SSE failed: {error}"))
        })?;
        return build_buffered_json_response(status, headers, Bytes::from(body), transform);
    }

    build_buffered_json_response(status, headers, body, transform)
}

fn buffered_body_looks_like_sse(body: &[u8]) -> bool {
    let Ok(body) = std::str::from_utf8(body) else {
        return false;
    };
    body.lines().map(str::trim).any(|line| {
        !line.is_empty()
            && !line.starts_with(':')
            && matches!(
                line.split_once(':'),
                Some(("event" | "data" | "id" | "retry", _))
            )
    })
}

fn responses_sse_to_web_search_response_value(
    body: &str,
    policy: &WebSearchRequestPolicy,
) -> Result<Value, ProxyError> {
    let mut buffer = body.to_string();
    let mut accumulator = WebSearchBufferedResponse::new();

    while let Some(block) = take_sse_block(&mut buffer) {
        let mut event_name = "";
        let mut data_lines = Vec::new();
        for line in block.lines() {
            if let Some(event) = strip_sse_field(line, "event") {
                event_name = event.trim();
            } else if let Some(data) = strip_sse_field(line, "data") {
                data_lines.push(data);
            }
        }
        if data_lines.is_empty() {
            continue;
        }
        let data = data_lines.join("\n");
        if data.trim() == "[DONE]" {
            continue;
        }
        let data: Value = serde_json::from_str(&data).map_err(|error| {
            ProxyError::TransformError(format!("Failed to parse upstream SSE event: {error}"))
        })?;
        let event_name = if event_name.is_empty() {
            data.get("type").and_then(Value::as_str).unwrap_or("")
        } else {
            event_name
        };
        if matches!(event_name, "response.failed" | "error") {
            return Err(responses_event_to_anthropic_error(&data));
        }
        accumulator
            .ingest_responses_event(event_name, &data)
            .map_err(ProxyError::TransformError)?;
    }

    if !buffer.trim().is_empty() {
        return Err(ProxyError::TransformError(
            "WebSearch stream ended with an incomplete SSE frame".to_string(),
        ));
    }

    accumulator
        .seal_responses(policy)
        .map_err(ProxyError::TransformError)
}

fn responses_sse_to_response_value(body: &str) -> Result<Value, ProxyError> {
    let mut buffer = body.to_string();
    let mut terminal_response: Option<(String, Value)> = None;
    let mut output_items: Vec<(Option<u64>, usize, Value)> = Vec::new();
    let mut arrival_index = 0usize;

    while let Some(block) = take_sse_block(&mut buffer) {
        let mut event_name = "";
        let mut data_lines = Vec::new();

        for line in block.lines() {
            if let Some(event) = strip_sse_field(line, "event") {
                event_name = event.trim();
            } else if let Some(data) = strip_sse_field(line, "data") {
                data_lines.push(data);
            }
        }

        if data_lines.is_empty() {
            continue;
        }

        let data = data_lines.join("\n");
        if data.trim() == "[DONE]" {
            continue;
        }

        let data: Value = serde_json::from_str(&data).map_err(|error| {
            ProxyError::TransformError(format!("Failed to parse upstream SSE event: {error}"))
        })?;
        let data_event_name = data.get("type").and_then(Value::as_str).unwrap_or("");
        if !event_name.is_empty() && !data_event_name.is_empty() && event_name != data_event_name {
            return Err(ProxyError::TransformError(format!(
                "Conflicting upstream SSE event names: event={event_name}, data.type={data_event_name}"
            )));
        }
        let event_name = if event_name.is_empty() {
            data_event_name
        } else {
            event_name
        };

        if terminal_response.is_some()
            && !matches!(event_name, "response.completed" | "response.incomplete")
        {
            return Err(ProxyError::TransformError(format!(
                "upstream SSE event {event_name:?} received after terminal event"
            )));
        }

        match event_name {
            "response.output_item.done" => {
                let item = data.get("item").ok_or_else(|| {
                    ProxyError::TransformError(
                        "response.output_item.done is missing item".to_string(),
                    )
                })?;
                let output_index = data.get("output_index").and_then(Value::as_u64);
                let item_id = item.get("id").and_then(Value::as_str);
                let duplicate_or_conflict =
                    output_items
                        .iter()
                        .find(|(existing_index, _, existing_item)| {
                            (*existing_index == output_index && output_index.is_some())
                                || item_id.is_some_and(|item_id| {
                                    existing_item.get("id").and_then(Value::as_str) == Some(item_id)
                                })
                        });
                if let Some((existing_index, _, existing_item)) = duplicate_or_conflict {
                    if *existing_index == output_index && existing_item == item {
                        continue;
                    }
                    return Err(ProxyError::TransformError(
                        "Conflicting response.output_item.done identity or payload".to_string(),
                    ));
                }
                output_items.push((output_index, arrival_index, item.clone()));
                arrival_index += 1;
            }
            "response.completed" | "response.incomplete" => {
                let response = data
                    .get("response")
                    .cloned()
                    .unwrap_or_else(|| data.clone());
                validate_responses_terminal_event(event_name, &response)?;
                if let Some((existing_event_name, existing_response)) = &terminal_response {
                    if existing_event_name == event_name && existing_response == &response {
                        continue;
                    }
                    return Err(ProxyError::TransformError(
                        "Conflicting or repeated terminal response event".to_string(),
                    ));
                }
                terminal_response = Some((event_name.to_string(), response));
            }
            "response.failed" | "error" => return Err(responses_event_to_anthropic_error(&data)),
            _ => {}
        }
    }

    if !buffer.trim().is_empty() {
        return Err(ProxyError::TransformError(
            "Responses stream ended with an incomplete SSE frame".to_string(),
        ));
    }

    let (_, mut response) = terminal_response.ok_or_else(|| {
        ProxyError::TransformError("No terminal response event in upstream SSE".to_string())
    })?;

    if !output_items.is_empty() {
        output_items.sort_by(|left, right| match (left.0, right.0) {
            (Some(left_index), Some(right_index)) => {
                left_index.cmp(&right_index).then(left.1.cmp(&right.1))
            }
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => left.1.cmp(&right.1),
        });
        let Some(response) = response.as_object_mut() else {
            return Err(ProxyError::TransformError(
                "terminal response payload is not an object".to_string(),
            ));
        };

        let mut merged_output = response
            .get("output")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .cloned()
                    .enumerate()
                    .map(|(index, item)| (index as u64, item))
                    .collect::<BTreeMap<_, _>>()
            })
            .unwrap_or_default();

        let mut snapshot_ids = BTreeMap::new();
        for (index, item) in &merged_output {
            if let Some(item_id) = item.get("id").and_then(Value::as_str) {
                if snapshot_ids.insert(item_id.to_string(), *index).is_some() {
                    return Err(ProxyError::TransformError(format!(
                        "terminal response contains duplicate output item id {item_id}"
                    )));
                }
            }
        }

        for (output_index, _, item) in output_items {
            if let Some(output_index) = output_index {
                let item_id = item.get("id").and_then(Value::as_str);
                if let Some(existing) = merged_output.get(&output_index) {
                    let existing_id = existing.get("id").and_then(Value::as_str);
                    if existing_id.is_some() && item_id.is_some() && existing_id != item_id {
                        return Err(ProxyError::TransformError(format!(
                            "output index {output_index} identifies conflicting item ids"
                        )));
                    }
                }
                if let Some(item_id) = item_id {
                    if let Some(existing_index) = snapshot_ids.get(item_id) {
                        if *existing_index != output_index {
                            return Err(ProxyError::TransformError(format!(
                                "output item id {item_id} appears at conflicting indexes"
                            )));
                        }
                    }
                    snapshot_ids.insert(item_id.to_string(), output_index);
                }
                merged_output.insert(output_index, item);
                continue;
            }

            let item_id = item.get("id").and_then(Value::as_str);
            let matching_index = item_id.and_then(|item_id| {
                merged_output.iter().find_map(|(index, existing)| {
                    (existing.get("id").and_then(Value::as_str) == Some(item_id)).then_some(*index)
                })
            });
            let output_index = matching_index.unwrap_or_else(|| {
                merged_output
                    .last_key_value()
                    .map(|(index, _)| index.saturating_add(1))
                    .unwrap_or(0)
            });
            merged_output.insert(output_index, item);
        }

        response.insert(
            "output".to_string(),
            Value::Array(merged_output.into_values().collect()),
        );
    }

    Ok(response)
}

fn responses_event_to_anthropic_error(data: &Value) -> ProxyError {
    let error = data
        .get("error")
        .or_else(|| data.pointer("/response/error"));
    let message = error
        .and_then(|error| error.get("message"))
        .and_then(Value::as_str)
        .unwrap_or("Responses upstream failure");
    let error_type = error
        .and_then(|error| error.get("type"))
        .and_then(Value::as_str)
        .unwrap_or("api_error");
    let body = json!({
        "type": "error",
        "error": {
            "type": error_type,
            "message": message
        }
    });
    ProxyError::UpstreamError {
        status: reqwest::StatusCode::BAD_GATEWAY.as_u16(),
        body: Some(body.to_string()),
    }
}

fn validate_responses_terminal_event(event_name: &str, response: &Value) -> Result<(), ProxyError> {
    if let Some(error) = response.get("error").filter(|error| !error.is_null()) {
        let message = error
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("terminal response contains an error");
        return Err(ProxyError::TransformError(message.to_string()));
    }

    let status = response.get("status").and_then(Value::as_str);
    match event_name {
        "response.completed" if status != Some("completed") => Err(ProxyError::TransformError(
            format!("response.completed carried invalid status {:?}", status),
        )),
        "response.incomplete" if status != Some("incomplete") => Err(ProxyError::TransformError(
            format!("response.incomplete carried invalid status {:?}", status),
        )),
        "response.incomplete" => {
            let reason = response
                .pointer("/incomplete_details/reason")
                .and_then(Value::as_str);
            if matches!(reason, Some("max_output_tokens") | Some("max_tokens")) {
                Ok(())
            } else {
                Err(ProxyError::TransformError(format!(
                    "unsupported response.incomplete reason {:?}",
                    reason
                )))
            }
        }
        _ => Ok(()),
    }
}

fn should_use_claude_transform_streaming(
    _requested_streaming: bool,
    upstream_is_sse: bool,
    api_format: &str,
    is_codex_oauth: bool,
) -> bool {
    upstream_is_sse || (is_codex_oauth && api_format == "openai_responses")
}

async fn handle_passthrough_request(
    state: ProxyServerState,
    headers: HeaderMap,
    body: Value,
    app_type: AppType,
    endpoint: String,
) -> Response {
    state
        .record_estimated_input_tokens(estimate_tokens_from_value(&body))
        .await;
    let context = match HandlerContext::load(&state, app_type, &headers, &body).await {
        Ok(context) => context,
        Err(error) => {
            state.record_request_error(&error).await;
            return proxy_error_response(error);
        }
    };

    let forwarder = match RequestForwarder::new(context.provider_router.clone()) {
        Ok(forwarder) => forwarder
            .with_optimizer_config(context.optimizer_config.clone())
            .with_copilot_optimizer_config(context.copilot_optimizer_config.clone())
            .with_session(context.session_id.clone(), context.session_client_provided)
            .with_codex_chat_history(context.state.codex_chat_history.clone()),
        Err(error) => {
            context.state.record_request_error(&error).await;
            return proxy_error_response(error);
        }
    };
    forwarder.prewarm_provider_clients(&context.app_type, context.providers());

    let is_stream = request_is_streaming(&context.app_type, &endpoint, &body);
    let codex_tool_context = matches!(context.app_type, AppType::Codex).then(|| {
        super::providers::transform_codex_chat::build_codex_tool_context_from_request(&body)
    });
    let options = if is_stream {
        ForwardOptions {
            max_retries: context.app_proxy.max_retries,
            request_timeout: context.streaming_first_byte_timeout(),
            bypass_circuit_breaker: !context.app_proxy.auto_failover_enabled,
        }
    } else {
        ForwardOptions {
            max_retries: context.app_proxy.max_retries,
            request_timeout: context.non_streaming_timeout(),
            bypass_circuit_breaker: !context.app_proxy.auto_failover_enabled,
        }
    };

    if is_stream {
        let first_byte_timeout = context.streaming_first_byte_timeout();
        let request_started_at = Instant::now();
        let forward_result = match forwarder
            .forward_response_detailed(
                &context.app_type,
                &endpoint,
                body,
                &headers,
                context.providers().to_vec(),
                options,
                RectifierConfig::default(),
            )
            .await
        {
            Ok(response) => response,
            Err(failure) => {
                let super::forwarder::ForwardFailure { provider, error } = failure;
                let provider = provider.or_else(|| context.primary_provider().cloned());
                if let Some(provider) = provider.as_ref() {
                    let request_log = RequestLogContext::from_handler(
                        &context,
                        provider.clone(),
                        true,
                        passthrough_usage_log_policy(&context.app_type, provider, &endpoint),
                    );
                    log_error_request(&context.state, &request_log, &error).await;
                }
                if matches!(context.app_type, AppType::Codex) {
                    context.state.record_request_error(&error).await;
                    return build_codex_proxy_error_response(
                        &context,
                        provider.as_ref(),
                        &endpoint,
                        &error,
                    );
                }
                context.state.record_request_error(&error).await;
                return proxy_error_response(error);
            }
        };

        let response = forward_result.response;
        let status = response.status();
        let converts_codex_chat = super::providers::should_convert_codex_responses_to_chat(
            &forward_result.provider,
            &endpoint,
        );
        let converts_codex_anthropic =
            super::providers::should_convert_codex_responses_to_anthropic(
                &forward_result.provider,
                &endpoint,
            );
        let success_sync = status.is_success().then(|| SuccessSyncInfo {
            app_type: context.app_type.clone(),
            provider: forward_result.provider.clone(),
            current_provider_id_at_start: context.current_provider_id_at_start.clone(),
        });
        let response_result = match response {
            super::forwarder::StreamingResponse::Live(response)
                if converts_codex_anthropic
                    && status.is_success()
                    && !is_json_response(&response) =>
            {
                build_codex_anthropic_stream_response_with_context(
                    response,
                    remaining_timeout(first_byte_timeout, request_started_at),
                    context.streaming_idle_timeout(),
                    codex_tool_context.clone().unwrap_or_default(),
                )
            }
            super::forwarder::StreamingResponse::Live(response) if converts_codex_anthropic => {
                build_codex_anthropic_response_with_context(
                    response,
                    remaining_timeout(first_byte_timeout, request_started_at),
                    true,
                    codex_tool_context.clone().unwrap_or_default(),
                )
                .await
            }
            super::forwarder::StreamingResponse::Live(response)
                if converts_codex_chat && status.is_success() =>
            {
                build_codex_chat_stream_response_with_context(
                    response,
                    remaining_timeout(first_byte_timeout, request_started_at),
                    context.streaming_idle_timeout(),
                    context.state.codex_chat_history.clone(),
                    codex_tool_context.clone().unwrap_or_default(),
                )
            }
            super::forwarder::StreamingResponse::Live(response) if converts_codex_chat => {
                build_codex_chat_error_response(
                    response,
                    remaining_timeout(first_byte_timeout, request_started_at),
                    context.state.codex_chat_history.clone(),
                )
                .await
            }
            super::forwarder::StreamingResponse::Live(response) => {
                build_passthrough_response(
                    response,
                    remaining_timeout(first_byte_timeout, request_started_at),
                    context.streaming_idle_timeout(),
                )
                .await
            }
            super::forwarder::StreamingResponse::Buffered(response) if converts_codex_anthropic => {
                build_buffered_codex_anthropic_response_with_context(
                    status,
                    &response.headers,
                    response.body,
                    true,
                    codex_tool_context.clone().unwrap_or_default(),
                )
            }
            super::forwarder::StreamingResponse::Buffered(response) if converts_codex_chat => {
                build_buffered_codex_chat_response_with_context(
                    status,
                    &response.headers,
                    response.body,
                    context.state.codex_chat_history.clone(),
                    codex_tool_context.clone().unwrap_or_default(),
                )
                .await
            }
            super::forwarder::StreamingResponse::Buffered(response) => {
                build_buffered_passthrough_response(status, &response.headers, response.body)
            }
        };
        let request_log = Some(RequestLogContext::from_handler(
            &context,
            forward_result.provider.clone(),
            true,
            if converts_codex_chat || converts_codex_anthropic {
                UsageLogPolicy::Transformed
            } else {
                UsageLogPolicy::Passthrough
            },
        ));
        return ResponseHandler::finish_streaming(
            &context.state,
            response_result,
            status,
            success_sync,
            request_log,
        )
        .await;
    }

    if matches!(context.app_type, AppType::Codex) {
        let streaming_first_byte_timeout = context.streaming_first_byte_timeout();
        let non_streaming_timeout = context.non_streaming_timeout();
        let request_started_at = Instant::now();
        let forward_result = match forwarder
            .forward_response_detailed(
                &context.app_type,
                &endpoint,
                body,
                &headers,
                context.providers().to_vec(),
                options,
                RectifierConfig::default(),
            )
            .await
        {
            Ok(response) => response,
            Err(failure) => {
                let super::forwarder::ForwardFailure { provider, error } = failure;
                let provider = provider.or_else(|| context.primary_provider().cloned());
                if let Some(provider) = provider.as_ref() {
                    let request_log = RequestLogContext::from_handler(
                        &context,
                        provider.clone(),
                        false,
                        passthrough_usage_log_policy(&context.app_type, provider, &endpoint),
                    );
                    log_error_request(&context.state, &request_log, &error).await;
                }
                context.state.record_request_error(&error).await;
                return build_codex_proxy_error_response(
                    &context,
                    provider.as_ref(),
                    &endpoint,
                    &error,
                );
            }
        };
        return finish_codex_live_aware_response(
            &context,
            &endpoint,
            forward_result,
            request_started_at,
            streaming_first_byte_timeout,
            non_streaming_timeout,
            codex_tool_context.unwrap_or_default(),
        )
        .await;
    }

    let forward_result = match forwarder
        .forward_buffered_response_detailed(
            &context.app_type,
            &endpoint,
            body,
            &headers,
            context.providers().to_vec(),
            options,
            RectifierConfig::default(),
        )
        .await
    {
        Ok(response) => response,
        Err(failure) => {
            let super::forwarder::ForwardFailure { provider, error } = failure;
            if let Some(provider) = provider.or_else(|| context.primary_provider().cloned()) {
                let request_log = RequestLogContext::from_handler(
                    &context,
                    provider,
                    false,
                    UsageLogPolicy::Passthrough,
                );
                log_error_request(&context.state, &request_log, &error).await;
            }
            context.state.record_request_error(&error).await;
            return proxy_error_response(error);
        }
    };

    let response = forward_result.response;
    let converts_codex_chat = super::providers::should_convert_codex_responses_to_chat(
        &forward_result.provider,
        &endpoint,
    );
    let success_sync = response.status.is_success().then(|| SuccessSyncInfo {
        app_type: context.app_type.clone(),
        provider: forward_result.provider.clone(),
        current_provider_id_at_start: context.current_provider_id_at_start.clone(),
    });
    let status = response.status;
    let request_log = Some(RequestLogContext::from_handler(
        &context,
        forward_result.provider.clone(),
        false,
        passthrough_usage_log_policy(&context.app_type, &forward_result.provider, &endpoint),
    ));
    let response_result = if converts_codex_chat {
        build_buffered_codex_chat_response(
            response.status,
            &response.headers,
            response.body,
            context.state.codex_chat_history.clone(),
        )
        .await
    } else {
        build_buffered_passthrough_response(response.status, &response.headers, response.body)
    };
    ResponseHandler::finish_buffered(
        &context.state,
        response_result,
        status,
        success_sync,
        request_log,
    )
    .await
}

fn build_codex_proxy_error_response(
    context: &HandlerContext,
    provider: Option<&Provider>,
    endpoint: &str,
    error: &ProxyError,
) -> Response {
    let body = codex_proxy_error_json(
        provider
            .map(|provider| provider.name.as_str())
            .unwrap_or("unknown"),
        &context.request_model,
        endpoint,
        error,
    );
    let body = match serde_json::to_vec(&body) {
        Ok(body) => body,
        Err(error) => {
            return proxy_error_response(ProxyError::Internal(format!(
                "serialize Codex proxy error failed: {error}"
            )));
        }
    };

    Response::builder()
        .status(error.status_code())
        .header("content-type", "application/json")
        .body(axum::body::Body::from(body))
        .unwrap_or_else(|error| {
            proxy_error_response(ProxyError::Internal(format!(
                "build Codex proxy error response failed: {error}"
            )))
        })
}

fn codex_proxy_error_json(
    provider_name: &str,
    request_model: &str,
    endpoint: &str,
    error: &ProxyError,
) -> Value {
    let (mut body, upstream_status) = match error {
        ProxyError::UpstreamError { status, body } => {
            let parsed_body = body
                .as_deref()
                .map(|body| serde_json::from_str::<Value>(body).unwrap_or_else(|_| json!(body)));
            (
                super::providers::transform_codex_chat::chat_error_to_response_error(
                    parsed_body.as_ref(),
                ),
                Some(*status),
            )
        }
        _ => (
            json!({
                "error": {
                    "message": codex_proxy_error_cause(error),
                    "type": "proxy_error",
                    "code": codex_proxy_error_code(error),
                    "param": Value::Null,
                }
            }),
            None,
        ),
    };

    let Some(error_obj) = body.get_mut("error").and_then(Value::as_object_mut) else {
        return body;
    };

    let cause = error_obj
        .get("message")
        .and_then(Value::as_str)
        .map(ToString::to_string)
        .filter(|message| !message.trim().is_empty())
        .unwrap_or_else(|| codex_proxy_error_cause(error));
    let status_fragment = upstream_status
        .map(|status| format!("; upstream_status: HTTP {status}"))
        .unwrap_or_default();
    let message = format!(
        "CC Switch local proxy failed while handling Codex endpoint {endpoint}. Provider: {provider_name}; model: {request_model}{status_fragment}; cause: {cause}"
    );

    error_obj.insert(
        "message".to_string(),
        Value::String(compact_codex_proxy_error_message(&message, 1800)),
    );

    if error_obj
        .get("type")
        .and_then(Value::as_str)
        .map(|value| value.trim().is_empty())
        .unwrap_or(true)
    {
        error_obj.insert("type".to_string(), Value::String("proxy_error".to_string()));
    }

    if error_obj.get("code").map(Value::is_null).unwrap_or(true) {
        error_obj.insert(
            "code".to_string(),
            Value::String(codex_proxy_error_code(error).to_string()),
        );
    }

    if !error_obj.contains_key("param") {
        error_obj.insert("param".to_string(), Value::Null);
    }

    error_obj.insert(
        "provider".to_string(),
        Value::String(provider_name.to_string()),
    );
    error_obj.insert(
        "model".to_string(),
        Value::String(request_model.to_string()),
    );
    error_obj.insert("endpoint".to_string(), Value::String(endpoint.to_string()));
    if let Some(status) = upstream_status {
        error_obj.insert(
            "upstream_status".to_string(),
            Value::Number(serde_json::Number::from(status)),
        );
    }

    body
}

fn codex_proxy_error_cause(error: &ProxyError) -> String {
    match error {
        ProxyError::ConfigError(message)
        | ProxyError::AuthError(message)
        | ProxyError::RequestFailed(message)
        | ProxyError::TransformError(message)
        | ProxyError::ForwardFailed(message)
        | ProxyError::BindFailed(message)
        | ProxyError::StopFailed(message)
        | ProxyError::ProviderUnhealthy(message)
        | ProxyError::DatabaseError(message)
        | ProxyError::InvalidRequest(message)
        | ProxyError::Timeout(message)
        | ProxyError::Internal(message) => message.clone(),
        other => other.to_string(),
    }
}

fn codex_proxy_error_code(error: &ProxyError) -> &'static str {
    match error {
        ProxyError::ForwardFailed(_) => "cc_switch_forward_failed",
        ProxyError::Timeout(_) | ProxyError::StreamIdleTimeout(_) => "cc_switch_timeout",
        ProxyError::NoAvailableProvider => "cc_switch_no_available_provider",
        ProxyError::AllProvidersCircuitOpen => "cc_switch_all_providers_circuit_open",
        ProxyError::NoProvidersConfigured => "cc_switch_no_providers_configured",
        ProxyError::MaxRetriesExceeded => "cc_switch_max_retries_exceeded",
        ProxyError::ProviderUnhealthy(_) => "cc_switch_provider_unhealthy",
        ProxyError::ConfigError(_) => "cc_switch_config_error",
        ProxyError::TransformError(_) => "cc_switch_transform_error",
        ProxyError::InvalidRequest(_) => "cc_switch_invalid_request",
        ProxyError::AuthError(_) => "cc_switch_auth_error",
        ProxyError::UpstreamError { .. } => "cc_switch_upstream_error",
        ProxyError::DatabaseError(_) => "cc_switch_database_error",
        ProxyError::Internal(_) => "cc_switch_internal_error",
        ProxyError::RequestFailed(_) => "cc_switch_request_failed",
        ProxyError::AlreadyRunning
        | ProxyError::NotRunning
        | ProxyError::BindFailed(_)
        | ProxyError::StopTimeout
        | ProxyError::StopFailed(_) => "cc_switch_proxy_error",
    }
}

fn compact_codex_proxy_error_message(message: &str, max_chars: usize) -> String {
    let normalized = message.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.chars().count() <= max_chars {
        return normalized;
    }

    let truncated = normalized
        .chars()
        .take(max_chars)
        .collect::<String>()
        .trim_end()
        .to_string();
    format!("{truncated}...(truncated)")
}

async fn finish_codex_live_aware_response(
    context: &HandlerContext,
    endpoint: &str,
    forward_result: super::forwarder::ForwardedResponse<super::forwarder::StreamingResponse>,
    request_started_at: Instant,
    streaming_first_byte_timeout: Option<Duration>,
    non_streaming_timeout: Option<Duration>,
    tool_context: super::providers::transform_codex_chat::CodexToolContext,
) -> Response {
    let provider = forward_result.provider;
    let response = forward_result.response;
    let status = response.status();
    let success_sync = status.is_success().then(|| SuccessSyncInfo {
        app_type: context.app_type.clone(),
        provider: provider.clone(),
        current_provider_id_at_start: context.current_provider_id_at_start.clone(),
    });

    if super::providers::should_convert_codex_responses_to_anthropic(&provider, endpoint) {
        let request_log = Some(RequestLogContext::from_handler(
            context,
            provider.clone(),
            false,
            UsageLogPolicy::Transformed,
        ));
        let response_result = match response {
            super::forwarder::StreamingResponse::Live(response) => {
                build_codex_anthropic_response_with_context(
                    response,
                    remaining_timeout(non_streaming_timeout, request_started_at),
                    false,
                    tool_context,
                )
                .await
            }
            super::forwarder::StreamingResponse::Buffered(response) => {
                build_buffered_codex_anthropic_response_with_context(
                    response.status,
                    &response.headers,
                    response.body,
                    false,
                    tool_context,
                )
            }
        };
        return ResponseHandler::finish_buffered(
            &context.state,
            response_result,
            status,
            success_sync,
            request_log,
        )
        .await;
    }

    if super::providers::should_convert_codex_responses_to_chat(&provider, endpoint) {
        return match response {
            super::forwarder::StreamingResponse::Live(response)
                if status.is_success() && is_sse_response(&response) =>
            {
                let request_log = Some(RequestLogContext::from_handler(
                    context,
                    provider.clone(),
                    true,
                    UsageLogPolicy::Transformed,
                ));
                let response_result = build_codex_chat_stream_response_with_context(
                    response,
                    remaining_timeout(streaming_first_byte_timeout, request_started_at),
                    context.streaming_idle_timeout(),
                    context.state.codex_chat_history.clone(),
                    tool_context,
                );
                ResponseHandler::finish_streaming(
                    &context.state,
                    response_result,
                    status,
                    success_sync,
                    request_log,
                )
                .await
            }
            super::forwarder::StreamingResponse::Live(response) => {
                let request_log = Some(RequestLogContext::from_handler(
                    context,
                    provider.clone(),
                    false,
                    UsageLogPolicy::Transformed,
                ));
                let timeout = remaining_timeout(non_streaming_timeout, request_started_at);
                let response_result = if status.is_success() {
                    build_codex_chat_response_with_context(
                        response,
                        timeout,
                        context.state.codex_chat_history.clone(),
                        tool_context,
                    )
                    .await
                } else {
                    build_codex_chat_error_response(
                        response,
                        timeout,
                        context.state.codex_chat_history.clone(),
                    )
                    .await
                };
                ResponseHandler::finish_buffered(
                    &context.state,
                    response_result,
                    status,
                    success_sync,
                    request_log,
                )
                .await
            }
            super::forwarder::StreamingResponse::Buffered(response) => {
                let request_log = Some(RequestLogContext::from_handler(
                    context,
                    provider.clone(),
                    false,
                    UsageLogPolicy::Transformed,
                ));
                let response_result = build_buffered_codex_chat_response_with_context(
                    response.status,
                    &response.headers,
                    response.body,
                    context.state.codex_chat_history.clone(),
                    tool_context,
                )
                .await;
                ResponseHandler::finish_buffered(
                    &context.state,
                    response_result,
                    status,
                    success_sync,
                    request_log,
                )
                .await
            }
        };
    }

    match response {
        super::forwarder::StreamingResponse::Live(response) => {
            let first_byte_timeout = if is_sse_response(&response) {
                streaming_first_byte_timeout
            } else {
                non_streaming_timeout
            };
            let response_result = build_passthrough_response(
                response,
                remaining_timeout(first_byte_timeout, request_started_at),
                context.streaming_idle_timeout(),
            )
            .await;
            if response_result
                .as_ref()
                .is_ok_and(|prepared| prepared.stream_completion.is_some())
            {
                let request_log = Some(RequestLogContext::from_handler(
                    context,
                    provider.clone(),
                    true,
                    UsageLogPolicy::Passthrough,
                ));
                ResponseHandler::finish_streaming(
                    &context.state,
                    response_result,
                    status,
                    success_sync,
                    request_log,
                )
                .await
            } else {
                let request_log = Some(RequestLogContext::from_handler(
                    context,
                    provider.clone(),
                    false,
                    UsageLogPolicy::Passthrough,
                ));
                ResponseHandler::finish_buffered(
                    &context.state,
                    response_result,
                    status,
                    success_sync,
                    request_log,
                )
                .await
            }
        }
        super::forwarder::StreamingResponse::Buffered(response) => {
            let request_log = Some(RequestLogContext::from_handler(
                context,
                provider.clone(),
                false,
                UsageLogPolicy::Passthrough,
            ));
            let response_result =
                build_buffered_passthrough_response(status, &response.headers, response.body);
            ResponseHandler::finish_buffered(
                &context.state,
                response_result,
                status,
                success_sync,
                request_log,
            )
            .await
        }
    }
}

fn request_is_streaming(app_type: &AppType, endpoint: &str, body: &Value) -> bool {
    if body
        .get("stream")
        .and_then(|value| value.as_bool())
        .unwrap_or(false)
    {
        return true;
    }

    matches!(app_type, AppType::Gemini)
        && (endpoint.contains("alt=sse") || endpoint.contains(":streamGenerateContent"))
}

fn passthrough_usage_log_policy(
    app_type: &AppType,
    provider: &Provider,
    endpoint: &str,
) -> UsageLogPolicy {
    if matches!(app_type, AppType::Codex)
        && (super::providers::should_convert_codex_responses_to_chat(provider, endpoint)
            || super::providers::should_convert_codex_responses_to_anthropic(provider, endpoint))
    {
        UsageLogPolicy::Transformed
    } else {
        UsageLogPolicy::Passthrough
    }
}

fn is_json_response(response: &super::forwarder::LiveResponse) -> bool {
    response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|content_type| {
            let media_type = content_type
                .split(';')
                .next()
                .unwrap_or("")
                .trim()
                .to_ascii_lowercase();
            media_type == "application/json" || media_type.ends_with("+json")
        })
}

fn remaining_timeout(timeout: Option<Duration>, started_at: Instant) -> Option<Duration> {
    timeout.map(|timeout| timeout.saturating_sub(started_at.elapsed()))
}

#[cfg(test)]
mod tests {
    use super::{
        build_buffered_claude_transform_response, endpoint_with_query, handle_responses,
        handle_responses_compact, responses_sse_to_response_value,
        responses_sse_to_web_search_response_value, should_use_claude_transform_streaming,
    };
    use crate::{
        app_config::AppType,
        database::Database,
        provider::Provider,
        proxy::{
            error::ProxyError,
            provider_router::ProviderRouter,
            providers::codex_chat_history::CodexChatHistoryStore,
            providers::gemini_shadow::GeminiShadowStore,
            providers::web_search_bridge::WebSearchRequestPolicy,
            providers::{ClaudeAdapter, ProviderAdapter},
            server::ProxyServerState,
            types::{ProxyConfig, ProxyStatus},
        },
    };
    use axum::{
        body::{to_bytes, Body},
        extract::State,
        http::{HeaderMap, StatusCode, Uri},
        response::Response,
        routing::any,
        Json, Router,
    };
    use bytes::Bytes;
    use futures::StreamExt;
    use serde_json::{json, Value};
    use std::{collections::HashMap, env, sync::Arc, time::Duration};
    use tempfile::TempDir;
    use tokio::sync::RwLock;

    struct TempHome {
        #[allow(dead_code)]
        dir: TempDir,
        original_home: Option<String>,
        original_userprofile: Option<String>,
        original_config_dir: Option<String>,
    }

    impl TempHome {
        fn new() -> Self {
            let dir = TempDir::new().expect("create temp home");
            let original_home = env::var("HOME").ok();
            let original_userprofile = env::var("USERPROFILE").ok();
            let original_config_dir = env::var("CC_SWITCH_CONFIG_DIR").ok();

            env::set_var("HOME", dir.path());
            env::set_var("USERPROFILE", dir.path());
            env::set_var("CC_SWITCH_CONFIG_DIR", dir.path().join(".cc-switch"));
            crate::settings::reload_test_settings();

            Self {
                dir,
                original_home,
                original_userprofile,
                original_config_dir,
            }
        }
    }

    impl Drop for TempHome {
        fn drop(&mut self) {
            match &self.original_home {
                Some(value) => env::set_var("HOME", value),
                None => env::remove_var("HOME"),
            }

            match &self.original_userprofile {
                Some(value) => env::set_var("USERPROFILE", value),
                None => env::remove_var("USERPROFILE"),
            }

            match &self.original_config_dir {
                Some(value) => env::set_var("CC_SWITCH_CONFIG_DIR", value),
                None => env::remove_var("CC_SWITCH_CONFIG_DIR"),
            }

            crate::settings::reload_test_settings();
        }
    }

    fn codex_test_state(db: Arc<Database>) -> ProxyServerState {
        ProxyServerState {
            db: db.clone(),
            config: Arc::new(RwLock::new(ProxyConfig::default())),
            status: Arc::new(RwLock::new(ProxyStatus::default())),
            start_time: Arc::new(RwLock::new(None)),
            current_providers: Arc::new(RwLock::new(HashMap::new())),
            provider_router: Arc::new(ProviderRouter::new(db)),
            codex_chat_history: Arc::new(CodexChatHistoryStore::default()),
            gemini_shadow: Arc::new(GeminiShadowStore::default()),
        }
    }

    async fn handle_hanging_chat_sse_upstream() -> Response {
        let stream = async_stream::stream! {
            yield Ok::<_, std::io::Error>(Bytes::from_static(
                b"data: {\"id\":\"chatcmpl_live\",\"object\":\"chat.completion.chunk\",\"created\":1710000000,\"model\":\"deepseek-chat\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hel\"}}]}\n\n",
            ));
            std::future::pending::<()>().await;
        };

        Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "text/event-stream")
            .body(Body::from_stream(stream))
            .expect("build hanging chat SSE response")
    }

    async fn spawn_hanging_chat_sse_upstream() -> (String, tokio::task::JoinHandle<()>) {
        let app = Router::new().route("/*path", any(handle_hanging_chat_sse_upstream));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind hanging SSE upstream listener");
        let address = listener.local_addr().expect("upstream listener address");
        let handle = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        (format!("http://{address}"), handle)
    }

    async fn handle_chat_tool_search_upstream() -> Json<Value> {
        Json(json!({
            "id": "chatcmpl_tool_search",
            "object": "chat.completion",
            "created": 1710000000,
            "model": "gpt-5.4",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call_tool_search_1",
                        "type": "function",
                        "function": {
                            "name": "tool_search",
                            "arguments": "{\"query\":\"Gmail search emails\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }))
    }

    async fn spawn_chat_tool_search_upstream() -> (String, tokio::task::JoinHandle<()>) {
        let app = Router::new().route("/*path", any(handle_chat_tool_search_upstream));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind tool_search upstream listener");
        let address = listener.local_addr().expect("upstream listener address");
        let handle = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        (format!("http://{address}"), handle)
    }

    async fn closed_base_url() -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind closed upstream listener");
        let address = listener.local_addr().expect("closed upstream address");
        drop(listener);
        format!("http://{address}")
    }

    #[test]
    fn codex_endpoint_with_query_preserves_responses_query() {
        let uri: Uri = "/v1/responses?foo=bar&api-version=2025-01-01"
            .parse()
            .unwrap();

        assert_eq!(
            endpoint_with_query(&uri, "/responses"),
            "/responses?foo=bar&api-version=2025-01-01"
        );
    }

    #[test]
    fn codex_endpoint_with_query_preserves_responses_compact_query() {
        let uri: Uri = "/v1/responses/compact?foo=bar".parse().unwrap();

        assert_eq!(
            endpoint_with_query(&uri, "/responses/compact"),
            "/responses/compact?foo=bar"
        );
    }

    #[tokio::test]
    #[serial_test::serial(home_settings)]
    async fn codex_non_stream_chat_sse_fallback_streams_without_waiting_for_eof() {
        let _home = TempHome::new();
        let (upstream_base_url, upstream_handle) = spawn_hanging_chat_sse_upstream().await;
        let db = Arc::new(Database::memory().expect("create memory database"));
        let provider = Provider::with_id(
            "codex-chat".to_string(),
            "Codex Chat".to_string(),
            json!({
                "api_format": "chat",
                "apiKey": "test-key",
                "base_url": upstream_base_url,
                "model": "deepseek-chat"
            }),
            None,
        );
        db.save_provider(AppType::Codex.as_str(), &provider)
            .expect("save Codex chat provider");
        db.set_current_provider(AppType::Codex.as_str(), &provider.id)
            .expect("set current Codex provider");
        let state = codex_test_state(db);

        let response = tokio::time::timeout(
            Duration::from_millis(500),
            handle_responses(
                State(state),
                Uri::from_static("/v1/responses"),
                HeaderMap::new(),
                Json(json!({
                    "model": "gpt-5.4",
                    "input": "hello"
                })),
            ),
        )
        .await
        .expect("handler should return before upstream SSE completes");

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response
                .headers()
                .get("content-type")
                .and_then(|value| value.to_str().ok()),
            Some("text/event-stream")
        );

        let mut stream = response.into_body().into_data_stream();
        let mut output = String::new();
        for _ in 0..8 {
            let chunk = tokio::time::timeout(Duration::from_millis(500), stream.next())
                .await
                .expect("converted SSE should yield before upstream EOF")
                .expect("stream should yield a chunk")
                .expect("stream chunk should be ok");
            output.push_str(core::str::from_utf8(&chunk).expect("SSE should be UTF-8"));

            if output.contains("event: response.output_text.delta") {
                break;
            }
        }

        assert!(output.contains("event: response.created"));
        assert!(output.contains("event: response.output_text.delta"));
        assert!(output.contains("\"delta\":\"Hel\""));

        upstream_handle.abort();
    }

    #[tokio::test]
    #[serial_test::serial(home_settings)]
    async fn codex_chat_provider_restores_tool_search_identity_through_handler() {
        let _home = TempHome::new();
        let (upstream_base_url, upstream_handle) = spawn_chat_tool_search_upstream().await;
        let db = Arc::new(Database::memory().expect("create memory database"));
        let provider = Provider::with_id(
            "codex-chat".to_string(),
            "Codex Chat".to_string(),
            json!({
                "api_format": "chat",
                "apiKey": "test-key",
                "base_url": upstream_base_url,
                "model": "gpt-5.4"
            }),
            None,
        );
        db.save_provider(AppType::Codex.as_str(), &provider)
            .expect("save Codex chat provider");
        db.set_current_provider(AppType::Codex.as_str(), &provider.id)
            .expect("set current Codex provider");
        let state = codex_test_state(db);

        let response = handle_responses(
            State(state),
            Uri::from_static("/v1/responses"),
            HeaderMap::new(),
            Json(json!({
                "model": "gpt-5.4",
                "tools": [{"type": "tool_search"}],
                "input": "Find Gmail tools"
            })),
        )
        .await;

        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("read response body");
        let body: Value = serde_json::from_slice(&body).expect("parse response json");

        assert_eq!(body["output"][0]["type"], "tool_search_call");
        assert_eq!(body["output"][0]["call_id"], "call_tool_search_1");
        assert_eq!(
            body["output"][0]["arguments"]["query"],
            "Gmail search emails"
        );

        upstream_handle.abort();
    }

    #[tokio::test]
    #[serial_test::serial(home_settings)]
    async fn codex_forward_failure_uses_responses_error_shape() {
        let _home = TempHome::new();
        let db = Arc::new(Database::memory().expect("create memory database"));
        let provider = Provider::with_id(
            "codex-broken".to_string(),
            "Broken Codex".to_string(),
            json!({
                "api_format": "responses",
                "apiKey": "test-key",
                "base_url": closed_base_url().await
            }),
            None,
        );
        db.save_provider(AppType::Codex.as_str(), &provider)
            .expect("save broken Codex provider");
        db.set_current_provider(AppType::Codex.as_str(), &provider.id)
            .expect("set current Codex provider");
        let state = codex_test_state(db);

        let response = handle_responses(
            State(state),
            Uri::from_static("/v1/responses?beta=true"),
            HeaderMap::new(),
            Json(json!({
                "model": "gpt-5.4",
                "input": "hello"
            })),
        )
        .await;

        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("read error response body");
        let body: Value = serde_json::from_slice(&body).expect("parse error response json");
        let error = &body["error"];
        let message = error["message"].as_str().expect("error message");

        assert!(message.contains("CC Switch local proxy failed"));
        assert!(message.contains("Broken Codex"));
        assert!(message.contains("gpt-5.4"));
        assert!(message.contains("/responses?beta=true"));
        assert_eq!(error["type"], "proxy_error");
        assert_eq!(error["code"], "cc_switch_forward_failed");
        assert!(error["param"].is_null());
        assert_eq!(error["provider"], "Broken Codex");
        assert_eq!(error["model"], "gpt-5.4");
        assert_eq!(error["endpoint"], "/responses?beta=true");
    }

    #[tokio::test]
    #[serial_test::serial(home_settings)]
    async fn streaming_codex_forward_failure_uses_responses_error_shape() {
        let _home = TempHome::new();
        let db = Arc::new(Database::memory().expect("create memory database"));
        let provider = Provider::with_id(
            "codex-broken".to_string(),
            "Broken Codex".to_string(),
            json!({
                "api_format": "responses",
                "apiKey": "test-key",
                "base_url": closed_base_url().await
            }),
            None,
        );
        db.save_provider(AppType::Codex.as_str(), &provider)
            .expect("save broken Codex provider");
        db.set_current_provider(AppType::Codex.as_str(), &provider.id)
            .expect("set current Codex provider");
        let state = codex_test_state(db);

        let response = handle_responses_compact(
            State(state),
            Uri::from_static("/v1/responses/compact?beta=true"),
            HeaderMap::new(),
            Json(json!({
                "model": "gpt-5.4",
                "input": "hello",
                "stream": true
            })),
        )
        .await;

        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("read error response body");
        let body: Value = serde_json::from_slice(&body).expect("parse error response json");
        let error = &body["error"];
        let message = error["message"].as_str().expect("error message");

        assert!(message.contains("CC Switch local proxy failed"));
        assert!(message.contains("Broken Codex"));
        assert!(message.contains("gpt-5.4"));
        assert!(message.contains("/responses/compact?beta=true"));
        assert_eq!(error["type"], "proxy_error");
        assert_eq!(error["code"], "cc_switch_forward_failed");
        assert!(error["param"].is_null());
        assert_eq!(error["provider"], "Broken Codex");
        assert_eq!(error["model"], "gpt-5.4");
        assert_eq!(error["endpoint"], "/responses/compact?beta=true");
    }

    #[test]
    fn responses_buffered_transform_aggregates_sse_before_json_parse() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::CONTENT_TYPE,
            reqwest::header::HeaderValue::from_static("text/event-stream"),
        );
        let sse = r#"event: response.output_item.done
data: {"type":"response.output_item.done","item":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hello"}]}}

event: response.completed
data: {"type":"response.completed","response":{"id":"resp_1","status":"completed","model":"gpt-5.4","output":[],"usage":{"input_tokens":10,"output_tokens":2}}}

"#;

        let prepared = build_buffered_claude_transform_response(
            reqwest::StatusCode::OK,
            &headers,
            Bytes::from(sse),
            true,
            None,
            Ok,
        )
        .unwrap();
        let body: Value = serde_json::from_slice(
            prepared
                .body_bytes
                .as_ref()
                .expect("buffered response should keep body bytes"),
        )
        .unwrap();

        assert_eq!(body["id"], "resp_1");
        assert_eq!(body["output"][0]["type"], "message");
        assert_eq!(body["output"][0]["content"][0]["text"], "hello");
    }

    #[test]
    fn responses_buffered_transform_aggregates_sse_without_content_type() {
        let headers = reqwest::header::HeaderMap::new();
        let sse = r#"event: response.completed
data: {"type":"response.completed","response":{"id":"resp_1","status":"completed","model":"gpt-5.4","output":[],"usage":{"input_tokens":10,"output_tokens":2}}}

"#;

        let prepared = build_buffered_claude_transform_response(
            reqwest::StatusCode::OK,
            &headers,
            Bytes::from(sse),
            true,
            None,
            Ok,
        )
        .unwrap();
        let body: Value = serde_json::from_slice(
            prepared
                .body_bytes
                .as_ref()
                .expect("buffered response should keep body bytes"),
        )
        .unwrap();

        assert_eq!(body["id"], "resp_1");
    }

    #[test]
    fn codex_oauth_responses_force_streaming_even_without_sse_content_type() {
        assert!(should_use_claude_transform_streaming(
            false,
            false,
            "openai_responses",
            true,
        ));
    }

    #[test]
    fn upstream_sse_response_always_uses_streaming_path() {
        assert!(should_use_claude_transform_streaming(
            false,
            true,
            "openai_chat",
            false,
        ));
    }

    #[test]
    fn non_streaming_response_stays_non_streaming_for_regular_openai_responses() {
        assert!(!should_use_claude_transform_streaming(
            false,
            false,
            "openai_responses",
            false,
        ));
    }

    #[test]
    fn responses_buffered_transform_keeps_json_fallback() {
        let headers = reqwest::header::HeaderMap::new();
        let json_body = json!({
            "id": "resp_json",
            "status": "completed",
            "model": "gpt-5.6-sol",
            "output": [],
            "usage": {"input_tokens": 3, "output_tokens": 1}
        });

        let prepared = build_buffered_claude_transform_response(
            reqwest::StatusCode::OK,
            &headers,
            Bytes::from(serde_json::to_vec(&json_body).unwrap()),
            true,
            None,
            Ok,
        )
        .unwrap();
        let body: Value = serde_json::from_slice(
            prepared
                .body_bytes
                .as_ref()
                .expect("buffered response should keep body bytes"),
        )
        .unwrap();

        assert_eq!(body, json_body);
    }

    #[test]
    fn responses_buffered_transform_preserves_plain_text_http_error() {
        let headers = reqwest::header::HeaderMap::new();
        let raw_body = Bytes::from_static(b"upstream unavailable");

        let prepared = build_buffered_claude_transform_response(
            reqwest::StatusCode::BAD_GATEWAY,
            &headers,
            raw_body.clone(),
            true,
            None,
            Ok,
        )
        .unwrap();

        assert_eq!(prepared.response.status(), reqwest::StatusCode::BAD_GATEWAY);
        assert_eq!(prepared.body_bytes.as_ref(), Some(&raw_body));
    }

    #[test]
    fn responses_buffered_transform_does_not_treat_http_error_as_sse() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::CONTENT_TYPE,
            reqwest::header::HeaderValue::from_static("text/plain"),
        );
        let raw_body = Bytes::from_static(b"data: relay overloaded\n\n");

        let prepared = build_buffered_claude_transform_response(
            reqwest::StatusCode::BAD_REQUEST,
            &headers,
            raw_body.clone(),
            true,
            None,
            Ok,
        )
        .unwrap();

        assert_eq!(prepared.response.status(), reqwest::StatusCode::BAD_REQUEST);
        assert_eq!(prepared.body_bytes.as_ref(), Some(&raw_body));
    }

    #[test]
    fn responses_buffered_failed_http_error_maps_to_anthropic_error() {
        let headers = reqwest::header::HeaderMap::new();
        let upstream = json!({
            "id": "resp_failed",
            "status": "failed",
            "error": {
                "message": "bad request",
                "type": "invalid_request_error"
            },
            "output": []
        });

        let prepared = build_buffered_claude_transform_response(
            reqwest::StatusCode::BAD_REQUEST,
            &headers,
            Bytes::from(serde_json::to_vec(&upstream).unwrap()),
            true,
            None,
            |body| ProviderAdapter::transform_response(&ClaudeAdapter, body),
        )
        .unwrap();
        let body: Value = serde_json::from_slice(prepared.body_bytes.as_ref().unwrap()).unwrap();

        assert_eq!(prepared.response.status(), reqwest::StatusCode::BAD_REQUEST);
        assert_eq!(body["type"], "error");
        assert_eq!(body["error"]["message"], "bad request");
    }

    #[test]
    fn responses_sse_to_response_value_collects_output_items() {
        let sse = r#"event: response.output_item.done
data: {"type":"response.output_item.done","item":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hello"}]}}

event: response.completed
data: {"type":"response.completed","response":{"id":"resp_1","status":"completed","model":"gpt-5.4","output":[],"usage":{"input_tokens":10,"output_tokens":2}}}

"#;

        let response = responses_sse_to_response_value(sse).unwrap();

        assert_eq!(response["id"], "resp_1");
        assert_eq!(response["output"][0]["type"], "message");
        assert_eq!(response["output"][0]["content"][0]["text"], "hello");
    }

    #[test]
    fn responses_sse_to_response_value_handles_crlf_delimiters() {
        let sse = "event: response.output_item.done\r\n\
data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"hi\"}]}}\r\n\
\r\n\
event: response.completed\r\n\
data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_crlf\",\"status\":\"completed\",\"model\":\"gpt-5.4\",\"output\":[],\"usage\":{\"input_tokens\":5,\"output_tokens\":1}}}\r\n\
\r\n";

        let response = responses_sse_to_response_value(sse).unwrap();

        assert_eq!(response["id"], "resp_crlf");
        assert_eq!(response["output"][0]["type"], "message");
        assert_eq!(response["output"][0]["content"][0]["text"], "hi");
    }

    #[test]
    fn responses_sse_data_only_incomplete_orders_items_and_maps_max_tokens() {
        let sse = concat!(
            "data: {\"type\":\"response.output_item.done\",\"output_index\":1,\"item\":{\"id\":\"msg_second\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"second\"}]}}\n\n",
            "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"id\":\"msg_first\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"first\"}]}}\n\n",
            "data: {\"type\":\"response.incomplete\",\"response\":{\"id\":\"resp_incomplete\",\"status\":\"incomplete\",\"model\":\"gpt-5.6-sol\",\"output\":[],\"incomplete_details\":{\"reason\":\"max_output_tokens\"},\"usage\":{\"input_tokens\":8,\"output_tokens\":6}}}\n\n"
        );

        let response = responses_sse_to_response_value(sse).unwrap();
        assert_eq!(response["status"], "incomplete");
        assert_eq!(response["output"][0]["id"], "msg_first");
        assert_eq!(response["output"][1]["id"], "msg_second");

        let anthropic =
            crate::proxy::providers::transform_responses::responses_to_anthropic(response).unwrap();
        assert_eq!(anthropic["stop_reason"], "max_tokens");
        assert_eq!(anthropic["content"][0]["text"], "first");
        assert_eq!(anthropic["content"][1]["text"], "second");
    }

    #[test]
    fn responses_sse_merges_partial_done_items_with_terminal_snapshot() {
        let sse = concat!(
            "data: {\"type\":\"response.output_item.done\",\"output_index\":1,\"item\":{\"id\":\"msg_b\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"B done\"}]}}\n\n",
            "data: {\"type\":\"response.output_item.done\",\"item\":{\"id\":\"msg_c\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"C done\"}]}}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_partial\",\"status\":\"completed\",\"model\":\"gpt-5.6-sol\",\"output\":[{\"id\":\"msg_a\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"A snapshot\"}]},{\"id\":\"msg_b\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"B snapshot\"}]},{\"id\":\"msg_c\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"C snapshot\"}]}],\"usage\":{\"input_tokens\":8,\"output_tokens\":6}}}\n\n"
        );

        let response = responses_sse_to_response_value(sse).unwrap();

        assert_eq!(response["output"].as_array().unwrap().len(), 3);
        assert_eq!(response["output"][0]["id"], "msg_a");
        assert_eq!(response["output"][0]["content"][0]["text"], "A snapshot");
        assert_eq!(response["output"][1]["id"], "msg_b");
        assert_eq!(response["output"][1]["content"][0]["text"], "B done");
        assert_eq!(response["output"][2]["id"], "msg_c");
        assert_eq!(response["output"][2]["content"][0]["text"], "C done");
    }

    #[test]
    fn responses_sse_rejects_indexed_done_with_conflicting_snapshot_id() {
        let sse = concat!(
            "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"id\":\"msg_b\",\"type\":\"message\",\"content\":[]}}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"output\":[{\"id\":\"msg_a\",\"type\":\"message\",\"content\":[]},{\"id\":\"msg_b\",\"type\":\"message\",\"content\":[]}]}}\n\n"
        );

        let error = responses_sse_to_response_value(sse).unwrap_err();
        assert!(error.to_string().contains("conflicting item ids"));
    }

    #[test]
    fn responses_sse_to_response_value_returns_err_on_response_failed() {
        let sse = "event: response.failed\n\
data: {\"type\":\"response.failed\",\"response\":{\"error\":{\"message\":\"upstream blew up\"}}}\n\n";

        let err = responses_sse_to_response_value(sse).unwrap_err();
        match err {
            ProxyError::UpstreamError { status, body } => {
                assert_eq!(status, reqwest::StatusCode::BAD_GATEWAY.as_u16());
                let body: Value = serde_json::from_str(body.as_deref().unwrap()).unwrap();
                assert_eq!(body["type"], "error");
                assert_eq!(body["error"]["message"], "upstream blew up");
            }
            other => panic!("expected UpstreamError, got {other:?}"),
        }
    }

    #[test]
    fn responses_sse_rejects_terminal_event_status_mismatch() {
        let sse = "event: response.completed\n\
data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_failed\",\"status\":\"failed\",\"error\":{\"message\":\"quota\"},\"output\":[]}}\n\n";

        let error = responses_sse_to_response_value(sse).unwrap_err();
        assert!(error.to_string().contains("quota"));
    }

    #[test]
    fn responses_sse_rejects_unsupported_incomplete_reason() {
        let sse = "event: response.incomplete\n\
data: {\"type\":\"response.incomplete\",\"response\":{\"id\":\"resp_filtered\",\"status\":\"incomplete\",\"incomplete_details\":{\"reason\":\"content_filter\"},\"output\":[]}}\n\n";

        let error = responses_sse_to_response_value(sse).unwrap_err();
        assert!(error.to_string().contains("content_filter"));
    }

    #[test]
    fn responses_sse_rejects_conflicting_done_item_identity() {
        let sse = concat!(
            "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"id\":\"msg_a\",\"type\":\"message\",\"content\":[]}}\n\n",
            "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"id\":\"msg_b\",\"type\":\"message\",\"content\":[]}}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"output\":[]}}\n\n"
        );

        let error = responses_sse_to_response_value(sse).unwrap_err();
        assert!(error
            .to_string()
            .contains("Conflicting response.output_item.done"));
    }

    #[test]
    fn responses_sse_rejects_content_after_terminal_event() {
        let sse = concat!(
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"output\":[]}}\n\n",
            "data: {\"type\":\"response.output_text.done\",\"output_index\":0,\"item_id\":\"msg_a\",\"text\":\"late\"}\n\n"
        );

        let error = responses_sse_to_response_value(sse).unwrap_err();
        assert!(error.to_string().contains("after terminal"));
    }

    #[test]
    fn responses_sse_rejects_conflicting_event_and_data_type() {
        let sse = "event: response.completed\n\
data: {\"type\":\"response.incomplete\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"output\":[]}}\n\n";

        let error = responses_sse_to_response_value(sse).unwrap_err();
        assert!(error
            .to_string()
            .contains("Conflicting upstream SSE event names"));
    }

    #[test]
    fn responses_sse_accepts_exact_duplicate_done_and_terminal_events() {
        let done = "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"id\":\"msg_a\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"A\"}]}}\n\n";
        let terminal = "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"output\":[]}}\n\n";
        let sse = format!("{done}{done}{terminal}{terminal}");

        let response = responses_sse_to_response_value(&sse).unwrap();
        assert_eq!(response["output"].as_array().unwrap().len(), 1);
        assert_eq!(response["output"][0]["id"], "msg_a");
    }

    #[test]
    fn responses_sse_to_response_value_errors_when_no_completed_event() {
        let sse = "event: response.output_item.done\n\
data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"message\"}}\n\n";

        assert!(responses_sse_to_response_value(sse).is_err());
    }

    #[test]
    fn web_search_sse_aggregation_rejects_terminal_snapshot_without_done_item() {
        let policy = WebSearchRequestPolicy::from_anthropic_request(&json!({
            "tools": [{"type": "web_search_20250305", "name": "web_search"}]
        }))
        .unwrap()
        .unwrap();
        let sse = r#"event: response.completed
data: {"type":"response.completed","response":{"id":"resp_1","status":"completed","model":"gpt-5.6-sol","output":[{"id":"ws_1","type":"web_search_call","status":"completed","action":{"type":"search","query":"q","sources":[]}}]}}

"#;

        let error = responses_sse_to_web_search_response_value(sse, &policy).unwrap_err();
        assert!(error.to_string().contains("without output_item.done"));
    }

    #[test]
    fn web_search_sse_aggregation_rejects_error_event_before_completed() {
        let policy = WebSearchRequestPolicy::from_anthropic_request(&json!({
            "tools": [{"type": "web_search_20250305", "name": "web_search"}]
        }))
        .unwrap()
        .unwrap();
        let sse = r#"event: error
data: {"type":"error","error":{"type":"server_error","message":"redacted"}}

event: response.completed
data: {"type":"response.completed","response":{"id":"resp_1","status":"completed","model":"gpt-5.6-sol","output":[]}}

"#;

        let error = responses_sse_to_web_search_response_value(sse, &policy).unwrap_err();
        match error {
            ProxyError::UpstreamError { body, .. } => {
                let body: Value = serde_json::from_str(body.as_deref().unwrap()).unwrap();
                assert_eq!(body["type"], "error");
                assert_eq!(body["error"]["message"], "redacted");
            }
            other => panic!("expected UpstreamError, got {other:?}"),
        }
    }
}
