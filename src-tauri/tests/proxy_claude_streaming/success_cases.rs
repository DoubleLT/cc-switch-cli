use axum::{
    body::Body, extract::State, http::HeaderMap, response::IntoResponse, routing::post, Json,
    Router,
};
use cc_switch_lib::{Database, Provider, ProviderMeta, ProxyService};
use serde_json::{json, Value};
use serial_test::serial;
use std::sync::Arc;

use crate::helpers::{
    bind_test_listener, handle_slow_streaming_chat, handle_streaming_chat, read_proxy_status,
    record_upstream_request, set_claude_proxy_port_to_ephemeral, wait_for_proxy_status,
    UpstreamState,
};

async fn handle_streaming_tool_calls_interleaved(
    State(state): State<UpstreamState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> impl IntoResponse {
    record_upstream_request(&state, &headers, body).await;

    let stream = async_stream::stream! {
        yield Ok::<_, std::io::Error>(bytes::Bytes::from_static(
            b"data: {\"id\":\"chatcmpl-tool\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_0\",\"type\":\"function\",\"function\":{\"name\":\"first_tool\"}}]}}]}\n\n",
        ));
        yield Ok::<_, std::io::Error>(bytes::Bytes::from_static(
            b"data: {\"id\":\"chatcmpl-tool\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":1,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"second_tool\"}}]}}]}\n\n",
        ));
        yield Ok::<_, std::io::Error>(bytes::Bytes::from_static(
            b"data: {\"id\":\"chatcmpl-tool\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":1,\"function\":{\"arguments\":\"{\\\"b\\\":2}\"}}]}}]}\n\n",
        ));
        yield Ok::<_, std::io::Error>(bytes::Bytes::from_static(
            b"data: {\"id\":\"chatcmpl-tool\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"a\\\":1}\"}}]}}]}\n\n",
        ));
        yield Ok::<_, std::io::Error>(bytes::Bytes::from_static(
            b"data: {\"id\":\"chatcmpl-tool\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
        ));
        yield Ok::<_, std::io::Error>(bytes::Bytes::from_static(b"data: [DONE]\n\n"));
    };

    (
        axum::http::StatusCode::OK,
        [("content-type", "text/event-stream")],
        Body::from_stream(stream),
    )
}

async fn handle_streaming_responses(
    State(state): State<UpstreamState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> impl IntoResponse {
    record_upstream_request(&state, &headers, body).await;

    let stream = async_stream::stream! {
        yield Ok::<_, std::io::Error>(bytes::Bytes::from_static(
            b"event: response.created\ndata: {\"response\":{\"id\":\"resp-stream\",\"model\":\"gpt-4.1-mini\",\"usage\":{\"input_tokens\":11,\"output_tokens\":0}}}\n\n",
        ));
        yield Ok::<_, std::io::Error>(bytes::Bytes::from_static(
            b"event: response.content_part.added\ndata: {\"item_id\":\"msg_1\",\"content_index\":0,\"part\":{\"type\":\"output_text\"}}\n\n",
        ));
        yield Ok::<_, std::io::Error>(bytes::Bytes::from_static(
            b"event: response.output_text.delta\ndata: {\"item_id\":\"msg_1\",\"content_index\":0,\"delta\":\"hello from responses\"}\n\n",
        ));
        yield Ok::<_, std::io::Error>(bytes::Bytes::from_static(
            b"event: response.content_part.done\ndata: {\"item_id\":\"msg_1\",\"content_index\":0}\n\n",
        ));
        yield Ok::<_, std::io::Error>(bytes::Bytes::from_static(
            b"event: response.completed\ndata: {\"response\":{\"status\":\"completed\",\"usage\":{\"input_tokens\":11,\"output_tokens\":7}}}\n\n",
        ));
    };

    (
        axum::http::StatusCode::OK,
        [("content-type", "text/event-stream")],
        Body::from_stream(stream),
    )
}

async fn handle_stream_only_responses(
    State(state): State<UpstreamState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> axum::response::Response {
    record_upstream_request(&state, &headers, body.clone()).await;
    if body.get("stream").and_then(Value::as_bool) != Some(true) {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            Json(json!({"error": "Stream must be set to true"})),
        )
            .into_response();
    }

    let sse = concat!(
        "event: response.output_item.done\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"id\":\"msg_non_stream\",\"type\":\"message\",\"status\":\"completed\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"non-stream bridge ok\",\"annotations\":[]}]}}\n\n",
        "event: response.completed\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_non_stream\",\"model\":\"gpt-5.6-sol\",\"status\":\"completed\",\"output\":[],\"usage\":{\"input_tokens\":9,\"output_tokens\":4}}}\n\n"
    );
    (
        axum::http::StatusCode::OK,
        [("content-type", "text/event-stream")],
        sse,
    )
        .into_response()
}

async fn handle_streaming_responses_web_search(
    State(state): State<UpstreamState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> impl IntoResponse {
    record_upstream_request(&state, &headers, body).await;
    let sse = concat!(
        "event: response.output_item.done\n",
        "data: {\"output_index\":1,\"item\":{\"id\":\"ws_search\",\"type\":\"web_search_call\",\"status\":\"completed\",\"action\":{\"type\":\"search\",\"query\":\"Rust\",\"sources\":[{\"type\":\"url\",\"url\":\"https://example.com/rust\"}]}}}\n\n",
        "event: response.output_item.done\n",
        "data: {\"output_index\":3,\"item\":{\"id\":\"ws_open\",\"type\":\"web_search_call\",\"status\":\"completed\",\"action\":{\"type\":\"open_page\",\"url\":\"https://example.com/rust\"}}}\n\n",
        "event: response.output_item.done\n",
        "data: {\"output_index\":5,\"item\":{\"id\":\"ws_find\",\"type\":\"web_search_call\",\"status\":\"completed\",\"action\":{\"type\":\"find_in_page\",\"url\":\"https://example.com/rust\",\"pattern\":\"ownership\"}}}\n\n",
        "event: response.output_item.done\n",
        "data: {\"output_index\":6,\"item\":{\"id\":\"msg_search\",\"type\":\"message\",\"status\":\"completed\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"A😊中B\",\"annotations\":[{\"type\":\"url_citation\",\"url\":\"https://example.com/rust\",\"title\":\"Rust ownership\",\"start_index\":1,\"end_index\":3}]}]}}\n\n",
        "event: response.completed\n",
        "data: {\"response\":{\"id\":\"resp_search\",\"model\":\"gpt-5.6-sol\",\"status\":\"completed\",\"output\":[],\"usage\":{\"input_tokens\":12,\"output_tokens\":5}}}\n\n"
    );
    (
        axum::http::StatusCode::OK,
        [("content-type", "text/event-stream")],
        sse,
    )
}

async fn handle_buffered_chat_fallback(
    State(state): State<UpstreamState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> impl IntoResponse {
    record_upstream_request(&state, &headers, body).await;

    (
        axum::http::StatusCode::OK,
        Json(json!({
            "id": "chatcmpl-buffered-fallback",
            "object": "chat.completion",
            "created": 123,
            "model": "gpt-4o-mini",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "hello back"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18
            }
        })),
    )
}

fn parse_sse_events(body: &str) -> Vec<Value> {
    body.split("\n\n")
        .filter_map(|block| {
            let data = block.lines().find_map(|line| line.strip_prefix("data: "))?;
            serde_json::from_str::<Value>(data).ok()
        })
        .collect()
}

#[tokio::test]
#[serial]
async fn stream_openai_chat_transforms_sse_and_maps_model() {
    let upstream_state = UpstreamState::default();
    let upstream_router = Router::new()
        .route("/v1/chat/completions", post(handle_streaming_chat))
        .with_state(upstream_state.clone());

    let upstream_listener = bind_test_listener().await;
    let upstream_addr = upstream_listener
        .local_addr()
        .expect("read upstream address");
    let upstream_handle = tokio::spawn(async move {
        let _ = axum::serve(upstream_listener, upstream_router).await;
    });

    let db = Arc::new(Database::memory().expect("create memory database"));
    let provider = Provider {
        id: "claude-openai-chat-stream".to_string(),
        name: "Claude OpenAI Chat Stream".to_string(),
        settings_config: json!({
            "env": {
                "ANTHROPIC_BASE_URL": format!("http://{}", upstream_addr),
                "ANTHROPIC_API_KEY": "sk-test-claude",
                "ANTHROPIC_DEFAULT_SONNET_MODEL": "mapped-sonnet"
            }
        }),
        website_url: None,
        category: Some("claude".to_string()),
        created_at: None,
        sort_index: None,
        notes: None,
        meta: Some(ProviderMeta {
            api_format: Some("openai_chat".to_string()),
            ..ProviderMeta::default()
        }),
        icon: None,
        icon_color: None,
        in_failover_queue: true,
    };
    db.save_provider("claude", &provider)
        .expect("save test provider");
    db.set_current_provider("claude", &provider.id)
        .expect("set current provider");

    set_claude_proxy_port_to_ephemeral(&db).await;
    let service = ProxyService::new(db);

    let mut config = service.get_config().await.expect("read proxy config");
    config.listen_port = 0;
    let proxy = service
        .start_with_runtime_config(config)
        .await
        .expect("start proxy service");
    let client = reqwest::Client::new();
    let response = client
        .post(format!(
            "http://{}:{}/v1/messages",
            proxy.address, proxy.port
        ))
        .json(&json!({
            "model": "claude-3-7-sonnet",
            "stream": true,
            "max_tokens": 64,
            "messages": [{
                "role": "user",
                "content": "hello"
            }]
        }))
        .send()
        .await
        .expect("send request to proxy");

    assert!(
        response.status().is_success(),
        "proxy should return streaming success"
    );
    assert_eq!(
        response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok()),
        Some("text/event-stream")
    );

    let in_flight = read_proxy_status(&client, &proxy.address, proxy.port).await;
    assert_eq!(in_flight.total_requests, 1);
    assert_eq!(in_flight.active_connections, 1);
    assert_eq!(in_flight.success_requests, 0);
    assert_eq!(in_flight.failed_requests, 0);
    assert!(in_flight.last_error.is_none());

    let body = response.text().await.expect("read streaming response body");
    assert!(body.contains("event: message_start"));
    assert!(body.contains("event: content_block_start"));
    assert!(body.contains("event: content_block_delta"));
    assert!(body.contains("hello"));
    assert!(body.contains("event: message_delta"));
    assert!(body.contains("\"input_tokens\":11"));
    assert!(body.contains("\"output_tokens\":7"));
    assert!(body.contains("event: message_stop"));

    let completed = wait_for_proxy_status(&client, &proxy.address, proxy.port, |status| {
        status.active_connections == 0 && status.success_requests == 1
    })
    .await;
    assert_eq!(completed.failed_requests, 0);
    assert!(completed.last_error.is_none());
    assert_eq!(completed.success_rate, 100.0);

    let upstream_body = upstream_state
        .request_body
        .lock()
        .await
        .clone()
        .expect("upstream should receive request body");
    assert_eq!(
        upstream_body.get("stream").and_then(|v| v.as_bool()),
        Some(true)
    );
    assert_eq!(
        upstream_body.get("model").and_then(|v| v.as_str()),
        Some("mapped-sonnet")
    );
    assert_eq!(
        upstream_state.authorization.lock().await.as_deref(),
        Some("Bearer sk-test-claude")
    );
    assert_eq!(upstream_state.api_key.lock().await.as_deref(), None);

    service.stop().await.expect("stop proxy service");
    upstream_handle.abort();
}

#[tokio::test]
#[serial]
async fn stream_openai_chat_buffered_json_fallback_marks_request_success() {
    let upstream_state = UpstreamState::default();
    let upstream_router = Router::new()
        .route("/v1/chat/completions", post(handle_buffered_chat_fallback))
        .with_state(upstream_state.clone());

    let upstream_listener = bind_test_listener().await;
    let upstream_addr = upstream_listener
        .local_addr()
        .expect("read upstream address");
    let upstream_handle = tokio::spawn(async move {
        let _ = axum::serve(upstream_listener, upstream_router).await;
    });

    let db = Arc::new(Database::memory().expect("create memory database"));
    let provider = Provider {
        id: "claude-openai-chat-stream-buffered-fallback".to_string(),
        name: "Claude OpenAI Chat Stream Buffered Fallback".to_string(),
        settings_config: json!({
            "env": {
                "ANTHROPIC_BASE_URL": format!("http://{}", upstream_addr),
                "ANTHROPIC_API_KEY": "sk-test-claude"
            }
        }),
        website_url: None,
        category: Some("claude".to_string()),
        created_at: None,
        sort_index: None,
        notes: None,
        meta: Some(ProviderMeta {
            api_format: Some("openai_chat".to_string()),
            ..ProviderMeta::default()
        }),
        icon: None,
        icon_color: None,
        in_failover_queue: true,
    };
    db.save_provider("claude", &provider)
        .expect("save test provider");
    db.set_current_provider("claude", &provider.id)
        .expect("set current provider");

    set_claude_proxy_port_to_ephemeral(&db).await;
    let service = ProxyService::new(db);

    let mut config = service.get_config().await.expect("read proxy config");
    config.listen_port = 0;
    let proxy = service
        .start_with_runtime_config(config)
        .await
        .expect("start proxy service");
    let client = reqwest::Client::new();
    let response = client
        .post(format!(
            "http://{}:{}/v1/messages",
            proxy.address, proxy.port
        ))
        .json(&json!({
            "model": "claude-3-7-sonnet",
            "stream": true,
            "max_tokens": 64,
            "messages": [{
                "role": "user",
                "content": "hello"
            }]
        }))
        .send()
        .await
        .expect("send request to proxy");

    assert_eq!(response.status(), reqwest::StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok()),
        Some("application/json")
    );

    let body: Value = response.json().await.expect("parse buffered fallback body");
    assert_eq!(
        body.get("type").and_then(|value| value.as_str()),
        Some("message")
    );
    assert_eq!(
        body.get("role").and_then(|value| value.as_str()),
        Some("assistant")
    );
    assert_eq!(
        body.pointer("/content/0/type")
            .and_then(|value| value.as_str()),
        Some("text")
    );
    assert_eq!(
        body.pointer("/content/0/text")
            .and_then(|value| value.as_str()),
        Some("hello back")
    );
    assert_eq!(
        body.pointer("/usage/input_tokens")
            .and_then(|value| value.as_u64()),
        Some(11)
    );
    assert_eq!(
        body.pointer("/usage/output_tokens")
            .and_then(|value| value.as_u64()),
        Some(7)
    );

    let completed = wait_for_proxy_status(&client, &proxy.address, proxy.port, |status| {
        status.active_connections == 0 && status.success_requests == 1
    })
    .await;
    assert_eq!(completed.total_requests, 1);
    assert_eq!(completed.failed_requests, 0);
    assert!(completed.last_error.is_none());

    let upstream_body = upstream_state
        .request_body
        .lock()
        .await
        .clone()
        .expect("upstream should receive request body");
    assert_eq!(
        upstream_body
            .get("stream")
            .and_then(|value| value.as_bool()),
        Some(true)
    );

    service.stop().await.expect("stop proxy service");
    upstream_handle.abort();
}

#[tokio::test]
#[serial]
async fn stream_openai_chat_tool_calls_interleaved_transform_to_stable_anthropic_blocks() {
    let upstream_state = UpstreamState::default();
    let upstream_router = Router::new()
        .route(
            "/v1/chat/completions",
            post(handle_streaming_tool_calls_interleaved),
        )
        .with_state(upstream_state.clone());

    let upstream_listener = bind_test_listener().await;
    let upstream_addr = upstream_listener
        .local_addr()
        .expect("read upstream address");
    let upstream_handle = tokio::spawn(async move {
        let _ = axum::serve(upstream_listener, upstream_router).await;
    });

    let db = Arc::new(Database::memory().expect("create memory database"));
    let provider = Provider {
        id: "claude-openai-chat-stream-tools".to_string(),
        name: "Claude OpenAI Chat Stream Tools".to_string(),
        settings_config: json!({
            "env": {
                "ANTHROPIC_BASE_URL": format!("http://{}", upstream_addr),
                "ANTHROPIC_API_KEY": "sk-test-claude",
                "ANTHROPIC_DEFAULT_SONNET_MODEL": "mapped-sonnet"
            }
        }),
        website_url: None,
        category: Some("claude".to_string()),
        created_at: None,
        sort_index: None,
        notes: None,
        meta: Some(ProviderMeta {
            api_format: Some("openai_chat".to_string()),
            ..ProviderMeta::default()
        }),
        icon: None,
        icon_color: None,
        in_failover_queue: true,
    };
    db.save_provider("claude", &provider)
        .expect("save test provider");
    db.set_current_provider("claude", &provider.id)
        .expect("set current provider");

    set_claude_proxy_port_to_ephemeral(&db).await;
    let service = ProxyService::new(db);

    let mut config = service.get_config().await.expect("read proxy config");
    config.listen_port = 0;
    let proxy = service
        .start_with_runtime_config(config)
        .await
        .expect("start proxy service");
    let client = reqwest::Client::new();
    let response = client
        .post(format!(
            "http://{}:{}/v1/messages",
            proxy.address, proxy.port
        ))
        .json(&json!({
            "model": "claude-3-7-sonnet",
            "stream": true,
            "max_tokens": 64,
            "messages": [{
                "role": "user",
                "content": "use tools"
            }]
        }))
        .send()
        .await
        .expect("send request to proxy");

    assert!(response.status().is_success());
    assert_eq!(
        response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok()),
        Some("text/event-stream")
    );

    let body = response.text().await.expect("read streaming response body");
    let events = parse_sse_events(&body);

    let mut tool_index_by_call: std::collections::HashMap<String, u64> =
        std::collections::HashMap::new();
    for event in &events {
        if event["type"] == "content_block_start" && event["content_block"]["type"] == "tool_use" {
            assert_eq!(event.pointer("/content_block/input"), Some(&json!({})));
            if let (Some(call_id), Some(index)) = (
                event.pointer("/content_block/id").and_then(|v| v.as_str()),
                event.get("index").and_then(|v| v.as_u64()),
            ) {
                tool_index_by_call.insert(call_id.to_string(), index);
            }
        }
    }

    assert_eq!(tool_index_by_call.len(), 2);
    assert_ne!(
        tool_index_by_call.get("call_0"),
        tool_index_by_call.get("call_1")
    );

    let deltas: Vec<(u64, String)> = events
        .iter()
        .filter(|event| {
            event["type"] == "content_block_delta" && event["delta"]["type"] == "input_json_delta"
        })
        .filter_map(|event| {
            let index = event.get("index").and_then(|v| v.as_u64())?;
            let partial_json = event
                .pointer("/delta/partial_json")
                .and_then(|v| v.as_str())?
                .to_string();
            Some((index, partial_json))
        })
        .collect();

    let second_idx = deltas
        .iter()
        .find_map(|(index, payload)| (payload == "{\"b\":2}").then_some(*index))
        .expect("second tool delta index");
    let first_idx = deltas
        .iter()
        .find_map(|(index, payload)| (payload == "{\"a\":1}").then_some(*index))
        .expect("first tool delta index");

    assert_eq!(second_idx, *tool_index_by_call.get("call_1").unwrap());
    assert_eq!(first_idx, *tool_index_by_call.get("call_0").unwrap());
    let message_delta = events
        .iter()
        .find(|event| event["type"] == "message_delta")
        .expect("message_delta event");
    assert_eq!(message_delta["delta"]["stop_reason"], "tool_use");
    assert!(message_delta["usage"].is_object());
    assert_eq!(message_delta["usage"]["output_tokens"], 0);

    service.stop().await.expect("stop proxy service");
    upstream_handle.abort();
}

#[tokio::test]
#[serial]
async fn proxy_claude_openai_responses_streaming_transforms_sse() {
    let upstream_state = UpstreamState::default();
    let upstream_router = Router::new()
        .route("/v1/responses", post(handle_streaming_responses))
        .with_state(upstream_state.clone());

    let upstream_listener = bind_test_listener().await;
    let upstream_addr = upstream_listener
        .local_addr()
        .expect("read upstream address");
    let upstream_handle = tokio::spawn(async move {
        let _ = axum::serve(upstream_listener, upstream_router).await;
    });

    let db = Arc::new(Database::memory().expect("create memory database"));
    let provider = Provider {
        id: "claude-openai-responses-stream".to_string(),
        name: "Claude OpenAI Responses Stream".to_string(),
        settings_config: json!({
            "env": {
                "ANTHROPIC_BASE_URL": format!("http://{}", upstream_addr),
                "ANTHROPIC_API_KEY": "sk-test-claude"
            }
        }),
        website_url: None,
        category: Some("claude".to_string()),
        created_at: None,
        sort_index: None,
        notes: None,
        meta: Some(ProviderMeta {
            api_format: Some("openai_responses".to_string()),
            ..ProviderMeta::default()
        }),
        icon: None,
        icon_color: None,
        in_failover_queue: true,
    };
    db.save_provider("claude", &provider)
        .expect("save test provider");
    db.set_current_provider("claude", &provider.id)
        .expect("set current provider");

    set_claude_proxy_port_to_ephemeral(&db).await;
    let service = ProxyService::new(db);

    let mut config = service.get_config().await.expect("read proxy config");
    config.listen_port = 0;
    let proxy = service
        .start_with_runtime_config(config)
        .await
        .expect("start proxy service");
    let client = reqwest::Client::new();
    let response = client
        .post(format!(
            "http://{}:{}/v1/messages",
            proxy.address, proxy.port
        ))
        .json(&json!({
            "model": "claude-3-7-sonnet",
            "stream": true,
            "max_tokens": 64,
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": "hello"
                }]
            }]
        }))
        .send()
        .await
        .expect("send request to proxy");

    assert!(response.status().is_success());
    assert_eq!(
        response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok()),
        Some("text/event-stream")
    );

    let body = response.text().await.expect("read streaming response body");
    assert!(body.contains("event: message_start"));
    assert!(body.contains("event: content_block_start"));
    assert!(body.contains("event: content_block_delta"));
    assert!(body.contains("hello from responses"));
    assert!(body.contains("event: message_delta"));
    assert!(body.contains("\"input_tokens\":11"));
    assert!(body.contains("\"output_tokens\":7"));
    assert!(body.contains("event: message_stop"));

    let upstream_body = upstream_state
        .request_body
        .lock()
        .await
        .clone()
        .expect("upstream should receive request body");
    assert_eq!(
        upstream_body
            .pointer("/input/0/role")
            .and_then(|v| v.as_str()),
        Some("user")
    );
    assert_eq!(
        upstream_body
            .pointer("/input/0/content/0/type")
            .and_then(|v| v.as_str()),
        Some("input_text")
    );
    assert_eq!(
        upstream_state.authorization.lock().await.as_deref(),
        Some("Bearer sk-test-claude")
    );
    assert_eq!(upstream_state.api_key.lock().await.as_deref(), None);

    service.stop().await.expect("stop proxy service");
    upstream_handle.abort();
}

#[tokio::test]
#[serial]
async fn proxy_claude_openai_responses_non_stream_uses_streaming_upstream() {
    let upstream_state = UpstreamState::default();
    let upstream_router = Router::new()
        .route("/v1/responses", post(handle_stream_only_responses))
        .with_state(upstream_state.clone());
    let upstream_listener = bind_test_listener().await;
    let upstream_addr = upstream_listener.local_addr().unwrap();
    let upstream_handle = tokio::spawn(async move {
        let _ = axum::serve(upstream_listener, upstream_router).await;
    });

    let db = Arc::new(Database::memory().expect("create memory database"));
    let provider = Provider {
        id: "claude-openai-responses-non-stream".to_string(),
        name: "Claude OpenAI Responses Non-stream".to_string(),
        settings_config: json!({
            "env": {
                "ANTHROPIC_BASE_URL": format!("http://{}", upstream_addr),
                "ANTHROPIC_API_KEY": "sk-test-claude"
            }
        }),
        website_url: None,
        category: Some("claude".to_string()),
        created_at: None,
        sort_index: None,
        notes: None,
        meta: Some(ProviderMeta {
            api_format: Some("openai_responses".to_string()),
            ..ProviderMeta::default()
        }),
        icon: None,
        icon_color: None,
        in_failover_queue: true,
    };
    db.save_provider("claude", &provider).unwrap();
    db.set_current_provider("claude", &provider.id).unwrap();
    set_claude_proxy_port_to_ephemeral(&db).await;
    let service = ProxyService::new(db);
    let mut config = service.get_config().await.unwrap();
    config.listen_port = 0;
    let proxy = service.start_with_runtime_config(config).await.unwrap();

    let response = reqwest::Client::new()
        .post(format!(
            "http://{}:{}/v1/messages",
            proxy.address, proxy.port
        ))
        .json(&json!({
            "model": "claude-haiku-4-5",
            "stream": false,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "verify a domain"}]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), reqwest::StatusCode::OK);
    assert!(response
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.starts_with("application/json")));
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["type"], "message");
    assert_eq!(body["content"][0]["type"], "text");
    assert_eq!(body["content"][0]["text"], "non-stream bridge ok");
    assert_eq!(body["usage"]["input_tokens"], 9);
    assert_eq!(body["usage"]["output_tokens"], 4);

    let upstream = upstream_state.request_body.lock().await.clone().unwrap();
    assert_eq!(upstream["stream"], true);
    assert_eq!(upstream["model"], "claude-haiku-4-5");

    service.stop().await.unwrap();
    upstream_handle.abort();
}

#[tokio::test]
#[serial]
async fn proxy_claude_web_search_uses_buffered_responses_bridge() {
    let upstream_state = UpstreamState::default();
    let upstream_router = Router::new()
        .route("/v1/responses", post(handle_streaming_responses_web_search))
        .with_state(upstream_state.clone());
    let upstream_listener = bind_test_listener().await;
    let upstream_addr = upstream_listener.local_addr().unwrap();
    let upstream_handle = tokio::spawn(async move {
        let _ = axum::serve(upstream_listener, upstream_router).await;
    });

    let db = Arc::new(Database::memory().expect("create memory database"));
    let provider = Provider {
        id: "claude-web-search-buffered".to_string(),
        name: "Claude WebSearch Buffered".to_string(),
        settings_config: json!({
            "env": {
                "ANTHROPIC_BASE_URL": format!("http://{}", upstream_addr),
                "ANTHROPIC_API_KEY": "sk-test-claude"
            }
        }),
        website_url: None,
        category: Some("claude".to_string()),
        created_at: None,
        sort_index: None,
        notes: None,
        meta: Some(ProviderMeta {
            api_format: Some("openai_responses".to_string()),
            ..ProviderMeta::default()
        }),
        icon: None,
        icon_color: None,
        in_failover_queue: true,
    };
    db.save_provider("claude", &provider).unwrap();
    db.set_current_provider("claude", &provider.id).unwrap();
    set_claude_proxy_port_to_ephemeral(&db).await;
    let service = ProxyService::new(db);
    let mut config = service.get_config().await.unwrap();
    config.listen_port = 0;
    let proxy = service.start_with_runtime_config(config).await.unwrap();

    let response = reqwest::Client::new()
        .post(format!(
            "http://{}:{}/v1/messages",
            proxy.address, proxy.port
        ))
        .json(&json!({
            "model": "gpt-5.6-sol",
            "stream": true,
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "Search Rust"}],
            "tools": [{
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 1,
                "allowed_domains": ["example.com"],
                "blocked_domains": []
            }],
            "tool_choice": {"type": "auto"}
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), reqwest::StatusCode::OK);
    let body = response.text().await.unwrap();
    let events = parse_sse_events(&body);
    let block_types = events
        .iter()
        .filter_map(|event| event.pointer("/content_block/type").and_then(Value::as_str))
        .collect::<Vec<_>>();
    assert_eq!(
        block_types,
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
    let result = events
        .iter()
        .find(|event| {
            event
                .pointer("/content_block/tool_use_id")
                .and_then(Value::as_str)
                == Some("ws_search")
        })
        .unwrap();
    assert_eq!(
        result["content_block"]["content"][0]["title"],
        "Rust ownership"
    );
    let citation = events
        .iter()
        .find_map(|event| event.pointer("/delta/citation"))
        .unwrap();
    assert_eq!(citation["cited_text"], "😊中");
    let terminal = events
        .iter()
        .find(|event| event.get("type").and_then(Value::as_str) == Some("message_delta"))
        .unwrap();
    assert_eq!(terminal["delta"]["stop_reason"], "end_turn");
    assert_eq!(
        terminal["usage"]["server_tool_use"]["web_search_requests"],
        1
    );

    let upstream = upstream_state.request_body.lock().await.clone().unwrap();
    assert_eq!(upstream["tools"][0]["type"], "web_search");
    assert_eq!(
        upstream["tools"][0]["filters"]["allowed_domains"],
        json!(["example.com"])
    );
    assert!(upstream.get("max_tool_calls").is_none());
    assert!(upstream["include"]
        .as_array()
        .is_some_and(|values| values.contains(&json!("web_search_call.action.sources"))));

    service.stop().await.unwrap();
    upstream_handle.abort();
}

#[tokio::test]
#[serial]
async fn proxy_claude_streaming_bypasses_first_byte_timeout_when_failover_disabled() {
    let upstream_state = UpstreamState::default();
    let upstream_router = Router::new()
        .route("/v1/chat/completions", post(handle_slow_streaming_chat))
        .with_state(upstream_state.clone());

    let upstream_listener = bind_test_listener().await;
    let upstream_addr = upstream_listener
        .local_addr()
        .expect("read upstream address");
    let upstream_handle = tokio::spawn(async move {
        let _ = axum::serve(upstream_listener, upstream_router).await;
    });

    let db = Arc::new(Database::memory().expect("create memory database"));
    let provider = Provider {
        id: "claude-openai-chat-stream-timeout-bypass".to_string(),
        name: "Claude OpenAI Chat Stream Timeout Bypass".to_string(),
        settings_config: json!({
            "env": {
                "ANTHROPIC_BASE_URL": format!("http://{}", upstream_addr),
                "ANTHROPIC_API_KEY": "sk-test-claude"
            }
        }),
        website_url: None,
        category: Some("claude".to_string()),
        created_at: None,
        sort_index: None,
        notes: None,
        meta: Some(ProviderMeta {
            api_format: Some("openai_chat".to_string()),
            ..ProviderMeta::default()
        }),
        icon: None,
        icon_color: None,
        in_failover_queue: true,
    };
    db.save_provider("claude", &provider)
        .expect("save test provider");
    db.set_current_provider("claude", &provider.id)
        .expect("set current provider");

    let mut claude_proxy = db
        .get_proxy_config_for_app("claude")
        .await
        .expect("get claude proxy config");
    claude_proxy.auto_failover_enabled = false;
    claude_proxy.max_retries = 0;
    claude_proxy.streaming_first_byte_timeout = 1;
    db.set_app_proxy_preferred_port("claude", 0)
        .expect("update claude app proxy port");
    db.update_proxy_config_for_app(claude_proxy)
        .await
        .expect("update claude proxy config");

    set_claude_proxy_port_to_ephemeral(&db).await;
    let service = ProxyService::new(db);

    let mut config = service.get_config().await.expect("read proxy config");
    config.listen_port = 0;
    let proxy = service
        .start_with_runtime_config(config)
        .await
        .expect("start proxy service");
    let client = reqwest::Client::new();
    let response = client
        .post(format!(
            "http://{}:{}/v1/messages",
            proxy.address, proxy.port
        ))
        .json(&json!({
            "model": "claude-3-7-sonnet",
            "stream": true,
            "max_tokens": 64,
            "messages": [{
                "role": "user",
                "content": "hello"
            }]
        }))
        .send()
        .await
        .expect("send request to proxy");

    assert!(response.status().is_success());
    let body = response.text().await.expect("read streaming response body");
    assert!(body.contains("late"));
    assert!(body.contains("event: message_stop"));

    service.stop().await.expect("stop proxy service");
    upstream_handle.abort();
}
