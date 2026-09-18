use std::{
    sync::atomic::Ordering,
    time::{Duration, Instant},
};

use axum::http::{HeaderMap, StatusCode};
use bytes::Bytes;
use futures::StreamExt;
use serde_json::{json, Value};

use super::{
    claude_provider, claude_request_body, closed_base_url, codex_provider,
    spawn_delayed_body_upstream, spawn_delayed_scripted_streaming_upstream,
    spawn_delayed_scripted_upstream, spawn_failing_body_upstream, spawn_mock_upstream,
    spawn_scripted_streaming_upstream, test_router, ScriptedStreamingBody,
};
use crate::{
    app_config::AppType,
    provider::ProviderMeta,
    proxy::{
        error::ProxyError,
        forwarder::{ForwardOptions, RequestForwarder, StreamingResponse},
        response::is_sse_response,
        types::RectifierConfig,
    },
};

#[tokio::test]
async fn codex_anthropic_empty_success_retries_before_client_commit() {
    const EMPTY: &str = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_empty\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-sonnet-5\",\"content\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\n",
        "event: message_delta\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":0}}\n\n",
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n\n"
    );
    const SUCCESS: &str = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_ok\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-sonnet-5\",\"content\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\n",
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"ok\"}}\n\n",
        "event: content_block_stop\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: message_delta\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\n",
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n\n"
    );
    let (base_url, hits, _bodies, server) = spawn_scripted_streaming_upstream(vec![
        (StatusCode::OK, ScriptedStreamingBody::Sse(EMPTY)),
        (StatusCode::OK, ScriptedStreamingBody::Sse(SUCCESS)),
    ])
    .await;
    let mut provider = codex_provider("p1", &base_url, false);
    provider.meta = Some(ProviderMeta {
        api_format: Some("anthropic".to_string()),
        ..Default::default()
    });
    let (db, router) = test_router().await;
    let forwarder = RequestForwarder::new(router).expect("create forwarder");
    db.save_provider("codex", &provider)
        .expect("save provider for health tracking");

    let result = forwarder
        .forward_response(
            &AppType::Codex,
            "/v1/responses",
            json!({"model": "gpt-5.6-terra", "input": "continue", "stream": true}),
            &HeaderMap::new(),
            vec![provider],
            ForwardOptions {
                max_retries: 2,
                request_timeout: Some(Duration::from_secs(5)),
                bypass_circuit_breaker: true,
            },
            RectifierConfig::default(),
        )
        .await
        .expect("empty Anthropic completion should retry before client commit");

    assert_eq!(result.response.status(), StatusCode::OK);
    assert_eq!(hits.count.load(Ordering::SeqCst), 2);
    server.abort();
}

#[tokio::test]
async fn codex_anthropic_empty_json_retries_before_client_commit() {
    let empty = json!({
        "id": "msg_empty",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5",
        "content": [],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 0}
    });
    let success = json!({
        "id": "msg_ok",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5",
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1}
    });
    let (base_url, hits, _bodies, server) = spawn_scripted_streaming_upstream(vec![
        (StatusCode::OK, ScriptedStreamingBody::Json(empty)),
        (StatusCode::OK, ScriptedStreamingBody::Json(success)),
    ])
    .await;
    let mut provider = codex_provider("p1", &base_url, false);
    provider.meta = Some(ProviderMeta {
        api_format: Some("anthropic".to_string()),
        ..Default::default()
    });
    let (db, router) = test_router().await;
    let forwarder = RequestForwarder::new(router).expect("create forwarder");
    db.save_provider("codex", &provider)
        .expect("save provider for health tracking");

    let result = forwarder
        .forward_response(
            &AppType::Codex,
            "/v1/responses",
            json!({"model": "gpt-5.6-terra", "input": "continue", "stream": true}),
            &HeaderMap::new(),
            vec![provider],
            ForwardOptions {
                max_retries: 2,
                request_timeout: Some(Duration::from_secs(5)),
                bypass_circuit_breaker: true,
            },
            RectifierConfig::default(),
        )
        .await
        .expect("empty Anthropic JSON should retry before client commit");

    assert_eq!(result.response.status(), StatusCode::OK);
    assert_eq!(hits.count.load(Ordering::SeqCst), 2);
    server.abort();
}

#[tokio::test]
async fn codex_anthropic_tool_use_commits_without_retry() {
    const TOOL_USE: &str = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_tool\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-sonnet-5\",\"content\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\n",
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"tool_1\",\"name\":\"exec\",\"input\":{}}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"code\\\":\\\"text('ok')\\\"}\"}}\n\n",
        "event: content_block_stop\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n\n"
    );
    let (base_url, hits, _bodies, server) = spawn_scripted_streaming_upstream(vec![(
        StatusCode::OK,
        ScriptedStreamingBody::Sse(TOOL_USE),
    )])
    .await;
    let mut provider = codex_provider("p1", &base_url, false);
    provider.meta = Some(ProviderMeta {
        api_format: Some("anthropic".to_string()),
        ..Default::default()
    });
    let (db, router) = test_router().await;
    let forwarder = RequestForwarder::new(router).expect("create forwarder");
    db.save_provider("codex", &provider)
        .expect("save provider for health tracking");

    let result = forwarder
        .forward_response(
            &AppType::Codex,
            "/v1/responses",
            json!({"model": "gpt-5.6-terra", "input": "run once", "stream": true}),
            &HeaderMap::new(),
            vec![provider],
            ForwardOptions {
                max_retries: 2,
                request_timeout: Some(Duration::from_secs(5)),
                bypass_circuit_breaker: true,
            },
            RectifierConfig::default(),
        )
        .await
        .expect("tool use should commit the first response without retrying");

    let StreamingResponse::Live(response) = result.response else {
        panic!("tool use should remain streaming");
    };
    let body = response
        .bytes_stream()
        .map(|chunk| chunk.expect("read tool-use response chunk"))
        .collect::<Vec<_>>()
        .await
        .concat();
    let body = String::from_utf8(body).expect("tool-use stream is utf-8");
    assert!(body.contains("content_block_start"));
    assert!(body.contains("input_json_delta"));
    assert_eq!(hits.count.load(Ordering::SeqCst), 1);
    server.abort();
}

#[tokio::test]
async fn codex_anthropic_empty_success_retry_exhaustion_returns_503() {
    const EMPTY: &str = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_empty\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-sonnet-5\",\"content\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\n",
        "event: message_delta\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":0}}\n\n",
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n\n"
    );
    let (base_url, hits, _bodies, server) = spawn_scripted_streaming_upstream(vec![
        (StatusCode::OK, ScriptedStreamingBody::Sse(EMPTY)),
        (StatusCode::OK, ScriptedStreamingBody::Sse(EMPTY)),
    ])
    .await;
    let mut provider = codex_provider("p1", &base_url, false);
    provider.meta = Some(ProviderMeta {
        api_format: Some("anthropic".to_string()),
        ..Default::default()
    });
    let (db, router) = test_router().await;
    let forwarder = RequestForwarder::new(router).expect("create forwarder");
    db.save_provider("codex", &provider)
        .expect("save provider for health tracking");

    let error = forwarder
        .forward_response(
            &AppType::Codex,
            "/v1/responses",
            json!({"model": "gpt-5.6-terra", "input": "continue", "stream": true}),
            &HeaderMap::new(),
            vec![provider],
            ForwardOptions {
                max_retries: 1,
                request_timeout: Some(Duration::from_secs(5)),
                bypass_circuit_breaker: true,
            },
            RectifierConfig::default(),
        )
        .await
        .expect_err("persistent empty Anthropic completions should fail explicitly");

    match error {
        ProxyError::UpstreamError { status, body } => {
            assert_eq!(status, 503);
            let body: Value = serde_json::from_str(body.as_deref().expect("error body"))
                .expect("parse Anthropic error envelope");
            assert_eq!(body["error"]["type"], "service_unavailable_error");
            assert!(body["error"]["message"]
                .as_str()
                .is_some_and(|message| message.contains("without text")));
        }
        other => panic!("expected upstream 503, got {other:?}"),
    }
    assert_eq!(hits.count.load(Ordering::SeqCst), 2);
    server.abort();
}

#[tokio::test]
async fn codex_anthropic_empty_and_invalid_json_retry_before_client_commit() {
    let success = json!({
        "id": "msg_ok",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5",
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1}
    });
    let (base_url, hits, _bodies, server) = spawn_scripted_streaming_upstream(vec![
        (StatusCode::OK, ScriptedStreamingBody::RawJson(Bytes::new())),
        (
            StatusCode::OK,
            ScriptedStreamingBody::RawJson(Bytes::from_static(b"{\"type\":")),
        ),
        (StatusCode::OK, ScriptedStreamingBody::Json(success)),
    ])
    .await;
    let mut provider = codex_provider("p1", &base_url, false);
    provider.meta = Some(ProviderMeta {
        api_format: Some("anthropic".to_string()),
        ..Default::default()
    });
    let (db, router) = test_router().await;
    let forwarder = RequestForwarder::new(router).expect("create forwarder");
    db.save_provider("codex", &provider)
        .expect("save provider for health tracking");

    let result = forwarder
        .forward_response(
            &AppType::Codex,
            "/v1/responses",
            json!({"model": "gpt-5.6-terra", "input": "continue", "stream": true}),
            &HeaderMap::new(),
            vec![provider],
            ForwardOptions {
                max_retries: 2,
                request_timeout: Some(Duration::from_secs(10)),
                bypass_circuit_breaker: true,
            },
            RectifierConfig::default(),
        )
        .await
        .expect("empty and truncated JSON should retry before client commit");

    assert_eq!(result.response.status(), StatusCode::OK);
    assert_eq!(hits.count.load(Ordering::SeqCst), 3);
    server.abort();
}

#[tokio::test]
async fn codex_anthropic_redacted_thinking_is_substantive_output() {
    const REDACTED: &str = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_reasoning\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-sonnet-5\",\"content\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n",
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"redacted_thinking\",\"data\":\"encrypted-reasoning\"}}\n\n",
        "event: content_block_stop\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n\n"
    );
    let (base_url, hits, _bodies, server) = spawn_scripted_streaming_upstream(vec![(
        StatusCode::OK,
        ScriptedStreamingBody::Sse(REDACTED),
    )])
    .await;
    let mut provider = codex_provider("p1", &base_url, false);
    provider.meta = Some(ProviderMeta {
        api_format: Some("anthropic".to_string()),
        ..Default::default()
    });
    let (db, router) = test_router().await;
    let forwarder = RequestForwarder::new(router).expect("create forwarder");
    db.save_provider("codex", &provider)
        .expect("save provider for health tracking");

    let result = forwarder
        .forward_response(
            &AppType::Codex,
            "/v1/responses",
            json!({"model": "gpt-5.6-terra", "input": "reason", "stream": true}),
            &HeaderMap::new(),
            vec![provider],
            ForwardOptions {
                max_retries: 2,
                request_timeout: Some(Duration::from_secs(5)),
                bypass_circuit_breaker: true,
            },
            RectifierConfig::default(),
        )
        .await
        .expect("redacted thinking should be committed without retry");

    let StreamingResponse::Live(response) = result.response else {
        panic!("redacted thinking should remain streaming");
    };
    let body = response
        .bytes_stream()
        .map(|chunk| chunk.expect("read redacted-thinking response chunk"))
        .collect::<Vec<_>>()
        .await
        .concat();
    assert!(String::from_utf8(body)
        .expect("redacted-thinking stream is utf-8")
        .contains("encrypted-reasoning"));
    assert_eq!(hits.count.load(Ordering::SeqCst), 1);
    server.abort();
}

#[test]
fn codex_anthropic_json_validation_handles_redacted_thinking_and_non_utf8() {
    let redacted = serde_json::to_vec(&json!({
        "id": "msg_reasoning",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "redacted_thinking", "data": "encrypted-reasoning"}]
    }))
    .expect("serialize redacted thinking response");
    super::super::validate_codex_anthropic_success_body(&redacted)
        .expect("redacted thinking is substantive output");

    let error = super::super::validate_codex_anthropic_success_body(&[0xff])
        .expect_err("non-UTF-8 success bodies must be rejected");
    assert!(matches!(
        error,
        ProxyError::UpstreamError { status: 503, .. }
    ));
}

#[tokio::test]
async fn codex_anthropic_final_sse_block_without_separator_is_inspected() {
    const TEXT_WITHOUT_SEPARATOR: &str = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_text\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-sonnet-5\",\"content\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n",
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"ok\"}}"
    );
    let (base_url, hits, _bodies, server) = spawn_scripted_streaming_upstream(vec![(
        StatusCode::OK,
        ScriptedStreamingBody::Sse(TEXT_WITHOUT_SEPARATOR),
    )])
    .await;
    let mut provider = codex_provider("p1", &base_url, false);
    provider.meta = Some(ProviderMeta {
        api_format: Some("anthropic".to_string()),
        ..Default::default()
    });
    let (db, router) = test_router().await;
    let forwarder = RequestForwarder::new(router).expect("create forwarder");
    db.save_provider("codex", &provider)
        .expect("save provider for health tracking");

    let result = forwarder
        .forward_response(
            &AppType::Codex,
            "/v1/responses",
            json!({"model": "gpt-5.6-terra", "input": "reply", "stream": true}),
            &HeaderMap::new(),
            vec![provider],
            ForwardOptions {
                max_retries: 0,
                request_timeout: Some(Duration::from_secs(5)),
                bypass_circuit_breaker: true,
            },
            RectifierConfig::default(),
        )
        .await
        .expect("a final unterminated SSE block with output should remain valid");

    assert_eq!(result.response.status(), StatusCode::OK);
    assert_eq!(hits.count.load(Ordering::SeqCst), 1);
    server.abort();
}

#[tokio::test]
async fn codex_anthropic_oversized_non_output_prelude_is_rejected() {
    let oversized_comment = format!(
        ":{}\n\nevent: content_block_start\ndata: {{\"type\":\"content_block_start\",\"index\":0,\"content_block\":{{\"type\":\"text\",\"text\":\"too late\"}}}}\n\n",
        "x".repeat(256 * 1024)
    );
    let (base_url, hits, _bodies, server) = spawn_scripted_streaming_upstream(vec![(
        StatusCode::OK,
        ScriptedStreamingBody::OwnedSse(oversized_comment),
    )])
    .await;
    let mut provider = codex_provider("p1", &base_url, false);
    provider.meta = Some(ProviderMeta {
        api_format: Some("anthropic".to_string()),
        ..Default::default()
    });
    let (db, router) = test_router().await;
    let forwarder = RequestForwarder::new(router).expect("create forwarder");
    db.save_provider("codex", &provider)
        .expect("save provider for health tracking");

    let error = forwarder
        .forward_response(
            &AppType::Codex,
            "/v1/responses",
            json!({"model": "gpt-5.6-terra", "input": "continue", "stream": true}),
            &HeaderMap::new(),
            vec![provider],
            ForwardOptions {
                max_retries: 0,
                request_timeout: Some(Duration::from_secs(5)),
                bypass_circuit_breaker: true,
            },
            RectifierConfig::default(),
        )
        .await
        .expect_err("an oversized prelude without output must not be committed");

    assert!(matches!(
        error,
        ProxyError::UpstreamError { status: 503, .. }
    ));
    assert_eq!(hits.count.load(Ordering::SeqCst), 1);
    server.abort();
}

#[tokio::test]
async fn codex_anthropic_malformed_tool_use_is_rejected() {
    const MALFORMED_TOOL: &str = concat!(
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"\",\"name\":\"\",\"input\":null}}\n\n",
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n\n"
    );
    let (base_url, hits, _bodies, server) = spawn_scripted_streaming_upstream(vec![(
        StatusCode::OK,
        ScriptedStreamingBody::Sse(MALFORMED_TOOL),
    )])
    .await;
    let mut provider = codex_provider("p1", &base_url, false);
    provider.meta = Some(ProviderMeta {
        api_format: Some("anthropic".to_string()),
        ..Default::default()
    });
    let (db, router) = test_router().await;
    let forwarder = RequestForwarder::new(router).expect("create forwarder");
    db.save_provider("codex", &provider)
        .expect("save provider for health tracking");

    let error = forwarder
        .forward_response(
            &AppType::Codex,
            "/v1/responses",
            json!({"model": "gpt-5.6-terra", "input": "run", "stream": true}),
            &HeaderMap::new(),
            vec![provider],
            ForwardOptions {
                max_retries: 0,
                request_timeout: Some(Duration::from_secs(5)),
                bypass_circuit_breaker: true,
            },
            RectifierConfig::default(),
        )
        .await
        .expect_err("a malformed tool use must not be committed");

    assert!(matches!(
        error,
        ProxyError::UpstreamError { status: 503, .. }
    ));
    assert_eq!(hits.count.load(Ordering::SeqCst), 1);
    server.abort();
}

#[test]
fn codex_anthropic_malformed_json_tool_use_is_rejected() {
    let malformed = serde_json::to_vec(&json!({
        "type": "message",
        "content": [{"type": "tool_use", "id": "tool_1", "name": "exec"}]
    }))
    .expect("serialize malformed tool response");
    let error = super::super::validate_codex_anthropic_success_body(&malformed)
        .expect_err("a JSON tool use without an input object must be rejected");
    assert!(matches!(
        error,
        ProxyError::UpstreamError { status: 503, .. }
    ));
}

#[test]
fn codex_anthropic_sse_validation_enforces_block_lifecycle_and_tool_json() {
    let mut validator = super::super::AnthropicStreamStartValidator::default();
    let orphan_delta = concat!(
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,",
        "\"delta\":{\"type\":\"text_delta\",\"text\":\"ignored\"}}"
    );
    assert!(validator.inspect_sse_block(orphan_delta).is_err());

    let mut validator = super::super::AnthropicStreamStartValidator::default();
    let tool_start = concat!(
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,",
        "\"content_block\":{\"type\":\"tool_use\",\"id\":\"tool_1\",",
        "\"name\":\"exec\",\"input\":{}}}"
    );
    let bad_delta = concat!(
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,",
        "\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{bad\"}}"
    );
    let tool_stop =
        "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}";
    assert!(!validator
        .inspect_sse_block(tool_start)
        .expect("valid tool start"));
    assert!(!validator
        .inspect_sse_block(bad_delta)
        .expect("tool JSON is incomplete until block stop"));
    assert!(validator.inspect_sse_block(tool_stop).is_err());

    let mut validator = super::super::AnthropicStreamStartValidator::default();
    assert!(validator
        .inspect_sse_block("event: error\ndata: {not-json}")
        .is_err());
}

#[test]
fn codex_anthropic_sse_validation_rejects_invalid_utf8() {
    let mut buffer = String::new();
    let mut remainder = Vec::new();
    let error = super::super::append_utf8_strict(
        &mut buffer,
        &mut remainder,
        b"data: {\"text\":\"\xff\"}\n\n",
    )
    .expect_err("invalid UTF-8 must not be replaced and accepted");
    assert!(matches!(
        error,
        ProxyError::UpstreamError { status: 503, .. }
    ));
}

#[tokio::test]
async fn codex_anthropic_crlf_and_split_utf8_remain_streamable() {
    let payload = concat!(
        "event: content_block_start\r\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\r\n\r\n",
        "event: content_block_delta\r\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"你好\"}}\r\n\r\n"
    );
    let split = payload
        .as_bytes()
        .windows("你".len())
        .position(|window| window == "你".as_bytes())
        .expect("find multibyte text")
        + 1;
    let chunks = vec![
        Bytes::copy_from_slice(&payload.as_bytes()[..split]),
        Bytes::copy_from_slice(&payload.as_bytes()[split..]),
    ];
    let (base_url, hits, _bodies, server) = spawn_scripted_streaming_upstream(vec![(
        StatusCode::OK,
        ScriptedStreamingBody::Chunks(chunks),
    )])
    .await;
    let mut provider = codex_provider("p1", &base_url, false);
    provider.meta = Some(ProviderMeta {
        api_format: Some("anthropic".to_string()),
        ..Default::default()
    });
    let (db, router) = test_router().await;
    let forwarder = RequestForwarder::new(router).expect("create forwarder");
    db.save_provider("codex", &provider)
        .expect("save provider for health tracking");

    let result = forwarder
        .forward_response(
            &AppType::Codex,
            "/v1/responses",
            json!({"model": "gpt-5.6-terra", "input": "reply", "stream": true}),
            &HeaderMap::new(),
            vec![provider],
            ForwardOptions {
                max_retries: 0,
                request_timeout: Some(Duration::from_secs(5)),
                bypass_circuit_breaker: true,
            },
            RectifierConfig::default(),
        )
        .await
        .expect("CRLF and split UTF-8 should remain valid");

    assert_eq!(result.response.status(), StatusCode::OK);
    assert_eq!(hits.count.load(Ordering::SeqCst), 1);
    server.abort();
}

#[tokio::test]
async fn codex_anthropic_oversized_json_is_rejected_before_commit() {
    let oversized = Bytes::from(vec![b' '; 16 * 1024 * 1024 + 1]);
    let (base_url, hits, _bodies, server) = spawn_scripted_streaming_upstream(vec![(
        StatusCode::OK,
        ScriptedStreamingBody::RawJson(oversized),
    )])
    .await;
    let mut provider = codex_provider("p1", &base_url, false);
    provider.meta = Some(ProviderMeta {
        api_format: Some("anthropic".to_string()),
        ..Default::default()
    });
    let (db, router) = test_router().await;
    let forwarder = RequestForwarder::new(router).expect("create forwarder");
    db.save_provider("codex", &provider)
        .expect("save provider for health tracking");

    let error = forwarder
        .forward_response(
            &AppType::Codex,
            "/v1/responses",
            json!({"model": "gpt-5.6-terra", "input": "continue", "stream": true}),
            &HeaderMap::new(),
            vec![provider],
            ForwardOptions {
                max_retries: 0,
                request_timeout: Some(Duration::from_secs(5)),
                bypass_circuit_breaker: true,
            },
            RectifierConfig::default(),
        )
        .await
        .expect_err("an oversized JSON body must not be buffered without a limit");

    assert!(matches!(
        error,
        ProxyError::UpstreamError { status: 503, .. }
    ));
    assert_eq!(hits.count.load(Ordering::SeqCst), 1);
    server.abort();
}

#[tokio::test]
async fn codex_anthropic_encoded_json_is_rejected_before_decompression() {
    let (base_url, hits, _bodies, server) = spawn_scripted_streaming_upstream(vec![(
        StatusCode::OK,
        ScriptedStreamingBody::EncodedJson(Bytes::from_static(b"not-really-gzip"), "gzip"),
    )])
    .await;
    let mut provider = codex_provider("p1", &base_url, false);
    provider.meta = Some(ProviderMeta {
        api_format: Some("anthropic".to_string()),
        ..Default::default()
    });
    let (db, router) = test_router().await;
    let forwarder = RequestForwarder::new(router).expect("create forwarder");
    db.save_provider("codex", &provider)
        .expect("save provider for health tracking");

    let error = forwarder
        .forward_response(
            &AppType::Codex,
            "/v1/responses",
            json!({"model": "gpt-5.6-terra", "input": "continue", "stream": true}),
            &HeaderMap::new(),
            vec![provider],
            ForwardOptions {
                max_retries: 0,
                request_timeout: Some(Duration::from_secs(5)),
                bypass_circuit_breaker: true,
            },
            RectifierConfig::default(),
        )
        .await
        .expect_err("encoded JSON must not bypass the decoded body limit");

    assert!(matches!(
        error,
        ProxyError::UpstreamError { status: 503, .. }
    ));
    assert_eq!(hits.count.load(Ordering::SeqCst), 1);
    server.abort();
}

#[tokio::test]
async fn codex_anthropic_json_overload_retries_with_503_semantics() {
    let overload = json!({
        "type": "error",
        "error": {"type": "overloaded_error", "message": "busy"}
    });
    let success = json!({
        "id": "msg_ok",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "ok"}]
    });
    let (base_url, hits, _bodies, server) = spawn_scripted_streaming_upstream(vec![
        (StatusCode::OK, ScriptedStreamingBody::Json(overload)),
        (StatusCode::OK, ScriptedStreamingBody::Json(success)),
    ])
    .await;
    let mut provider = codex_provider("p1", &base_url, false);
    provider.meta = Some(ProviderMeta {
        api_format: Some("anthropic".to_string()),
        ..Default::default()
    });
    let (db, router) = test_router().await;
    let forwarder = RequestForwarder::new(router).expect("create forwarder");
    db.save_provider("codex", &provider)
        .expect("save provider for health tracking");

    let result = forwarder
        .forward_response(
            &AppType::Codex,
            "/v1/responses",
            json!({"model": "gpt-5.6-terra", "input": "continue", "stream": true}),
            &HeaderMap::new(),
            vec![provider],
            ForwardOptions {
                max_retries: 1,
                request_timeout: Some(Duration::from_secs(5)),
                bypass_circuit_breaker: true,
            },
            RectifierConfig::default(),
        )
        .await
        .expect("a JSON overload envelope should retry and recover");

    assert_eq!(result.response.status(), StatusCode::OK);
    assert_eq!(hits.count.load(Ordering::SeqCst), 2);
    server.abort();
}

#[tokio::test]
async fn codex_anthropic_semantic_retry_shares_total_timeout() {
    const EMPTY: &str = concat!(
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n\n"
    );
    const SUCCESS: &str = concat!(
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"late\"}}\n\n"
    );
    let (base_url, hits, _bodies, server) = spawn_delayed_scripted_streaming_upstream(vec![
        (
            Duration::ZERO,
            StatusCode::OK,
            ScriptedStreamingBody::Sse(EMPTY),
        ),
        (
            Duration::from_millis(800),
            StatusCode::OK,
            ScriptedStreamingBody::Sse(SUCCESS),
        ),
    ])
    .await;
    let mut provider = codex_provider("p1", &base_url, false);
    provider.meta = Some(ProviderMeta {
        api_format: Some("anthropic".to_string()),
        ..Default::default()
    });
    let (db, router) = test_router().await;
    let forwarder = RequestForwarder::new(router).expect("create forwarder");
    db.save_provider("codex", &provider)
        .expect("save provider for health tracking");

    let started = Instant::now();
    let error = forwarder
        .forward_response(
            &AppType::Codex,
            "/v1/responses",
            json!({"model": "gpt-5.6-terra", "input": "continue", "stream": true}),
            &HeaderMap::new(),
            vec![provider],
            ForwardOptions {
                max_retries: 1,
                request_timeout: Some(Duration::from_millis(1500)),
                bypass_circuit_breaker: true,
            },
            RectifierConfig::default(),
        )
        .await
        .expect_err("semantic retries must share the configured total timeout");

    assert!(matches!(error, ProxyError::Timeout(_)));
    assert!(started.elapsed() < Duration::from_millis(1900));
    assert_eq!(hits.count.load(Ordering::SeqCst), 2);
    server.abort();
}

#[tokio::test]
async fn codex_native_responses_does_not_inherit_anthropic_retry_state() {
    const NATIVE_FAILURE: &str = concat!(
        "event: response.failed\n",
        "data: {\"type\":\"response.failed\",\"response\":{\"status\":\"failed\",\"error\":{\"type\":\"service_unavailable_error\",\"message\":\"busy\"}}}\n\n"
    );
    const ANTHROPIC_SUCCESS: &str = concat!(
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"ok\"}}\n\n"
    );
    let (native_url, native_hits, _native_bodies, native_server) =
        spawn_scripted_streaming_upstream(vec![(
            StatusCode::OK,
            ScriptedStreamingBody::Sse(NATIVE_FAILURE),
        )])
        .await;
    let (anthropic_url, anthropic_hits, _anthropic_bodies, anthropic_server) =
        spawn_scripted_streaming_upstream(vec![(
            StatusCode::OK,
            ScriptedStreamingBody::Sse(ANTHROPIC_SUCCESS),
        )])
        .await;
    let native = codex_provider("native", &native_url, false);
    let mut anthropic = codex_provider("anthropic", &anthropic_url, false);
    anthropic.meta = Some(ProviderMeta {
        api_format: Some("anthropic".to_string()),
        ..Default::default()
    });
    let (db, router) = test_router().await;
    let forwarder = RequestForwarder::new(router).expect("create forwarder");
    for provider in [&native, &anthropic] {
        db.save_provider("codex", provider)
            .expect("save provider for health tracking");
    }

    let result = forwarder
        .forward_response(
            &AppType::Codex,
            "/v1/responses",
            json!({"model": "gpt-5.6-terra", "input": "continue", "stream": true}),
            &HeaderMap::new(),
            vec![native, anthropic],
            ForwardOptions {
                max_retries: 1,
                request_timeout: Some(Duration::from_secs(5)),
                bypass_circuit_breaker: true,
            },
            RectifierConfig::default(),
        )
        .await
        .expect("native failure should fail over without an internal semantic retry");

    assert_eq!(result.provider.id, "anthropic");
    assert_eq!(native_hits.count.load(Ordering::SeqCst), 1);
    assert_eq!(anthropic_hits.count.load(Ordering::SeqCst), 1);
    native_server.abort();
    anthropic_server.abort();
}

#[tokio::test]
async fn single_provider_buffered_claude_non_2xx_returns_upstream_error() {
    let (primary_url, primary_hits, primary_server) = spawn_mock_upstream(
        StatusCode::TOO_MANY_REQUESTS,
        json!({"error": {"message": "rate limited"}}),
    )
    .await;
    let provider = claude_provider("p1", &primary_url, None);
    let (db, router) = test_router().await;
    let forwarder = RequestForwarder::new(router.clone()).expect("create forwarder");

    db.save_provider("claude", &provider)
        .expect("save provider for health tracking");

    let error = forwarder
        .forward_buffered_response(
            &AppType::Claude,
            "/v1/messages",
            claude_request_body(),
            &HeaderMap::new(),
            vec![provider],
            ForwardOptions {
                max_retries: 0,
                request_timeout: Some(Duration::from_secs(2)),
                bypass_circuit_breaker: false,
            },
            RectifierConfig::default(),
        )
        .await
        .expect_err("single-provider Claude 429 should surface as UpstreamError");

    match error {
        ProxyError::UpstreamError { status, body } => {
            assert_eq!(status, 429);
            assert_eq!(
                body.as_deref(),
                Some(r#"{"error":{"message":"rate limited"}}"#)
            );
        }
        other => panic!("expected UpstreamError, got {other:?}"),
    }
    assert_eq!(primary_hits.count.load(Ordering::SeqCst), 1);

    primary_server.abort();
}

#[tokio::test]
async fn last_provider_429_returns_upstream_error() {
    let (primary_url, _primary_hits, primary_server) = spawn_mock_upstream(
        StatusCode::INTERNAL_SERVER_ERROR,
        json!({"error": {"message": "primary down"}}),
    )
    .await;
    let (secondary_url, _secondary_hits, secondary_server) = spawn_mock_upstream(
        StatusCode::TOO_MANY_REQUESTS,
        json!({"error": {"message": "rate limited"}}),
    )
    .await;
    let (db, router) = test_router().await;
    let forwarder = RequestForwarder::new(router).expect("create forwarder");
    let provider_one = claude_provider("p1", &primary_url, None);
    let provider_two = claude_provider("p2", &secondary_url, None);

    db.save_provider("claude", &provider_one)
        .expect("save primary provider for health tracking");
    db.save_provider("claude", &provider_two)
        .expect("save secondary provider for health tracking");

    let error = forwarder
        .forward_buffered_response(
            &AppType::Claude,
            "/v1/messages",
            claude_request_body(),
            &HeaderMap::new(),
            vec![provider_one, provider_two],
            ForwardOptions {
                max_retries: 1,
                request_timeout: Some(Duration::from_secs(2)),
                bypass_circuit_breaker: false,
            },
            RectifierConfig::default(),
        )
        .await
        .expect_err("last provider 429 should surface as UpstreamError");

    match error {
        ProxyError::UpstreamError { status, body } => {
            assert_eq!(status, 429);
            let parsed: Value =
                serde_json::from_str(body.as_deref().expect("preserve upstream body"))
                    .expect("parse body");
            assert_eq!(parsed, json!({"error": {"message": "rate limited"}}));
        }
        other => panic!("expected UpstreamError, got {other:?}"),
    }

    primary_server.abort();
    secondary_server.abort();
}

#[tokio::test]
async fn last_streaming_provider_429_returns_upstream_error() {
    let (primary_url, _primary_hits, _primary_bodies, primary_server) =
        spawn_scripted_streaming_upstream(vec![(
            StatusCode::INTERNAL_SERVER_ERROR,
            ScriptedStreamingBody::Json(json!({"error": {"message": "primary down"}})),
        )])
        .await;
    let (secondary_url, _secondary_hits, _secondary_bodies, secondary_server) =
        spawn_scripted_streaming_upstream(vec![(
            StatusCode::TOO_MANY_REQUESTS,
            ScriptedStreamingBody::Json(json!({"error": {"message": "rate limited"}})),
        )])
        .await;
    let (db, router) = test_router().await;
    let forwarder = RequestForwarder::new(router).expect("create forwarder");
    let provider_one = claude_provider("p1", &primary_url, None);
    let provider_two = claude_provider("p2", &secondary_url, None);

    db.save_provider("claude", &provider_one)
        .expect("save primary provider for health tracking");
    db.save_provider("claude", &provider_two)
        .expect("save secondary provider for health tracking");

    let error = forwarder
        .forward_response(
            &AppType::Claude,
            "/v1/messages",
            json!({
                "model": "claude-3-7-sonnet-20250219",
                "stream": true,
                "max_tokens": 32,
                "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
            }),
            &HeaderMap::new(),
            vec![provider_one, provider_two],
            ForwardOptions {
                max_retries: 1,
                request_timeout: Some(Duration::from_secs(2)),
                bypass_circuit_breaker: false,
            },
            RectifierConfig::default(),
        )
        .await
        .expect_err("last streaming provider 429 should surface as UpstreamError");

    match error {
        ProxyError::UpstreamError { status, body } => {
            assert_eq!(status, 429);
            let parsed: Value =
                serde_json::from_str(body.as_deref().expect("preserve upstream body"))
                    .expect("parse body");
            assert_eq!(parsed, json!({"error": {"message": "rate limited"}}));
        }
        other => panic!("expected UpstreamError, got {other:?}"),
    }

    primary_server.abort();
    secondary_server.abort();
}

#[tokio::test]
async fn buffered_timeout_includes_body_read_budget_after_headers() {
    let (base_url, hits, server) = spawn_delayed_body_upstream().await;
    let provider = claude_provider("p1", &base_url, None);
    let (db, router) = test_router().await;
    let forwarder = RequestForwarder::new(router).expect("create forwarder");

    db.save_provider("claude", &provider)
        .expect("save provider for health tracking");

    let error = forwarder
        .forward_buffered_response(
            &AppType::Claude,
            "/v1/messages",
            claude_request_body(),
            &HeaderMap::new(),
            vec![provider],
            ForwardOptions {
                max_retries: 0,
                request_timeout: Some(Duration::from_millis(50)),
                bypass_circuit_breaker: true,
            },
            RectifierConfig::default(),
        )
        .await
        .expect_err("buffered request should time out while waiting for body");

    assert!(matches!(error, ProxyError::Timeout(message) if message.contains("request timed out")));
    assert_eq!(hits.count.load(Ordering::SeqCst), 1);

    server.abort();
}

#[tokio::test]
async fn buffered_body_timeout_after_headers_fails_over() {
    let (slow_url, slow_hits, slow_server) = spawn_delayed_body_upstream().await;
    let (fallback_url, fallback_hits, fallback_server) =
        spawn_mock_upstream(StatusCode::OK, json!({"ok": true})).await;
    let slow_provider = claude_provider("p1", &slow_url, None);
    let fallback_provider = claude_provider("p2", &fallback_url, None);
    let (db, router) = test_router().await;
    let forwarder = RequestForwarder::new(router).expect("create forwarder");

    db.save_provider("claude", &slow_provider)
        .expect("save slow provider for health tracking");
    db.save_provider("claude", &fallback_provider)
        .expect("save fallback provider for health tracking");

    let result = forwarder
        .forward_buffered_response(
            &AppType::Claude,
            "/v1/messages",
            claude_request_body(),
            &HeaderMap::new(),
            vec![slow_provider, fallback_provider],
            ForwardOptions {
                max_retries: 1,
                request_timeout: Some(Duration::from_millis(50)),
                bypass_circuit_breaker: false,
            },
            RectifierConfig::default(),
        )
        .await
        .expect("body timeout before commit should fail over");

    assert_eq!(result.provider.id, "p2");
    assert_eq!(result.response.status, StatusCode::OK);
    assert_eq!(slow_hits.count.load(Ordering::SeqCst), 1);
    assert_eq!(fallback_hits.count.load(Ordering::SeqCst), 1);

    slow_server.abort();
    fallback_server.abort();
}

#[tokio::test]
async fn streaming_first_chunk_timeout_after_headers_fails_over_and_replays_fallback() {
    let (slow_url, slow_hits, slow_server) = spawn_delayed_body_upstream().await;
    let fallback_sse =
        "data: {\"id\":\"msg_fallback\",\"type\":\"message_start\"}\n\ndata: [DONE]\n\n";
    let (fallback_url, fallback_hits, _bodies, fallback_server) =
        spawn_scripted_streaming_upstream(vec![(
            StatusCode::OK,
            ScriptedStreamingBody::Sse(fallback_sse),
        )])
        .await;
    let slow_provider = claude_provider("p1", &slow_url, None);
    let fallback_provider = claude_provider("p2", &fallback_url, None);
    let (db, router) = test_router().await;
    let forwarder = RequestForwarder::new(router).expect("create forwarder");

    db.save_provider("claude", &slow_provider)
        .expect("save slow provider");
    db.save_provider("claude", &fallback_provider)
        .expect("save fallback provider");

    let result = forwarder
        .forward_response(
            &AppType::Claude,
            "/v1/messages",
            json!({
                "model": "claude-3-7-sonnet-20250219",
                "stream": true,
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "hello"}]
            }),
            &HeaderMap::new(),
            vec![slow_provider, fallback_provider],
            ForwardOptions {
                max_retries: 1,
                request_timeout: Some(Duration::from_millis(50)),
                bypass_circuit_breaker: false,
            },
            RectifierConfig::default(),
        )
        .await
        .expect("first chunk timeout should fail over");

    assert_eq!(result.provider.id, "p2");
    let StreamingResponse::Live(response) = result.response else {
        panic!("fallback response should remain streaming");
    };
    let chunks = response
        .bytes_stream()
        .map(|chunk| chunk.expect("read replayed fallback chunk"))
        .collect::<Vec<_>>()
        .await;
    let replayed = chunks.concat();
    assert_eq!(replayed.as_slice(), fallback_sse.as_bytes());
    assert_eq!(slow_hits.count.load(Ordering::SeqCst), 1);
    assert_eq!(fallback_hits.count.load(Ordering::SeqCst), 1);

    slow_server.abort();
    fallback_server.abort();
}

#[tokio::test]
async fn streaming_first_chunk_error_fails_over() {
    let (failing_url, failing_hits, failing_server) = spawn_failing_body_upstream().await;
    let (fallback_url, fallback_hits, _bodies, fallback_server) =
        spawn_scripted_streaming_upstream(vec![(
            StatusCode::OK,
            ScriptedStreamingBody::Sse(
                "data: {\"id\":\"msg_ok\",\"type\":\"message_start\"}\n\ndata: [DONE]\n\n",
            ),
        )])
        .await;
    let failing_provider = claude_provider("p1", &failing_url, None);
    let fallback_provider = claude_provider("p2", &fallback_url, None);
    let (db, router) = test_router().await;
    let forwarder = RequestForwarder::new(router).expect("create forwarder");

    db.save_provider("claude", &failing_provider)
        .expect("save failing provider");
    db.save_provider("claude", &fallback_provider)
        .expect("save fallback provider");

    let result = forwarder
        .forward_response(
            &AppType::Claude,
            "/v1/messages",
            json!({
                "model": "claude-3-7-sonnet-20250219",
                "stream": true,
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "hello"}]
            }),
            &HeaderMap::new(),
            vec![failing_provider, fallback_provider],
            ForwardOptions {
                max_retries: 1,
                request_timeout: Some(Duration::from_secs(2)),
                bypass_circuit_breaker: false,
            },
            RectifierConfig::default(),
        )
        .await
        .expect("first chunk error should fail over");

    assert_eq!(result.provider.id, "p2");
    assert_eq!(failing_hits.count.load(Ordering::SeqCst), 1);
    assert_eq!(fallback_hits.count.load(Ordering::SeqCst), 1);

    failing_server.abort();
    fallback_server.abort();
}

#[tokio::test]
async fn max_retries_does_not_retry_the_same_provider() {
    let (base_url, hits, _bodies, server) = spawn_delayed_scripted_upstream(vec![
        (
            Duration::from_millis(100),
            StatusCode::OK,
            json!({"id": "first-attempt"}),
        ),
        (
            Duration::from_millis(0),
            StatusCode::OK,
            json!({"id": "second-attempt"}),
        ),
    ])
    .await;
    let provider = claude_provider("p1", &base_url, None);
    let (db, router) = test_router().await;
    let forwarder = RequestForwarder::new(router).expect("create forwarder");

    db.save_provider("claude", &provider)
        .expect("save provider for health tracking");

    let error = forwarder
        .forward_buffered_response(
            &AppType::Claude,
            "/v1/messages",
            claude_request_body(),
            &HeaderMap::new(),
            vec![provider],
            ForwardOptions {
                max_retries: 1,
                request_timeout: Some(Duration::from_millis(50)),
                bypass_circuit_breaker: true,
            },
            RectifierConfig::default(),
        )
        .await
        .expect_err("max_retries should only permit another provider attempt");

    assert!(matches!(error, ProxyError::Timeout(message) if message.contains("request timed out")));
    assert_eq!(hits.count.load(Ordering::SeqCst), 1);

    server.abort();
}

#[tokio::test]
async fn buffered_connect_error_maps_to_forward_failed() {
    let provider = claude_provider("p1", &closed_base_url().await, None);
    let (db, router) = test_router().await;
    let forwarder = RequestForwarder::new(router).expect("create forwarder");

    db.save_provider("claude", &provider)
        .expect("save provider for health tracking");

    let error = forwarder
        .forward_buffered_response(
            &AppType::Claude,
            "/v1/messages",
            claude_request_body(),
            &HeaderMap::new(),
            vec![provider],
            ForwardOptions {
                max_retries: 2,
                request_timeout: Some(Duration::from_secs(1)),
                bypass_circuit_breaker: true,
            },
            RectifierConfig::default(),
        )
        .await
        .expect_err("connect failures should map to forward failed");

    assert!(matches!(error, ProxyError::ForwardFailed(_)));
}

#[tokio::test]
async fn buffered_rectifier_retry_shares_request_timeout_budget() {
    let (base_url, hits, bodies, server) = spawn_delayed_scripted_upstream(vec![
        (
            Duration::from_millis(20),
            StatusCode::BAD_REQUEST,
            json!({"error": {"message": "messages.1.content.0: Invalid `signature` in `thinking` block"}}),
        ),
        (
            Duration::from_millis(40),
            StatusCode::OK,
            json!({"id": "msg_123", "content": []}),
        ),
    ])
    .await;
    let provider = claude_provider("p1", &base_url, None);
    let (db, router) = test_router().await;
    let forwarder = RequestForwarder::new(router).expect("create forwarder");

    db.save_provider("claude", &provider)
        .expect("save provider for health tracking");

    let body = json!({
        "model": "claude-3-7-sonnet-20250219",
        "max_tokens": 32,
        "messages": [{
            "role": "assistant",
            "content": [
                { "type": "thinking", "thinking": "t", "signature": "sig" },
                { "type": "text", "text": "hello", "signature": "text-sig" }
            ]
        }]
    });

    let error = forwarder
        .forward_buffered_response(
            &AppType::Claude,
            "/v1/messages",
            body,
            &HeaderMap::new(),
            vec![provider],
            ForwardOptions {
                max_retries: 0,
                request_timeout: Some(Duration::from_millis(50)),
                bypass_circuit_breaker: true,
            },
            RectifierConfig::default(),
        )
        .await
        .expect_err("rectifier retry should share a single buffered request timeout budget");

    assert!(matches!(error, ProxyError::Timeout(message) if message.contains("request timed out")));
    assert_eq!(hits.count.load(Ordering::SeqCst), 2);
    assert_eq!(bodies.lock().await.len(), 2);

    server.abort();
}

#[tokio::test]
async fn streaming_transport_timeout_fails_over_without_same_provider_retry() {
    let (primary_url, primary_hits, primary_bodies, primary_server) =
        spawn_delayed_scripted_streaming_upstream(vec![
            (
                Duration::from_millis(100),
                StatusCode::OK,
                ScriptedStreamingBody::Sse(
                    "data: {\"id\":\"primary-retry\",\"type\":\"message_start\"}\n\ndata: [DONE]\n\n",
                ),
            ),
            (
                Duration::from_millis(0),
                StatusCode::OK,
                ScriptedStreamingBody::Sse(
                    "data: {\"id\":\"primary-second\",\"type\":\"message_start\"}\n\ndata: [DONE]\n\n",
                ),
            ),
        ])
        .await;
    let (secondary_url, secondary_hits, secondary_bodies, secondary_server) =
        spawn_delayed_scripted_streaming_upstream(vec![(
            Duration::from_millis(0),
            StatusCode::OK,
            ScriptedStreamingBody::Sse(
                "data: {\"id\":\"secondary\",\"type\":\"message_start\"}\n\ndata: [DONE]\n\n",
            ),
        )])
        .await;
    let provider_one = claude_provider("p1", &primary_url, None);
    let provider_two = claude_provider("p2", &secondary_url, None);
    let (db, router) = test_router().await;
    let forwarder = RequestForwarder::new(router).expect("create forwarder");

    db.save_provider("claude", &provider_one)
        .expect("save primary provider for health tracking");
    db.save_provider("claude", &provider_two)
        .expect("save secondary provider for health tracking");

    let result = forwarder
        .forward_response(
            &AppType::Claude,
            "/v1/messages",
            json!({
                "model": "claude-3-7-sonnet-20250219",
                "stream": true,
                "max_tokens": 32,
                "messages": [{
                    "role": "user",
                    "content": [{ "type": "text", "text": "hello" }]
                }]
            }),
            &HeaderMap::new(),
            vec![provider_one, provider_two],
            ForwardOptions {
                max_retries: 1,
                request_timeout: Some(Duration::from_millis(50)),
                bypass_circuit_breaker: true,
            },
            RectifierConfig::default(),
        )
        .await
        .expect("transport timeout should fail over to next provider");

    assert_eq!(result.provider.id, "p2");
    assert_eq!(result.response.status(), StatusCode::OK);
    assert_eq!(primary_hits.count.load(Ordering::SeqCst), 1);
    assert_eq!(secondary_hits.count.load(Ordering::SeqCst), 1);
    assert_eq!(primary_bodies.lock().await.len(), 1);
    assert_eq!(secondary_bodies.lock().await.len(), 1);

    primary_server.abort();
    secondary_server.abort();
}

#[tokio::test]
async fn claude_streaming_success_path_does_not_trigger_rectifier_retry() {
    let (base_url, hits, bodies, server) = spawn_scripted_streaming_upstream(vec![(
        StatusCode::OK,
        ScriptedStreamingBody::Sse(
            "data: {\"id\":\"msg_123\",\"type\":\"message_start\"}\n\ndata: [DONE]\n\n",
        ),
    )])
    .await;
    let provider = claude_provider("p1", &base_url, None);
    let (db, router) = test_router().await;
    let forwarder = RequestForwarder::new(router).expect("create forwarder");

    db.save_provider("claude", &provider)
        .expect("save provider for health tracking");

    let body = json!({
        "model": "claude-3-7-sonnet-20250219",
        "stream": true,
        "max_tokens": 32,
        "messages": [{
            "role": "assistant",
            "content": [
                { "type": "thinking", "thinking": "t", "signature": "sig" },
                { "type": "text", "text": "hello", "signature": "text-sig" }
            ]
        }]
    });

    let result = forwarder
        .forward_response(
            &AppType::Claude,
            "/v1/messages",
            body,
            &HeaderMap::new(),
            vec![provider],
            ForwardOptions {
                max_retries: 0,
                request_timeout: Some(Duration::from_secs(2)),
                bypass_circuit_breaker: true,
            },
            RectifierConfig::default(),
        )
        .await
        .expect("streaming success path should not use rectifier retry");

    assert_eq!(result.response.status(), StatusCode::OK);
    assert!(matches!(
        &result.response,
        StreamingResponse::Live(response) if is_sse_response(response)
    ));
    assert_eq!(hits.count.load(Ordering::SeqCst), 1);

    let sent_bodies = bodies.lock().await;
    assert_eq!(sent_bodies.len(), 1);
    assert_eq!(
        sent_bodies[0]["messages"][0]["content"]
            .as_array()
            .unwrap()
            .len(),
        2
    );

    server.abort();
}

#[tokio::test]
async fn responses_overload_before_output_retries_same_provider_then_succeeds() {
    const OVERLOADED: &str = concat!(
        "event: response.failed\n",
        "data: {\"type\":\"response.failed\",\"response\":{\"status\":\"failed\",",
        "\"error\":{\"type\":\"service_unavailable_error\",",
        "\"message\":\"Our servers are currently overloaded. Please try again later.\"}}}\n\n"
    );
    const SUCCESS: &str = concat!(
        "event: response.output_text.delta\n",
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\n\n",
        "event: response.completed\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\n"
    );
    let (base_url, hits, _bodies, server) = spawn_scripted_streaming_upstream(vec![
        (StatusCode::OK, ScriptedStreamingBody::Sse(OVERLOADED)),
        (StatusCode::OK, ScriptedStreamingBody::Sse(OVERLOADED)),
        (StatusCode::OK, ScriptedStreamingBody::Sse(SUCCESS)),
    ])
    .await;
    let provider = claude_provider("p1", &base_url, Some("openai_responses"));
    let (db, router) = test_router().await;
    let forwarder = RequestForwarder::new(router).expect("create forwarder");
    db.save_provider("claude", &provider)
        .expect("save provider for health tracking");

    let result = forwarder
        .forward_response(
            &AppType::Claude,
            "/v1/messages",
            json!({
                "model": "claude-3-7-sonnet-20250219",
                "stream": true,
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "hello"}]
            }),
            &HeaderMap::new(),
            vec![provider],
            ForwardOptions {
                max_retries: 3,
                request_timeout: Some(Duration::from_secs(10)),
                bypass_circuit_breaker: true,
            },
            RectifierConfig::default(),
        )
        .await
        .expect("temporary overload should recover on the same provider");

    let StreamingResponse::Live(response) = result.response else {
        panic!("successful Responses retry should remain streaming");
    };
    let body = response
        .bytes_stream()
        .map(|chunk| chunk.expect("read retried response chunk"))
        .collect::<Vec<_>>()
        .await
        .concat();
    let body = String::from_utf8(body).expect("response stream is utf-8");
    assert!(body.contains("response.output_text.delta"));
    assert!(!body.contains("service_unavailable_error"));
    assert_eq!(hits.count.load(Ordering::SeqCst), 3);

    server.abort();
}

#[tokio::test]
async fn responses_overload_retry_budget_exhaustion_returns_503() {
    const OVERLOADED: &str = concat!(
        "event: response.failed\n",
        "data: {\"type\":\"response.failed\",\"response\":{\"status\":\"failed\",",
        "\"error\":{\"type\":\"service_unavailable_error\",\"message\":\"overloaded\"}}}\n\n"
    );
    let (base_url, hits, _bodies, server) = spawn_scripted_streaming_upstream(vec![
        (StatusCode::OK, ScriptedStreamingBody::Sse(OVERLOADED)),
        (StatusCode::OK, ScriptedStreamingBody::Sse(OVERLOADED)),
    ])
    .await;
    let provider = claude_provider("p1", &base_url, Some("openai_responses"));
    let (db, router) = test_router().await;
    let forwarder = RequestForwarder::new(router).expect("create forwarder");
    db.save_provider("claude", &provider)
        .expect("save provider for health tracking");

    let error = forwarder
        .forward_response(
            &AppType::Claude,
            "/v1/messages",
            json!({
                "model": "claude-3-7-sonnet-20250219",
                "stream": true,
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "hello"}]
            }),
            &HeaderMap::new(),
            vec![provider],
            ForwardOptions {
                max_retries: 1,
                request_timeout: Some(Duration::from_secs(5)),
                bypass_circuit_breaker: true,
            },
            RectifierConfig::default(),
        )
        .await
        .expect_err("persistent overload should stop after the configured retry budget");

    match error {
        ProxyError::UpstreamError { status, body } => {
            assert_eq!(status, 503);
            let body: Value = serde_json::from_str(body.as_deref().expect("error body"))
                .expect("parse Anthropic error envelope");
            assert_eq!(body["type"], "error");
            assert_eq!(body["error"]["type"], "service_unavailable_error");
            assert_eq!(body["error"]["message"], "overloaded");
        }
        other => panic!("expected upstream 503, got {other:?}"),
    }
    assert_eq!(hits.count.load(Ordering::SeqCst), 2);

    server.abort();
}

#[tokio::test]
async fn responses_invalid_request_before_output_is_not_retried() {
    const INVALID_REQUEST: &str = concat!(
        "event: response.failed\n",
        "data: {\"type\":\"response.failed\",\"response\":{\"status\":\"failed\",",
        "\"error\":{\"type\":\"invalid_request_error\",\"message\":\"bad input\"}}}\n\n"
    );
    let (base_url, hits, _bodies, server) = spawn_scripted_streaming_upstream(vec![(
        StatusCode::OK,
        ScriptedStreamingBody::Sse(INVALID_REQUEST),
    )])
    .await;
    let provider = claude_provider("p1", &base_url, Some("openai_responses"));
    let (db, router) = test_router().await;
    let forwarder = RequestForwarder::new(router).expect("create forwarder");
    db.save_provider("claude", &provider)
        .expect("save provider for health tracking");

    let error = forwarder
        .forward_response(
            &AppType::Claude,
            "/v1/messages",
            json!({
                "model": "claude-3-7-sonnet-20250219",
                "stream": true,
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "hello"}]
            }),
            &HeaderMap::new(),
            vec![provider],
            ForwardOptions {
                max_retries: 3,
                request_timeout: Some(Duration::from_secs(5)),
                bypass_circuit_breaker: true,
            },
            RectifierConfig::default(),
        )
        .await
        .expect_err("invalid requests must not be replayed");

    assert!(matches!(
        error,
        ProxyError::UpstreamError { status: 400, .. }
    ));
    assert_eq!(hits.count.load(Ordering::SeqCst), 1);

    server.abort();
}

#[tokio::test]
async fn responses_overload_retry_does_not_exceed_shared_timeout() {
    const OVERLOADED: &str = concat!(
        "event: response.failed\n",
        "data: {\"type\":\"response.failed\",\"response\":{\"status\":\"failed\",",
        "\"error\":{\"type\":\"overloaded_error\",\"message\":\"busy\"}}}\n\n"
    );
    let (base_url, hits, _bodies, server) = spawn_scripted_streaming_upstream(vec![(
        StatusCode::OK,
        ScriptedStreamingBody::Sse(OVERLOADED),
    )])
    .await;
    let provider = claude_provider("p1", &base_url, Some("openai_responses"));
    let (db, router) = test_router().await;
    let forwarder = RequestForwarder::new(router).expect("create forwarder");
    db.save_provider("claude", &provider)
        .expect("save provider for health tracking");

    let started_at = std::time::Instant::now();
    let error = forwarder
        .forward_response(
            &AppType::Claude,
            "/v1/messages",
            json!({
                "model": "claude-3-7-sonnet-20250219",
                "stream": true,
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "hello"}]
            }),
            &HeaderMap::new(),
            vec![provider],
            ForwardOptions {
                max_retries: 3,
                request_timeout: Some(Duration::from_millis(200)),
                bypass_circuit_breaker: true,
            },
            RectifierConfig::default(),
        )
        .await
        .expect_err("retry backoff must respect the original timeout budget");

    assert!(matches!(error, ProxyError::Timeout(_)));
    assert!(started_at.elapsed() < Duration::from_secs(1));
    assert_eq!(hits.count.load(Ordering::SeqCst), 1);

    server.abort();
}

#[tokio::test]
async fn responses_failure_after_first_output_is_not_replayed() {
    const PARTIAL_THEN_FAILED: &str = concat!(
        "event: response.output_text.delta\n",
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"partial\"}\n\n",
        "event: response.failed\n",
        "data: {\"type\":\"response.failed\",\"response\":{\"status\":\"failed\",",
        "\"error\":{\"type\":\"service_unavailable_error\",\"message\":\"late failure\"}}}\n\n"
    );
    let (base_url, hits, _bodies, server) = spawn_scripted_streaming_upstream(vec![(
        StatusCode::OK,
        ScriptedStreamingBody::Sse(PARTIAL_THEN_FAILED),
    )])
    .await;
    let provider = claude_provider("p1", &base_url, Some("openai_responses"));
    let (db, router) = test_router().await;
    let forwarder = RequestForwarder::new(router).expect("create forwarder");
    db.save_provider("claude", &provider)
        .expect("save provider for health tracking");

    let result = forwarder
        .forward_response(
            &AppType::Claude,
            "/v1/messages",
            json!({
                "model": "claude-3-7-sonnet-20250219",
                "stream": true,
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "hello"}]
            }),
            &HeaderMap::new(),
            vec![provider],
            ForwardOptions {
                max_retries: 3,
                request_timeout: Some(Duration::from_secs(5)),
                bypass_circuit_breaker: true,
            },
            RectifierConfig::default(),
        )
        .await
        .expect("the forwarder must commit once output has started");

    let StreamingResponse::Live(response) = result.response else {
        panic!("partial Responses output should remain streaming");
    };
    let body = response
        .bytes_stream()
        .map(|chunk| chunk.expect("read partial response chunk"))
        .collect::<Vec<_>>()
        .await
        .concat();
    let body = String::from_utf8(body).expect("response stream is utf-8");
    assert!(body.contains("partial"));
    assert!(body.contains("late failure"));
    assert_eq!(hits.count.load(Ordering::SeqCst), 1);

    server.abort();
}

#[tokio::test]
async fn buffered_responses_overload_sse_retries_same_provider_then_succeeds() {
    const OVERLOADED: &str = concat!(
        "event: response.failed\n",
        "data: {\"type\":\"response.failed\",\"response\":{\"status\":\"failed\",",
        "\"error\":{\"type\":\"service_unavailable_error\",\"message\":\"busy\"}}}\n\n"
    );
    const SUCCESS: &str = concat!(
        "event: response.output_text.delta\n",
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\n\n",
        "event: response.completed\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\n"
    );
    let (base_url, hits, _bodies, server) = spawn_scripted_streaming_upstream(vec![
        (StatusCode::OK, ScriptedStreamingBody::Sse(OVERLOADED)),
        (StatusCode::OK, ScriptedStreamingBody::Sse(SUCCESS)),
    ])
    .await;
    let provider = claude_provider("p1", &base_url, Some("openai_responses"));
    let (db, router) = test_router().await;
    let forwarder = RequestForwarder::new(router).expect("create forwarder");
    db.save_provider("claude", &provider)
        .expect("save provider for health tracking");

    let result = forwarder
        .forward_buffered_response(
            &AppType::Claude,
            "/v1/messages",
            json!({
                "model": "claude-3-7-sonnet-20250219",
                "stream": false,
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "hello"}]
            }),
            &HeaderMap::new(),
            vec![provider],
            ForwardOptions {
                max_retries: 1,
                request_timeout: Some(Duration::from_secs(5)),
                bypass_circuit_breaker: true,
            },
            RectifierConfig::default(),
        )
        .await
        .expect("buffered Responses overload should recover");

    assert!(String::from_utf8_lossy(&result.response.body).contains("response.completed"));
    assert_eq!(hits.count.load(Ordering::SeqCst), 2);
    server.abort();
}

#[tokio::test]
async fn buffered_responses_json_overload_retries_but_invalid_request_does_not() {
    const SUCCESS: &str = concat!(
        "event: response.completed\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\n"
    );
    let (retry_url, retry_hits, _retry_bodies, retry_server) =
        spawn_scripted_streaming_upstream(vec![
            (
                StatusCode::OK,
                ScriptedStreamingBody::Json(json!({
                    "code": "server_error",
                    "message": "busy"
                })),
            ),
            (StatusCode::OK, ScriptedStreamingBody::Sse(SUCCESS)),
        ])
        .await;
    let (invalid_url, invalid_hits, _invalid_bodies, invalid_server) =
        spawn_scripted_streaming_upstream(vec![(
            StatusCode::OK,
            ScriptedStreamingBody::Json(json!({
                "status": "failed",
                "error": {"type": "invalid_request_error", "message": "bad input"}
            })),
        )])
        .await;
    let retry_provider = claude_provider("retry", &retry_url, Some("openai_responses"));
    let invalid_provider = claude_provider("invalid", &invalid_url, Some("openai_responses"));
    let (db, router) = test_router().await;
    let forwarder = RequestForwarder::new(router).expect("create forwarder");
    db.save_provider("claude", &retry_provider)
        .expect("save retry provider");
    db.save_provider("claude", &invalid_provider)
        .expect("save invalid provider");
    let request = json!({
        "model": "claude-3-7-sonnet-20250219",
        "stream": false,
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "hello"}]
    });
    let options = ForwardOptions {
        max_retries: 1,
        request_timeout: Some(Duration::from_secs(5)),
        bypass_circuit_breaker: true,
    };

    forwarder
        .forward_buffered_response(
            &AppType::Claude,
            "/v1/messages",
            request.clone(),
            &HeaderMap::new(),
            vec![retry_provider],
            options,
            RectifierConfig::default(),
        )
        .await
        .expect("JSON server_error should retry");
    let error = forwarder
        .forward_buffered_response(
            &AppType::Claude,
            "/v1/messages",
            request,
            &HeaderMap::new(),
            vec![invalid_provider],
            options,
            RectifierConfig::default(),
        )
        .await
        .expect_err("JSON invalid_request_error must not retry");

    assert!(matches!(
        error,
        ProxyError::UpstreamError { status: 400, .. }
    ));
    assert_eq!(retry_hits.count.load(Ordering::SeqCst), 2);
    assert_eq!(invalid_hits.count.load(Ordering::SeqCst), 1);
    retry_server.abort();
    invalid_server.abort();
}

#[tokio::test]
async fn http_503_responses_overload_retries_for_streaming_and_buffered_requests() {
    const STREAM_SUCCESS: &str = concat!(
        "event: response.output_text.delta\n",
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\n\n"
    );
    let overload = json!({
        "error": {"type": "overloaded_error", "message": "busy"}
    });
    let (stream_url, stream_hits, _stream_bodies, stream_server) =
        spawn_scripted_streaming_upstream(vec![
            (
                StatusCode::SERVICE_UNAVAILABLE,
                ScriptedStreamingBody::Json(overload.clone()),
            ),
            (StatusCode::OK, ScriptedStreamingBody::Sse(STREAM_SUCCESS)),
        ])
        .await;
    let (buffered_url, buffered_hits, _buffered_bodies, buffered_server) =
        spawn_scripted_streaming_upstream(vec![
            (
                StatusCode::SERVICE_UNAVAILABLE,
                ScriptedStreamingBody::Json(overload),
            ),
            (
                StatusCode::OK,
                ScriptedStreamingBody::Json(json!({
                    "id": "resp_ok",
                    "status": "completed",
                    "output": []
                })),
            ),
        ])
        .await;
    let stream_provider = claude_provider("stream", &stream_url, Some("openai_responses"));
    let buffered_provider = claude_provider("buffered", &buffered_url, Some("openai_responses"));
    let (db, router) = test_router().await;
    let forwarder = RequestForwarder::new(router).expect("create forwarder");
    db.save_provider("claude", &stream_provider)
        .expect("save streaming provider");
    db.save_provider("claude", &buffered_provider)
        .expect("save buffered provider");
    let options = ForwardOptions {
        max_retries: 1,
        request_timeout: Some(Duration::from_secs(5)),
        bypass_circuit_breaker: true,
    };

    forwarder
        .forward_response(
            &AppType::Claude,
            "/v1/messages",
            json!({
                "model": "claude-3-7-sonnet-20250219",
                "stream": true,
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "hello"}]
            }),
            &HeaderMap::new(),
            vec![stream_provider],
            options,
            RectifierConfig::default(),
        )
        .await
        .expect("streaming HTTP 503 overload should retry");
    forwarder
        .forward_buffered_response(
            &AppType::Claude,
            "/v1/messages",
            json!({
                "model": "claude-3-7-sonnet-20250219",
                "stream": false,
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "hello"}]
            }),
            &HeaderMap::new(),
            vec![buffered_provider],
            options,
            RectifierConfig::default(),
        )
        .await
        .expect("buffered HTTP 503 overload should retry");

    assert_eq!(stream_hits.count.load(Ordering::SeqCst), 2);
    assert_eq!(buffered_hits.count.load(Ordering::SeqCst), 2);
    stream_server.abort();
    buffered_server.abort();
}

#[tokio::test]
async fn top_level_responses_error_code_is_classified_and_retried() {
    const SERVER_ERROR: &str = concat!(
        "event: error\n",
        "data: {\"type\":\"error\",\"code\":\"server_error\",",
        "\"error\":{\"message\":\"busy\"}}\n\n"
    );
    const SUCCESS: &str = concat!(
        "event: response.output_text.delta\n",
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\n\n"
    );
    let (base_url, hits, _bodies, server) = spawn_scripted_streaming_upstream(vec![
        (StatusCode::OK, ScriptedStreamingBody::Sse(SERVER_ERROR)),
        (StatusCode::OK, ScriptedStreamingBody::Sse(SUCCESS)),
    ])
    .await;
    let provider = claude_provider("p1", &base_url, Some("openai_responses"));
    let (db, router) = test_router().await;
    let forwarder = RequestForwarder::new(router).expect("create forwarder");
    db.save_provider("claude", &provider)
        .expect("save provider for health tracking");

    forwarder
        .forward_response(
            &AppType::Claude,
            "/v1/messages",
            json!({
                "model": "claude-3-7-sonnet-20250219",
                "stream": true,
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "hello"}]
            }),
            &HeaderMap::new(),
            vec![provider],
            ForwardOptions {
                max_retries: 1,
                request_timeout: Some(Duration::from_secs(5)),
                bypass_circuit_breaker: true,
            },
            RectifierConfig::default(),
        )
        .await
        .expect("top-level server_error should be retryable");

    assert_eq!(hits.count.load(Ordering::SeqCst), 2);
    server.abort();
}

#[tokio::test]
async fn data_only_responses_error_code_is_classified_and_retried() {
    const SERVER_ERROR: &str = "data: {\"code\":\"server_error\",\"message\":\"busy\"}\n\n";
    const SUCCESS: &str = concat!(
        "event: response.output_text.delta\n",
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\n\n"
    );
    let (base_url, hits, _bodies, server) = spawn_scripted_streaming_upstream(vec![
        (StatusCode::OK, ScriptedStreamingBody::Sse(SERVER_ERROR)),
        (StatusCode::OK, ScriptedStreamingBody::Sse(SUCCESS)),
    ])
    .await;
    let provider = claude_provider("p1", &base_url, Some("openai_responses"));
    let (db, router) = test_router().await;
    let forwarder = RequestForwarder::new(router).expect("create forwarder");
    db.save_provider("claude", &provider)
        .expect("save provider for health tracking");

    forwarder
        .forward_response(
            &AppType::Claude,
            "/v1/messages",
            json!({
                "model": "claude-3-7-sonnet-20250219",
                "stream": true,
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "hello"}]
            }),
            &HeaderMap::new(),
            vec![provider],
            ForwardOptions {
                max_retries: 1,
                request_timeout: Some(Duration::from_secs(5)),
                bypass_circuit_breaker: true,
            },
            RectifierConfig::default(),
        )
        .await
        .expect("data-only server_error should be retryable");

    assert_eq!(hits.count.load(Ordering::SeqCst), 2);
    server.abort();
}

#[tokio::test]
async fn non_emitting_output_item_added_keeps_semantic_retry_available() {
    const MESSAGE_THEN_FAILED: &str = concat!(
        "event: response.output_item.added\n",
        "data: {\"type\":\"response.output_item.added\",",
        "\"item\":{\"type\":\"message\",\"id\":\"msg_1\"}}\n\n",
        "event: response.failed\n",
        "data: {\"type\":\"response.failed\",\"response\":{\"status\":\"failed\",",
        "\"error\":{\"type\":\"server_error\",\"message\":\"busy\"}}}\n\n"
    );
    const SUCCESS: &str = concat!(
        "event: response.output_text.delta\n",
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\n\n"
    );
    let (base_url, hits, _bodies, server) = spawn_scripted_streaming_upstream(vec![
        (
            StatusCode::OK,
            ScriptedStreamingBody::Sse(MESSAGE_THEN_FAILED),
        ),
        (StatusCode::OK, ScriptedStreamingBody::Sse(SUCCESS)),
    ])
    .await;
    let provider = claude_provider("p1", &base_url, Some("openai_responses"));
    let (db, router) = test_router().await;
    let forwarder = RequestForwarder::new(router).expect("create forwarder");
    db.save_provider("claude", &provider)
        .expect("save provider for health tracking");

    forwarder
        .forward_response(
            &AppType::Claude,
            "/v1/messages",
            json!({
                "model": "claude-3-7-sonnet-20250219",
                "stream": true,
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "hello"}]
            }),
            &HeaderMap::new(),
            vec![provider],
            ForwardOptions {
                max_retries: 1,
                request_timeout: Some(Duration::from_secs(5)),
                bypass_circuit_breaker: true,
            },
            RectifierConfig::default(),
        )
        .await
        .expect("a non-emitting lifecycle item must not consume the retry opportunity");

    assert_eq!(hits.count.load(Ordering::SeqCst), 2);
    server.abort();
}

#[tokio::test]
async fn web_search_terminal_validation_rejects_single_chunk_above_hard_limit() {
    const MAX_TERMINAL_VALIDATION_BYTES: usize = 4 * 1024 * 1024;
    let oversized_terminal = format!(
        ":{}\nevent: response.completed\ndata: {{\"type\":\"response.completed\",\"response\":{{\"status\":\"completed\"}}}}\n\n",
        "x".repeat(MAX_TERMINAL_VALIDATION_BYTES)
    );
    let (base_url, hits, _bodies, server) = spawn_scripted_streaming_upstream(vec![(
        StatusCode::OK,
        ScriptedStreamingBody::OwnedSse(oversized_terminal),
    )])
    .await;
    let provider = claude_provider("p1", &base_url, Some("openai_responses"));
    let (db, router) = test_router().await;
    let forwarder = RequestForwarder::new(router).expect("create forwarder");
    db.save_provider("claude", &provider)
        .expect("save provider for health tracking");

    let error = forwarder
        .forward_response(
            &AppType::Claude,
            "/v1/messages",
            json!({
                "model": "claude-3-7-sonnet-20250219",
                "stream": true,
                "max_tokens": 32,
                "tools": [{"type": "web_search_20250305", "name": "web_search"}],
                "messages": [{"role": "user", "content": "search"}]
            }),
            &HeaderMap::new(),
            vec![provider],
            ForwardOptions {
                max_retries: 0,
                request_timeout: Some(Duration::from_secs(5)),
                bypass_circuit_breaker: true,
            },
            RectifierConfig::default(),
        )
        .await
        .expect_err("a terminal event must not bypass the WebSearch pre-read hard limit");

    assert!(error
        .to_string()
        .contains("Responses stream exceeded 4194304 bytes"));
    assert_eq!(hits.count.load(Ordering::SeqCst), 1);
    server.abort();
}

#[tokio::test]
async fn web_search_terminal_validation_drops_unbounded_post_terminal_tail() {
    const TERMINAL: &str = concat!(
        "event: response.completed\n",
        "data: {\"type\":\"response.completed\",",
        "\"response\":{\"status\":\"completed\"}}\n\n"
    );
    let tail = Bytes::from(vec![b'x'; 5 * 1024 * 1024]);
    let (base_url, hits, _bodies, server) = spawn_scripted_streaming_upstream(vec![(
        StatusCode::OK,
        ScriptedStreamingBody::Chunks(vec![Bytes::from_static(TERMINAL.as_bytes()), tail]),
    )])
    .await;
    let provider = claude_provider("p1", &base_url, Some("openai_responses"));
    let (db, router) = test_router().await;
    let forwarder = RequestForwarder::new(router).expect("create forwarder");
    db.save_provider("claude", &provider)
        .expect("save provider for health tracking");

    let result = forwarder
        .forward_response(
            &AppType::Claude,
            "/v1/messages",
            json!({
                "model": "claude-3-7-sonnet-20250219",
                "stream": true,
                "max_tokens": 32,
                "tools": [{"type": "web_search_20250305", "name": "web_search"}],
                "messages": [{"role": "user", "content": "search"}]
            }),
            &HeaderMap::new(),
            vec![provider],
            ForwardOptions {
                max_retries: 0,
                request_timeout: Some(Duration::from_secs(5)),
                bypass_circuit_breaker: true,
            },
            RectifierConfig::default(),
        )
        .await
        .expect("terminal WebSearch prefix should be accepted");
    let StreamingResponse::Live(response) = result.response else {
        panic!("WebSearch terminal prefix should remain a finite live response");
    };
    let body = response
        .bytes_stream()
        .map(|chunk| chunk.expect("read terminal prefix"))
        .collect::<Vec<_>>()
        .await
        .concat();

    assert_eq!(body.as_slice(), TERMINAL.as_bytes());
    assert_eq!(hits.count.load(Ordering::SeqCst), 1);
    server.abort();
}

#[test]
fn mixed_responses_error_envelopes_preserve_retryable_type() {
    for body in [
        json!({"code": "server_error", "message": "busy"}),
        json!({"error": "overloaded_error", "message": "busy"}),
        json!({"type": "error", "code": "service_unavailable_error", "error": {"message": "busy"}}),
    ] {
        let encoded = serde_json::to_vec(&body).expect("encode error fixture");
        let error = super::super::validate_responses_success_body(&encoded)
            .expect_err("temporary error envelope must not be accepted as success");
        assert!(matches!(
            error,
            ProxyError::UpstreamError { status: 503, .. }
        ));
    }

    let event = concat!(
        "event: response.failed\n",
        "data: {\"type\":\"response.failed\",\"status\":\"failed\",",
        "\"code\":\"server_error\",\"message\":\"busy\"}\n\n"
    );
    let error = super::super::inspect_responses_start_event(event)
        .expect("inspect error event")
        .expect_err("server_error event must fail semantic validation");
    assert!(matches!(
        error,
        ProxyError::UpstreamError { status: 503, .. }
    ));
}

#[tokio::test]
async fn responses_semantic_retry_budget_is_shared_across_provider_failover() {
    const OVERLOADED: &str = concat!(
        "event: response.failed\n",
        "data: {\"type\":\"response.failed\",\"response\":{\"status\":\"failed\",",
        "\"error\":{\"type\":\"overloaded_error\",\"message\":\"busy\"}}}\n\n"
    );
    const SUCCESS: &str = concat!(
        "event: response.output_text.delta\n",
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\n\n"
    );
    let (first_url, first_hits, _first_bodies, first_server) =
        spawn_scripted_streaming_upstream(vec![
            (StatusCode::OK, ScriptedStreamingBody::Sse(OVERLOADED)),
            (StatusCode::OK, ScriptedStreamingBody::Sse(OVERLOADED)),
            (StatusCode::OK, ScriptedStreamingBody::Sse(OVERLOADED)),
        ])
        .await;
    let (second_url, second_hits, _second_bodies, second_server) =
        spawn_scripted_streaming_upstream(vec![(
            StatusCode::OK,
            ScriptedStreamingBody::Sse(OVERLOADED),
        )])
        .await;
    let (third_url, third_hits, _third_bodies, third_server) = spawn_scripted_streaming_upstream(
        vec![(StatusCode::OK, ScriptedStreamingBody::Sse(SUCCESS))],
    )
    .await;
    let first = claude_provider("p1", &first_url, Some("openai_responses"));
    let second = claude_provider("p2", &second_url, Some("openai_responses"));
    let third = claude_provider("p3", &third_url, Some("openai_responses"));
    let (db, router) = test_router().await;
    let forwarder = RequestForwarder::new(router).expect("create forwarder");
    for provider in [&first, &second, &third] {
        db.save_provider("claude", provider)
            .expect("save provider for health tracking");
    }

    let result = forwarder
        .forward_response(
            &AppType::Claude,
            "/v1/messages",
            json!({
                "model": "claude-3-7-sonnet-20250219",
                "stream": true,
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "hello"}]
            }),
            &HeaderMap::new(),
            vec![first, second, third],
            ForwardOptions {
                max_retries: 2,
                request_timeout: Some(Duration::from_secs(10)),
                bypass_circuit_breaker: false,
            },
            RectifierConfig::default(),
        )
        .await
        .expect("failover should continue after the shared semantic retry budget is exhausted");

    assert_eq!(result.provider.id, "p3");
    assert_eq!(first_hits.count.load(Ordering::SeqCst), 3);
    assert_eq!(second_hits.count.load(Ordering::SeqCst), 1);
    assert_eq!(third_hits.count.load(Ordering::SeqCst), 1);
    first_server.abort();
    second_server.abort();
    third_server.abort();
}

#[tokio::test]
async fn web_search_overload_before_terminal_is_retried_before_client_commit() {
    const SEARCH_THEN_FAILED: &str = concat!(
        "event: response.output_item.done\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,",
        "\"item\":{\"type\":\"web_search_call\",\"id\":\"ws_1\",\"status\":\"completed\"}}\n\n",
        "event: response.failed\n",
        "data: {\"type\":\"response.failed\",\"response\":{\"status\":\"failed\",",
        "\"error\":{\"type\":\"service_unavailable_error\",\"message\":\"busy\"}}}\n\n"
    );
    const SUCCESS: &str = concat!(
        "event: response.output_item.done\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,",
        "\"item\":{\"type\":\"web_search_call\",\"id\":\"ws_2\",\"status\":\"completed\"}}\n\n",
        "event: response.completed\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\n"
    );
    let (base_url, hits, _bodies, server) = spawn_scripted_streaming_upstream(vec![
        (
            StatusCode::OK,
            ScriptedStreamingBody::Sse(SEARCH_THEN_FAILED),
        ),
        (StatusCode::OK, ScriptedStreamingBody::Sse(SUCCESS)),
    ])
    .await;
    let provider = claude_provider("p1", &base_url, Some("openai_responses"));
    let (db, router) = test_router().await;
    let forwarder = RequestForwarder::new(router).expect("create forwarder");
    db.save_provider("claude", &provider)
        .expect("save provider for health tracking");

    forwarder
        .forward_response(
            &AppType::Claude,
            "/v1/messages",
            json!({
                "model": "claude-3-7-sonnet-20250219",
                "stream": true,
                "max_tokens": 32,
                "tools": [{"type": "web_search_20250305", "name": "web_search"}],
                "messages": [{"role": "user", "content": "search"}]
            }),
            &HeaderMap::new(),
            vec![provider],
            ForwardOptions {
                max_retries: 1,
                request_timeout: Some(Duration::from_secs(5)),
                bypass_circuit_breaker: true,
            },
            RectifierConfig::default(),
        )
        .await
        .expect("WebSearch overload before terminal should recover before client commit");

    assert_eq!(hits.count.load(Ordering::SeqCst), 2);
    server.abort();
}
