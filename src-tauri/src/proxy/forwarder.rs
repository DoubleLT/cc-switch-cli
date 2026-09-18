use axum::http::HeaderMap;
use bytes::Bytes;
use futures::{stream::BoxStream, StreamExt};
use serde_json::{json, Value};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::{Duration, Instant},
};

use crate::{app_config::AppType, provider::Provider};

use super::{
    error::ProxyError,
    provider_router::ProviderRouter,
    providers::codex_chat_history::CodexChatHistoryStore,
    providers::gemini_shadow::GeminiShadowStore,
    providers::get_adapter,
    response::decode_buffered_response_body,
    thinking_budget_rectifier::{rectify_thinking_budget, should_rectify_thinking_budget},
    thinking_rectifier::{
        normalize_thinking_type, rectify_anthropic_request, should_rectify_thinking_signature,
    },
    types::{CopilotOptimizerConfig, OptimizerConfig, RectifierConfig},
};

mod request_builder;

pub struct RequestForwarder {
    router: Arc<ProviderRouter>,
    optimizer_config: OptimizerConfig,
    copilot_optimizer_config: CopilotOptimizerConfig,
    session_id: String,
    session_client_provided: bool,
    codex_chat_history: Option<Arc<CodexChatHistoryStore>>,
    gemini_shadow: Option<Arc<GeminiShadowStore>>,
}

#[derive(Debug, Clone, Copy)]
pub struct ForwardOptions {
    pub max_retries: u32,
    pub request_timeout: Option<Duration>,
    pub bypass_circuit_breaker: bool,
}

const RESPONSES_RETRY_BACKOFF_BASE: Duration = Duration::from_secs(1);
const RESPONSES_RETRY_BACKOFF_MAX: Duration = Duration::from_secs(4);
const RESPONSES_SEMANTIC_RETRY_LIMIT: u32 = 3;
const RESPONSES_RETRY_FALLBACK_TIMEOUT: Duration = Duration::from_secs(90);
const CODEX_ANTHROPIC_JSON_BODY_LIMIT: usize = 16 * 1024 * 1024;

struct ResponsesRetryState {
    remaining: u32,
    used: u32,
    deadline_started_at: Option<Instant>,
}

impl ResponsesRetryState {
    fn new(enabled: bool, configured_retries: u32) -> Self {
        let remaining = if enabled {
            configured_retries.min(RESPONSES_SEMANTIC_RETRY_LIMIT)
        } else {
            0
        };
        Self {
            remaining,
            used: 0,
            deadline_started_at: None,
        }
    }

    fn with_fallback_deadline(mut self) -> Self {
        if self.remaining > 0 {
            self.deadline_started_at = Some(Instant::now());
        }
        self
    }

    fn timeout_origin(&self, provider_started_at: Instant) -> Instant {
        self.deadline_started_at.unwrap_or(provider_started_at)
    }

    fn effective_timeout(&self, configured_timeout: Option<Duration>) -> Option<Duration> {
        if self.deadline_started_at.is_some() {
            configured_timeout.or(Some(RESPONSES_RETRY_FALLBACK_TIMEOUT))
        } else {
            configured_timeout
        }
    }

    async fn wait_to_retry(
        &mut self,
        error: &ProxyError,
        provider_started_at: Instant,
        configured_timeout: Option<Duration>,
    ) -> Result<bool, ProxyError> {
        if !is_retryable_responses_pre_output_error(error) || self.remaining == 0 {
            return Ok(false);
        }

        let retry_started_at = provider_started_at;
        let deadline_started_at = *self.deadline_started_at.get_or_insert(retry_started_at);
        let delay = responses_retry_backoff(self.used);
        wait_for_responses_retry(
            delay,
            deadline_started_at,
            configured_timeout.or(Some(RESPONSES_RETRY_FALLBACK_TIMEOUT)),
        )
        .await?;
        self.remaining -= 1;
        self.used += 1;
        Ok(true)
    }

    fn active_remaining_timeout(
        &self,
        configured_timeout: Option<Duration>,
    ) -> Result<Option<Duration>, ProxyError> {
        let Some(started_at) = self.deadline_started_at else {
            return Ok(None);
        };
        let timeout = configured_timeout.unwrap_or(RESPONSES_RETRY_FALLBACK_TIMEOUT);
        let remaining = timeout.saturating_sub(started_at.elapsed());
        if remaining.is_zero() {
            return Err(request_timeout_error(timeout));
        }
        Ok(Some(remaining))
    }
}

#[derive(Debug)]
pub struct BufferedResponse {
    pub status: reqwest::StatusCode,
    pub headers: reqwest::header::HeaderMap,
    pub body: Bytes,
}

#[derive(Debug)]
pub struct ForwardedResponse<T> {
    pub provider: Provider,
    pub response: T,
}

#[derive(Debug)]
pub struct ForwardFailure {
    pub provider: Option<Provider>,
    pub error: ProxyError,
}

impl ForwardFailure {
    fn new(provider: Option<Provider>, error: ProxyError) -> Self {
        Self { provider, error }
    }
}

pub struct LiveResponse {
    status: reqwest::StatusCode,
    headers: reqwest::header::HeaderMap,
    stream: BoxStream<'static, Result<Bytes, reqwest::Error>>,
}

impl std::fmt::Debug for LiveResponse {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LiveResponse")
            .field("status", &self.status)
            .field("headers", &self.headers)
            .finish_non_exhaustive()
    }
}

impl LiveResponse {
    fn from_reqwest(response: reqwest::Response) -> Self {
        let status = response.status();
        let headers = response.headers().clone();
        Self {
            status,
            headers,
            stream: response.bytes_stream().boxed(),
        }
    }

    fn from_stream(
        status: reqwest::StatusCode,
        headers: reqwest::header::HeaderMap,
        stream: impl futures::Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
    ) -> Self {
        Self {
            status,
            headers,
            stream: stream.boxed(),
        }
    }

    pub fn status(&self) -> reqwest::StatusCode {
        self.status
    }

    pub fn headers(&self) -> &reqwest::header::HeaderMap {
        &self.headers
    }

    pub fn bytes_stream(self) -> BoxStream<'static, Result<Bytes, reqwest::Error>> {
        self.stream
    }
}

#[derive(Debug)]
pub enum StreamingResponse {
    Live(LiveResponse),
    Buffered(BufferedResponse),
}

impl StreamingResponse {
    pub fn status(&self) -> reqwest::StatusCode {
        match self {
            Self::Live(response) => response.status(),
            Self::Buffered(response) => response.status,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AttemptDecision {
    ProviderFailure,
    NeutralRelease,
    FatalStop,
}

enum BufferedRequestError {
    BeforeResponse(ProxyError),
    AfterResponse(ProxyError),
}

enum StreamingRequestError {
    BeforeResponse(ProxyError),
    AfterResponse(ProxyError),
}

struct BufferedAttemptOutcome {
    response: BufferedResponse,
    attempt_decision: AttemptDecision,
}

struct StreamingAttemptOutcome {
    response: StreamingResponse,
    attempt_decision: AttemptDecision,
}

impl RequestForwarder {
    pub fn new(router: Arc<ProviderRouter>) -> Result<Self, ProxyError> {
        Ok(Self {
            router,
            optimizer_config: OptimizerConfig::default(),
            copilot_optimizer_config: CopilotOptimizerConfig::default(),
            session_id: String::new(),
            session_client_provided: false,
            codex_chat_history: None,
            gemini_shadow: None,
        })
    }

    pub(super) fn prewarm_provider_clients(&self, app_type: &AppType, providers: &[Provider]) {
        for provider in providers {
            let _ = self.client_for_provider(app_type, provider);
        }
    }

    pub fn with_optimizer_config(mut self, optimizer_config: OptimizerConfig) -> Self {
        self.optimizer_config = optimizer_config;
        self
    }

    pub fn with_copilot_optimizer_config(
        mut self,
        copilot_optimizer_config: CopilotOptimizerConfig,
    ) -> Self {
        self.copilot_optimizer_config = copilot_optimizer_config;
        self
    }

    pub fn with_session(mut self, session_id: String, client_provided: bool) -> Self {
        self.session_id = session_id;
        self.session_client_provided = client_provided;
        self
    }

    pub fn with_codex_chat_history(mut self, history: Arc<CodexChatHistoryStore>) -> Self {
        self.codex_chat_history = Some(history);
        self
    }

    pub fn with_gemini_shadow(mut self, shadow: Arc<GeminiShadowStore>) -> Self {
        self.gemini_shadow = Some(shadow);
        self
    }

    #[cfg(test)]
    #[expect(
        clippy::too_many_arguments,
        reason = "test helper mirrors proxy forwarding inputs"
    )]
    pub async fn forward_response(
        &self,
        app_type: &AppType,
        endpoint: &str,
        body: Value,
        headers: &HeaderMap,
        providers: Vec<Provider>,
        options: ForwardOptions,
        rectifier_config: RectifierConfig,
    ) -> Result<ForwardedResponse<StreamingResponse>, ProxyError> {
        self.forward_response_detailed(
            app_type,
            endpoint,
            body,
            headers,
            providers,
            options,
            rectifier_config,
        )
        .await
        .map_err(|failure| failure.error)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "forwarding requires request, provider, and retry options"
    )]
    pub async fn forward_response_detailed(
        &self,
        app_type: &AppType,
        endpoint: &str,
        body: Value,
        headers: &HeaderMap,
        providers: Vec<Provider>,
        options: ForwardOptions,
        rectifier_config: RectifierConfig,
    ) -> Result<ForwardedResponse<StreamingResponse>, ForwardFailure> {
        if providers.is_empty() {
            return Err(ForwardFailure::new(None, ProxyError::NoAvailableProvider));
        }

        let claude_error_path = matches!(app_type, AppType::Claude);
        let bypass_circuit_breaker = options.bypass_circuit_breaker;
        let mut last_error = None;
        let mut attempted_provider = false;
        let mut attempted_providers = 0usize;
        let mut pending_upstream_response = None;
        let max_attempts = (options.max_retries as usize).saturating_add(1);
        let mut responses_retry_state =
            ResponsesRetryState::new(matches!(app_type, AppType::Claude), options.max_retries);
        let mut codex_anthropic_retry_state =
            ResponsesRetryState::new(matches!(app_type, AppType::Codex), options.max_retries)
                .with_fallback_deadline();

        for provider in providers {
            if attempted_providers >= max_attempts {
                break;
            }

            let permit = if bypass_circuit_breaker {
                super::circuit_breaker::AllowResult {
                    allowed: true,
                    used_half_open_permit: false,
                }
            } else {
                self.router
                    .allow_provider_request(&provider.id, app_type.as_str())
                    .await
            };

            if !permit.allowed {
                continue;
            }

            attempted_provider = true;
            attempted_providers += 1;
            pending_upstream_response = None;
            let provider_needs_transform = matches!(app_type, AppType::Claude)
                && get_adapter(app_type).needs_transform(&provider);
            let retry_state = if uses_codex_anthropic_protocol(app_type, &provider, endpoint) {
                &mut codex_anthropic_retry_state
            } else {
                &mut responses_retry_state
            };
            match self
                .send_streaming_request(
                    app_type,
                    &provider,
                    endpoint,
                    &body,
                    headers,
                    ForwardOptions {
                        max_retries: 0,
                        ..options
                    },
                    retry_state,
                    &rectifier_config,
                )
                .await
            {
                Ok(outcome) => {
                    let response = outcome.response;
                    if response.status().is_success() {
                        if !bypass_circuit_breaker {
                            let _ = self
                                .router
                                .record_result(
                                    &provider.id,
                                    app_type.as_str(),
                                    permit.used_half_open_permit,
                                    true,
                                    None,
                                )
                                .await;
                        }

                        return Ok(ForwardedResponse { provider, response });
                    }

                    match outcome.attempt_decision {
                        AttemptDecision::NeutralRelease => {
                            if !bypass_circuit_breaker {
                                self.router
                                    .release_permit_neutral(
                                        &provider.id,
                                        app_type.as_str(),
                                        permit.used_half_open_permit,
                                    )
                                    .await;
                            }

                            if claude_error_path && !provider_needs_transform {
                                return Err(ForwardFailure::new(
                                    Some(provider),
                                    streaming_response_to_upstream_error(response),
                                ));
                            }

                            return Ok(ForwardedResponse { provider, response });
                        }
                        AttemptDecision::ProviderFailure => {
                            if !bypass_circuit_breaker {
                                let _ = self
                                    .router
                                    .record_result(
                                        &provider.id,
                                        app_type.as_str(),
                                        permit.used_half_open_permit,
                                        false,
                                        Some(format!(
                                            "upstream returned {}",
                                            response.status().as_u16()
                                        )),
                                    )
                                    .await;
                            }

                            if claude_error_path && !provider_needs_transform {
                                last_error = Some(ForwardFailure::new(
                                    Some(provider.clone()),
                                    streaming_response_to_upstream_error(response),
                                ));
                            } else {
                                pending_upstream_response =
                                    Some(ForwardedResponse { provider, response });
                                last_error = Some(ForwardFailure::new(
                                    pending_upstream_response
                                        .as_ref()
                                        .map(|response| response.provider.clone()),
                                    ProxyError::UpstreamError {
                                        status: pending_upstream_response
                                            .as_ref()
                                            .expect("pending upstream response")
                                            .response
                                            .status()
                                            .as_u16(),
                                        body: None,
                                    },
                                ));
                            }
                            continue;
                        }
                        _ => {
                            if !bypass_circuit_breaker {
                                let _ = self
                                    .router
                                    .record_result(
                                        &provider.id,
                                        app_type.as_str(),
                                        permit.used_half_open_permit,
                                        false,
                                        Some(format!(
                                            "upstream returned {}",
                                            response.status().as_u16()
                                        )),
                                    )
                                    .await;
                            }

                            return Ok(ForwardedResponse { provider, response });
                        }
                    }
                }
                Err(StreamingRequestError::BeforeResponse(error))
                | Err(StreamingRequestError::AfterResponse(error)) => {
                    match classify_attempt_error(&error, app_type, &provider) {
                        AttemptDecision::ProviderFailure => {
                            if !bypass_circuit_breaker {
                                let _ = self
                                    .router
                                    .record_result(
                                        &provider.id,
                                        app_type.as_str(),
                                        permit.used_half_open_permit,
                                        false,
                                        Some(error.to_string()),
                                    )
                                    .await;
                            }
                            last_error = Some(ForwardFailure::new(Some(provider.clone()), error));
                        }
                        AttemptDecision::NeutralRelease | AttemptDecision::FatalStop => {
                            if !bypass_circuit_breaker {
                                self.router
                                    .release_permit_neutral(
                                        &provider.id,
                                        app_type.as_str(),
                                        permit.used_half_open_permit,
                                    )
                                    .await;
                            }
                            return Err(ForwardFailure::new(Some(provider), error));
                        }
                    }
                }
            }
        }

        if let Some(response) = pending_upstream_response {
            return Ok(response);
        }

        if attempted_provider {
            Err(last_error
                .unwrap_or_else(|| ForwardFailure::new(None, ProxyError::NoAvailableProvider)))
        } else {
            Err(ForwardFailure::new(None, ProxyError::NoAvailableProvider))
        }
    }

    #[allow(dead_code)]
    #[expect(
        clippy::too_many_arguments,
        reason = "forwarding requires request, provider, and retry options"
    )]
    pub async fn forward_buffered_response(
        &self,
        app_type: &AppType,
        endpoint: &str,
        body: Value,
        headers: &HeaderMap,
        providers: Vec<Provider>,
        options: ForwardOptions,
        rectifier_config: RectifierConfig,
    ) -> Result<ForwardedResponse<BufferedResponse>, ProxyError> {
        self.forward_buffered_response_detailed(
            app_type,
            endpoint,
            body,
            headers,
            providers,
            options,
            rectifier_config,
        )
        .await
        .map_err(|failure| failure.error)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "forwarding requires request, provider, and retry options"
    )]
    pub async fn forward_buffered_response_detailed(
        &self,
        app_type: &AppType,
        endpoint: &str,
        body: Value,
        headers: &HeaderMap,
        providers: Vec<Provider>,
        options: ForwardOptions,
        rectifier_config: RectifierConfig,
    ) -> Result<ForwardedResponse<BufferedResponse>, ForwardFailure> {
        if providers.is_empty() {
            return Err(ForwardFailure::new(None, ProxyError::NoAvailableProvider));
        }

        let claude_error_path = matches!(app_type, AppType::Claude);
        let bypass_circuit_breaker = options.bypass_circuit_breaker;
        let mut last_error = None;
        let mut attempted_provider = false;
        let mut attempted_providers = 0usize;
        let mut pending_upstream_response = None;
        let max_attempts = (options.max_retries as usize).saturating_add(1);
        let mut responses_retry_state =
            ResponsesRetryState::new(matches!(app_type, AppType::Claude), options.max_retries);
        let mut codex_anthropic_retry_state =
            ResponsesRetryState::new(matches!(app_type, AppType::Codex), options.max_retries)
                .with_fallback_deadline();

        for provider in providers {
            if attempted_providers >= max_attempts {
                break;
            }

            let permit = if bypass_circuit_breaker {
                super::circuit_breaker::AllowResult {
                    allowed: true,
                    used_half_open_permit: false,
                }
            } else {
                self.router
                    .allow_provider_request(&provider.id, app_type.as_str())
                    .await
            };

            if !permit.allowed {
                continue;
            }

            attempted_provider = true;
            attempted_providers += 1;
            pending_upstream_response = None;
            let provider_needs_transform = matches!(app_type, AppType::Claude)
                && get_adapter(app_type).needs_transform(&provider);
            let retry_state = if uses_codex_anthropic_protocol(app_type, &provider, endpoint) {
                &mut codex_anthropic_retry_state
            } else {
                &mut responses_retry_state
            };

            match self
                .send_buffered_request(
                    app_type,
                    &provider,
                    endpoint,
                    &body,
                    headers,
                    ForwardOptions {
                        max_retries: 0,
                        ..options
                    },
                    retry_state,
                    &rectifier_config,
                )
                .await
            {
                Ok(outcome) => {
                    let response = outcome.response;
                    if response.status.is_success() {
                        if !bypass_circuit_breaker {
                            let _ = self
                                .router
                                .record_result(
                                    &provider.id,
                                    app_type.as_str(),
                                    permit.used_half_open_permit,
                                    true,
                                    None,
                                )
                                .await;
                        }

                        return Ok(ForwardedResponse { provider, response });
                    }

                    match outcome.attempt_decision {
                        AttemptDecision::NeutralRelease => {
                            if !bypass_circuit_breaker {
                                self.router
                                    .release_permit_neutral(
                                        &provider.id,
                                        app_type.as_str(),
                                        permit.used_half_open_permit,
                                    )
                                    .await;
                            }

                            if claude_error_path && !provider_needs_transform {
                                return Err(ForwardFailure::new(
                                    Some(provider),
                                    buffered_response_to_upstream_error(response),
                                ));
                            }

                            return Ok(ForwardedResponse { provider, response });
                        }
                        AttemptDecision::ProviderFailure => {
                            if !bypass_circuit_breaker {
                                let _ = self
                                    .router
                                    .record_result(
                                        &provider.id,
                                        app_type.as_str(),
                                        permit.used_half_open_permit,
                                        false,
                                        Some(format!(
                                            "upstream returned {}",
                                            response.status.as_u16()
                                        )),
                                    )
                                    .await;
                            }

                            if claude_error_path && !provider_needs_transform {
                                last_error = Some(ForwardFailure::new(
                                    Some(provider.clone()),
                                    buffered_response_to_upstream_error(response),
                                ));
                            } else {
                                pending_upstream_response =
                                    Some(ForwardedResponse { provider, response });
                                last_error = Some(ForwardFailure::new(
                                    pending_upstream_response
                                        .as_ref()
                                        .map(|response| response.provider.clone()),
                                    ProxyError::UpstreamError {
                                        status: pending_upstream_response
                                            .as_ref()
                                            .expect("pending upstream response")
                                            .response
                                            .status
                                            .as_u16(),
                                        body: None,
                                    },
                                ));
                            }
                            continue;
                        }
                        _ => {
                            if !bypass_circuit_breaker {
                                let _ = self
                                    .router
                                    .record_result(
                                        &provider.id,
                                        app_type.as_str(),
                                        permit.used_half_open_permit,
                                        false,
                                        Some(format!(
                                            "upstream returned {}",
                                            response.status.as_u16()
                                        )),
                                    )
                                    .await;
                            }

                            return Ok(ForwardedResponse { provider, response });
                        }
                    }
                }
                Err(BufferedRequestError::BeforeResponse(error))
                | Err(BufferedRequestError::AfterResponse(error)) => {
                    match classify_attempt_error(&error, app_type, &provider) {
                        AttemptDecision::ProviderFailure => {
                            if !bypass_circuit_breaker {
                                let _ = self
                                    .router
                                    .record_result(
                                        &provider.id,
                                        app_type.as_str(),
                                        permit.used_half_open_permit,
                                        false,
                                        Some(error.to_string()),
                                    )
                                    .await;
                            }
                            last_error = Some(ForwardFailure::new(Some(provider.clone()), error));
                        }
                        AttemptDecision::NeutralRelease | AttemptDecision::FatalStop => {
                            if !bypass_circuit_breaker {
                                self.router
                                    .release_permit_neutral(
                                        &provider.id,
                                        app_type.as_str(),
                                        permit.used_half_open_permit,
                                    )
                                    .await;
                            }
                            return Err(ForwardFailure::new(Some(provider), error));
                        }
                    }
                }
            }
        }

        if let Some(response) = pending_upstream_response {
            return Ok(response);
        }

        if attempted_provider {
            Err(last_error
                .unwrap_or_else(|| ForwardFailure::new(None, ProxyError::NoAvailableProvider)))
        } else {
            Err(ForwardFailure::new(None, ProxyError::NoAvailableProvider))
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "request execution needs provider, endpoint, headers, and retry options"
    )]
    async fn send_streaming_request(
        &self,
        app_type: &AppType,
        provider: &Provider,
        endpoint: &str,
        body: &Value,
        headers: &HeaderMap,
        options: ForwardOptions,
        responses_retry_state: &mut ResponsesRetryState,
        rectifier_config: &RectifierConfig,
    ) -> Result<StreamingAttemptOutcome, StreamingRequestError> {
        // Provider-specific clients may need to load native roots. Build and
        // retain this one before the upstream request timeout starts.
        let client = self.client_for_provider(app_type, provider);
        let provider_started_at = Instant::now();
        let allow_transport_retry = uses_internal_transport_retry(app_type);
        let mut request_body = body.clone();
        let mut rectifier_retried = false;

        'request_loop: loop {
            let preparation_timeout = responses_retry_state
                .active_remaining_timeout(options.request_timeout)
                .map_err(|error| {
                    if rectifier_retried {
                        StreamingRequestError::AfterResponse(error)
                    } else {
                        StreamingRequestError::BeforeResponse(error)
                    }
                })?;
            let preparation = self.prepare_request_with_client(
                app_type,
                provider,
                &client,
                endpoint,
                &request_body,
                headers,
                options,
            );
            let base_request = match preparation_timeout {
                Some(remaining) => {
                    tokio::time::timeout(remaining, preparation)
                        .await
                        .map_err(|_| {
                            let error = request_timeout_error(
                                options
                                    .request_timeout
                                    .unwrap_or(RESPONSES_RETRY_FALLBACK_TIMEOUT),
                            );
                            if rectifier_retried {
                                StreamingRequestError::AfterResponse(error)
                            } else {
                                StreamingRequestError::BeforeResponse(error)
                            }
                        })?
                }
                None => preparation.await,
            }
            .map_err(StreamingRequestError::BeforeResponse)?;
            let mut attempt = 0u32;

            loop {
                let request_timeout =
                    responses_retry_state.effective_timeout(options.request_timeout);
                let attempt_started_at = if allow_transport_retry
                    && responses_retry_state.deadline_started_at.is_none()
                {
                    Instant::now()
                } else {
                    responses_retry_state.timeout_origin(provider_started_at)
                };
                let remaining_timeout = match request_timeout {
                    Some(request_timeout) => {
                        let remaining_timeout =
                            request_timeout.saturating_sub(attempt_started_at.elapsed());
                        if remaining_timeout.is_zero() {
                            let timeout_error = request_timeout_error(request_timeout);
                            return Err(if rectifier_retried {
                                StreamingRequestError::AfterResponse(timeout_error)
                            } else {
                                StreamingRequestError::BeforeResponse(timeout_error)
                            });
                        }
                        Some(remaining_timeout)
                    }
                    None => None,
                };

                let request =
                    clone_request(&base_request).map_err(StreamingRequestError::BeforeResponse)?;

                match match remaining_timeout {
                    Some(remaining_timeout) => {
                        tokio::time::timeout(remaining_timeout, request.send())
                            .await
                            .map_err(|_| ())
                    }
                    None => Ok(request.send().await),
                } {
                    Ok(Ok(response)) => {
                        if response.status().is_success() {
                            if uses_codex_anthropic_protocol(app_type, provider, endpoint)
                                && response_is_json(&response)
                            {
                                let buffered_response = read_buffered_response(
                                    response,
                                    attempt_started_at,
                                    request_timeout.or(Some(RESPONSES_RETRY_FALLBACK_TIMEOUT)),
                                    Some(CODEX_ANTHROPIC_JSON_BODY_LIMIT),
                                    true,
                                )
                                .await
                                .map_err(StreamingRequestError::AfterResponse)?;
                                if let Err(error) =
                                    validate_codex_anthropic_success_body(&buffered_response.body)
                                {
                                    match responses_retry_state
                                        .wait_to_retry(
                                            &error,
                                            provider_started_at,
                                            options.request_timeout,
                                        )
                                        .await
                                    {
                                        Ok(true) => continue,
                                        Ok(false) => {}
                                        Err(timeout_error) => {
                                            return Err(if rectifier_retried {
                                                StreamingRequestError::AfterResponse(timeout_error)
                                            } else {
                                                StreamingRequestError::BeforeResponse(timeout_error)
                                            });
                                        }
                                    }
                                    return Err(if rectifier_retried {
                                        StreamingRequestError::AfterResponse(error)
                                    } else {
                                        StreamingRequestError::BeforeResponse(error)
                                    });
                                }
                                return Ok(StreamingAttemptOutcome {
                                    response: StreamingResponse::Buffered(buffered_response),
                                    attempt_decision: AttemptDecision::FatalStop,
                                });
                            }
                            let response = match prepare_success_streaming_response(
                                response,
                                attempt_started_at,
                                request_timeout,
                                uses_responses_protocol(app_type, provider, endpoint),
                                uses_codex_anthropic_protocol(app_type, provider, endpoint),
                                anthropic_request_uses_web_search(app_type, body),
                                responses_retry_state.deadline_started_at.is_some(),
                            )
                            .await
                            {
                                Ok(response) => response,
                                Err(error) => {
                                    match responses_retry_state
                                        .wait_to_retry(
                                            &error,
                                            provider_started_at,
                                            options.request_timeout,
                                        )
                                        .await
                                    {
                                        Ok(true) => {
                                            log::warn!(
                                                "Responses upstream was temporarily unavailable before output; retrying provider {} ({}/{})",
                                                provider.id,
                                                responses_retry_state.used,
                                                responses_retry_state.used
                                                    + responses_retry_state.remaining
                                            );
                                            continue;
                                        }
                                        Ok(false) => {}
                                        Err(timeout_error) => {
                                            return Err(if rectifier_retried {
                                                StreamingRequestError::AfterResponse(timeout_error)
                                            } else {
                                                StreamingRequestError::BeforeResponse(timeout_error)
                                            });
                                        }
                                    }
                                    return Err(if rectifier_retried {
                                        StreamingRequestError::AfterResponse(error)
                                    } else {
                                        StreamingRequestError::BeforeResponse(error)
                                    });
                                }
                            };
                            return Ok(StreamingAttemptOutcome {
                                response: StreamingResponse::Live(response),
                                attempt_decision: AttemptDecision::FatalStop,
                            });
                        }

                        if should_buffer_streaming_error_response(app_type, response.status()) {
                            let buffered_response = read_buffered_response(
                                response,
                                attempt_started_at,
                                request_timeout,
                                None,
                                true,
                            )
                            .await
                            .map_err(StreamingRequestError::AfterResponse)?;

                            if uses_responses_protocol(app_type, provider, endpoint) {
                                if let Err(error) =
                                    validate_buffered_responses_body(&buffered_response.body)
                                {
                                    match responses_retry_state
                                        .wait_to_retry(
                                            &error,
                                            provider_started_at,
                                            options.request_timeout,
                                        )
                                        .await
                                    {
                                        Ok(true) => {
                                            log::warn!(
                                                "Responses upstream was temporarily unavailable before output; retrying provider {} ({}/{})",
                                                provider.id,
                                                responses_retry_state.used,
                                                responses_retry_state.used
                                                    + responses_retry_state.remaining
                                            );
                                            continue;
                                        }
                                        Ok(false)
                                            if is_retryable_responses_pre_output_error(&error) =>
                                        {
                                            return Err(if rectifier_retried {
                                                StreamingRequestError::AfterResponse(error)
                                            } else {
                                                StreamingRequestError::BeforeResponse(error)
                                            });
                                        }
                                        Ok(false) => {}
                                        Err(timeout_error) => {
                                            return Err(if rectifier_retried {
                                                StreamingRequestError::AfterResponse(timeout_error)
                                            } else {
                                                StreamingRequestError::BeforeResponse(timeout_error)
                                            });
                                        }
                                    }
                                }
                            }

                            if !rectifier_retried {
                                if let Some(rectified_body) = maybe_rectify_claude_buffered_request(
                                    app_type,
                                    &buffered_response,
                                    &request_body,
                                    rectifier_config,
                                ) {
                                    rectifier_retried = true;
                                    request_body = rectified_body;
                                    continue 'request_loop;
                                }
                            }

                            return Ok(StreamingAttemptOutcome {
                                attempt_decision: classify_upstream_response(
                                    buffered_response.status,
                                    rectifier_retried,
                                    app_type,
                                    provider,
                                ),
                                response: StreamingResponse::Buffered(buffered_response),
                            });
                        }

                        return Ok(StreamingAttemptOutcome {
                            attempt_decision: classify_upstream_response(
                                response.status(),
                                rectifier_retried,
                                app_type,
                                provider,
                            ),
                            response: StreamingResponse::Live(LiveResponse::from_reqwest(response)),
                        });
                    }
                    Ok(Err(error)) => {
                        if allow_transport_retry
                            && attempt < options.max_retries
                            && is_retryable_transport_error(&error)
                        {
                            attempt += 1;
                            continue;
                        }

                        let mapped_error = map_request_send_error(error, request_timeout);
                        return Err(if rectifier_retried {
                            StreamingRequestError::AfterResponse(mapped_error)
                        } else {
                            StreamingRequestError::BeforeResponse(mapped_error)
                        });
                    }
                    Err(_) => {
                        if allow_transport_retry && attempt < options.max_retries {
                            attempt += 1;
                            continue;
                        }

                        let timeout_error = request_timeout_error(
                            request_timeout
                                .expect("request timeout should exist when timeout future errors"),
                        );
                        return Err(if rectifier_retried {
                            StreamingRequestError::AfterResponse(timeout_error)
                        } else {
                            StreamingRequestError::BeforeResponse(timeout_error)
                        });
                    }
                }
            }
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "request execution needs provider, endpoint, headers, and retry options"
    )]
    async fn send_buffered_request(
        &self,
        app_type: &AppType,
        provider: &Provider,
        endpoint: &str,
        body: &Value,
        headers: &HeaderMap,
        options: ForwardOptions,
        responses_retry_state: &mut ResponsesRetryState,
        rectifier_config: &RectifierConfig,
    ) -> Result<BufferedAttemptOutcome, BufferedRequestError> {
        // Keep provider proxy client construction outside the shared request
        // timeout and retain it for rectifier retries.
        let client = self.client_for_provider(app_type, provider);
        let mut request_body = body.clone();
        let mut rectifier_retried = false;
        let provider_started_at = Instant::now();
        let allow_transport_retry = uses_internal_transport_retry(app_type);

        'request_loop: loop {
            let preparation_timeout = responses_retry_state
                .active_remaining_timeout(options.request_timeout)
                .map_err(|error| {
                    if rectifier_retried {
                        BufferedRequestError::AfterResponse(error)
                    } else {
                        BufferedRequestError::BeforeResponse(error)
                    }
                })?;
            let preparation = self.prepare_request_with_client(
                app_type,
                provider,
                &client,
                endpoint,
                &request_body,
                headers,
                options,
            );
            let base_request = match preparation_timeout {
                Some(remaining) => {
                    tokio::time::timeout(remaining, preparation)
                        .await
                        .map_err(|_| {
                            let error = request_timeout_error(
                                options
                                    .request_timeout
                                    .unwrap_or(RESPONSES_RETRY_FALLBACK_TIMEOUT),
                            );
                            if rectifier_retried {
                                BufferedRequestError::AfterResponse(error)
                            } else {
                                BufferedRequestError::BeforeResponse(error)
                            }
                        })?
                }
                None => preparation.await,
            }
            .map_err(BufferedRequestError::BeforeResponse)?;
            let mut attempt = 0u32;

            loop {
                let request_timeout =
                    responses_retry_state.effective_timeout(options.request_timeout);
                let attempt_started_at = if allow_transport_retry
                    && responses_retry_state.deadline_started_at.is_none()
                {
                    Instant::now()
                } else {
                    responses_retry_state.timeout_origin(provider_started_at)
                };
                let remaining_timeout = match request_timeout {
                    Some(request_timeout) => {
                        let remaining_timeout =
                            request_timeout.saturating_sub(attempt_started_at.elapsed());
                        if remaining_timeout.is_zero() {
                            let timeout_error = request_timeout_error(request_timeout);
                            return Err(if rectifier_retried {
                                BufferedRequestError::AfterResponse(timeout_error)
                            } else {
                                BufferedRequestError::BeforeResponse(timeout_error)
                            });
                        }
                        Some(remaining_timeout)
                    }
                    None => None,
                };

                let request =
                    clone_request(&base_request).map_err(BufferedRequestError::BeforeResponse)?;

                match match remaining_timeout {
                    Some(remaining_timeout) => {
                        tokio::time::timeout(remaining_timeout, request.send())
                            .await
                            .map_err(|_| ())
                    }
                    None => Ok(request.send().await),
                } {
                    Ok(Ok(response)) => {
                        let body_limit = (response.status().is_success()
                            && uses_codex_anthropic_protocol(app_type, provider, endpoint))
                        .then_some(CODEX_ANTHROPIC_JSON_BODY_LIMIT);
                        let body_timeout = if body_limit.is_some() {
                            request_timeout.or(Some(RESPONSES_RETRY_FALLBACK_TIMEOUT))
                        } else {
                            request_timeout
                        };
                        let buffered_response = read_buffered_response(
                            response,
                            attempt_started_at,
                            body_timeout,
                            body_limit,
                            false,
                        )
                        .await
                        .map_err(BufferedRequestError::AfterResponse)?;

                        if !buffered_response.status.is_success()
                            && uses_responses_protocol(app_type, provider, endpoint)
                        {
                            if let Err(error) =
                                validate_buffered_responses_body(&buffered_response.body)
                            {
                                match responses_retry_state
                                    .wait_to_retry(
                                        &error,
                                        provider_started_at,
                                        options.request_timeout,
                                    )
                                    .await
                                {
                                    Ok(true) => {
                                        log::warn!(
                                            "Responses upstream was temporarily unavailable before output; retrying provider {} ({}/{})",
                                            provider.id,
                                            responses_retry_state.used,
                                            responses_retry_state.used
                                                + responses_retry_state.remaining
                                        );
                                        continue;
                                    }
                                    Ok(false)
                                        if is_retryable_responses_pre_output_error(&error) =>
                                    {
                                        return Err(if rectifier_retried {
                                            BufferedRequestError::AfterResponse(error)
                                        } else {
                                            BufferedRequestError::BeforeResponse(error)
                                        });
                                    }
                                    Ok(false) => {}
                                    Err(timeout_error) => {
                                        return Err(if rectifier_retried {
                                            BufferedRequestError::AfterResponse(timeout_error)
                                        } else {
                                            BufferedRequestError::BeforeResponse(timeout_error)
                                        });
                                    }
                                }
                            }
                        }

                        if buffered_response.status.is_success()
                            && uses_codex_anthropic_protocol(app_type, provider, endpoint)
                        {
                            if let Err(error) =
                                validate_codex_anthropic_success_body(&buffered_response.body)
                            {
                                match responses_retry_state
                                    .wait_to_retry(
                                        &error,
                                        provider_started_at,
                                        options.request_timeout,
                                    )
                                    .await
                                {
                                    Ok(true) => continue,
                                    Ok(false) => {}
                                    Err(timeout_error) => {
                                        return Err(if rectifier_retried {
                                            BufferedRequestError::AfterResponse(timeout_error)
                                        } else {
                                            BufferedRequestError::BeforeResponse(timeout_error)
                                        });
                                    }
                                }
                                return Err(if rectifier_retried {
                                    BufferedRequestError::AfterResponse(error)
                                } else {
                                    BufferedRequestError::BeforeResponse(error)
                                });
                            }
                        } else if buffered_response.status.is_success()
                            && uses_responses_protocol(app_type, provider, endpoint)
                        {
                            if let Err(error) =
                                validate_buffered_responses_body(&buffered_response.body)
                            {
                                match responses_retry_state
                                    .wait_to_retry(
                                        &error,
                                        provider_started_at,
                                        options.request_timeout,
                                    )
                                    .await
                                {
                                    Ok(true) => {
                                        log::warn!(
                                            "Responses upstream was temporarily unavailable before output; retrying provider {} ({}/{})",
                                            provider.id,
                                            responses_retry_state.used,
                                            responses_retry_state.used
                                                + responses_retry_state.remaining
                                        );
                                        continue;
                                    }
                                    Ok(false) => {}
                                    Err(timeout_error) => {
                                        return Err(if rectifier_retried {
                                            BufferedRequestError::AfterResponse(timeout_error)
                                        } else {
                                            BufferedRequestError::BeforeResponse(timeout_error)
                                        });
                                    }
                                }
                                return Err(if rectifier_retried {
                                    BufferedRequestError::AfterResponse(error)
                                } else {
                                    BufferedRequestError::BeforeResponse(error)
                                });
                            }
                        }

                        if !rectifier_retried {
                            if let Some(rectified_body) = maybe_rectify_claude_buffered_request(
                                app_type,
                                &buffered_response,
                                &request_body,
                                rectifier_config,
                            ) {
                                rectifier_retried = true;
                                request_body = rectified_body;
                                continue 'request_loop;
                            }
                        }

                        return Ok(BufferedAttemptOutcome {
                            attempt_decision: classify_upstream_response(
                                buffered_response.status,
                                rectifier_retried,
                                app_type,
                                provider,
                            ),
                            response: buffered_response,
                        });
                    }
                    Ok(Err(error)) => {
                        if allow_transport_retry
                            && attempt < options.max_retries
                            && is_retryable_transport_error(&error)
                        {
                            attempt += 1;
                            continue;
                        }

                        let mapped_error = map_request_send_error(error, request_timeout);
                        return Err(if rectifier_retried {
                            BufferedRequestError::AfterResponse(mapped_error)
                        } else {
                            BufferedRequestError::BeforeResponse(mapped_error)
                        });
                    }
                    Err(_) => {
                        if allow_transport_retry && attempt < options.max_retries {
                            attempt += 1;
                            continue;
                        }

                        let timeout_error = request_timeout_error(
                            request_timeout
                                .expect("request timeout should exist when timeout future errors"),
                        );
                        return Err(if rectifier_retried {
                            BufferedRequestError::AfterResponse(timeout_error)
                        } else {
                            BufferedRequestError::BeforeResponse(timeout_error)
                        });
                    }
                }
            }
        }
    }
}

fn classify_attempt_error(
    error: &ProxyError,
    app_type: &AppType,
    provider: &Provider,
) -> AttemptDecision {
    if matches!(app_type, AppType::Codex)
        && provider.is_codex_official()
        && (matches!(error, ProxyError::AuthError(_))
            || matches!(
                error,
                ProxyError::UpstreamError {
                    status: 401 | 403,
                    ..
                }
            ))
    {
        return AttemptDecision::NeutralRelease;
    }

    match error {
        ProxyError::UpstreamError {
            status: 400 | 405 | 406 | 413 | 414 | 415 | 422 | 501,
            ..
        } => AttemptDecision::NeutralRelease,
        ProxyError::AlreadyRunning
        | ProxyError::NotRunning
        | ProxyError::BindFailed(_)
        | ProxyError::StopTimeout
        | ProxyError::StopFailed(_)
        | ProxyError::NoAvailableProvider
        | ProxyError::AllProvidersCircuitOpen
        | ProxyError::NoProvidersConfigured
        | ProxyError::DatabaseError(_)
        | ProxyError::InvalidRequest(_)
        | ProxyError::Internal(_) => AttemptDecision::FatalStop,
        _ => AttemptDecision::ProviderFailure,
    }
}

fn maybe_rectify_claude_buffered_request(
    app_type: &AppType,
    response: &BufferedResponse,
    request_body: &Value,
    rectifier_config: &RectifierConfig,
) -> Option<Value> {
    if *app_type != AppType::Claude {
        return None;
    }

    if !matches!(response.status.as_u16(), 400 | 422) {
        return None;
    }

    let error_message = extract_upstream_error_message(&response.body);

    if should_rectify_thinking_signature(error_message.as_deref(), rectifier_config) {
        let mut rectified_body = request_body.clone();
        let result = rectify_anthropic_request(&mut rectified_body);
        if result.applied {
            return Some(normalize_thinking_type(rectified_body));
        }
    }

    if should_rectify_thinking_budget(error_message.as_deref(), rectifier_config) {
        let mut rectified_body = request_body.clone();
        let result = rectify_thinking_budget(&mut rectified_body);
        if result.applied {
            return Some(normalize_thinking_type(rectified_body));
        }
    }

    None
}

fn should_buffer_streaming_error_response(app_type: &AppType, status: reqwest::StatusCode) -> bool {
    *app_type == AppType::Claude && !status.is_success()
}

fn uses_responses_protocol(app_type: &AppType, provider: &Provider, endpoint: &str) -> bool {
    if matches!(app_type, AppType::Claude) {
        return super::providers::get_claude_api_format(provider) == "openai_responses";
    }

    let path = endpoint.split_once('?').map_or(endpoint, |(path, _)| path);
    matches!(
        path,
        "/responses" | "/v1/responses" | "/responses/compact" | "/v1/responses/compact"
    ) && !super::providers::should_convert_codex_responses_to_chat(provider, endpoint)
        && !super::providers::should_convert_codex_responses_to_anthropic(provider, endpoint)
}

fn anthropic_request_uses_web_search(app_type: &AppType, body: &Value) -> bool {
    matches!(app_type, AppType::Claude)
        && body
            .get("tools")
            .and_then(Value::as_array)
            .is_some_and(|tools| {
                tools.iter().any(|tool| {
                    tool.get("type")
                        .and_then(Value::as_str)
                        .is_some_and(|tool_type| tool_type.starts_with("web_search_"))
                })
            })
}

fn uses_codex_anthropic_protocol(app_type: &AppType, provider: &Provider, endpoint: &str) -> bool {
    matches!(app_type, AppType::Codex)
        && super::providers::should_convert_codex_responses_to_anthropic(provider, endpoint)
}

fn response_is_json(response: &reqwest::Response) -> bool {
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

async fn prepare_success_streaming_response(
    response: reqwest::Response,
    started_at: Instant,
    request_timeout: Option<Duration>,
    validate_responses_semantics: bool,
    validate_codex_anthropic_semantics: bool,
    validate_responses_until_terminal: bool,
    enforce_responses_total_timeout: bool,
) -> Result<LiveResponse, ProxyError> {
    if validate_responses_semantics {
        return validate_responses_stream_start(
            response,
            started_at,
            request_timeout,
            validate_responses_until_terminal,
            enforce_responses_total_timeout,
        )
        .await;
    }

    if validate_codex_anthropic_semantics {
        return validate_codex_anthropic_stream_start(
            response,
            started_at,
            request_timeout.or(Some(RESPONSES_RETRY_FALLBACK_TIMEOUT)),
        )
        .await;
    }

    let Some(request_timeout) = request_timeout else {
        return Ok(LiveResponse::from_reqwest(response));
    };

    let status = response.status();
    let headers = response.headers().clone();
    let mut stream = response.bytes_stream().boxed();
    let remaining_timeout = request_timeout.saturating_sub(started_at.elapsed());
    if remaining_timeout.is_zero() {
        return Err(stream_first_byte_timeout_error(request_timeout));
    }

    let first = tokio::time::timeout(remaining_timeout, stream.next())
        .await
        .map_err(|_| stream_first_byte_timeout_error(request_timeout))?;
    let Some(first) = first else {
        return Err(ProxyError::ForwardFailed(
            "stream ended before the first response chunk".to_string(),
        ));
    };
    let first = first.map_err(|error| {
        ProxyError::ForwardFailed(format!("read first response chunk failed: {error}"))
    })?;

    let replay = futures::stream::once(async move { Ok(first) }).chain(stream);
    Ok(LiveResponse::from_stream(status, headers, replay))
}

async fn validate_codex_anthropic_stream_start(
    response: reqwest::Response,
    started_at: Instant,
    request_timeout: Option<Duration>,
) -> Result<LiveResponse, ProxyError> {
    const MAX_PRIME_BYTES: usize = 256 * 1024;

    let status = response.status();
    let headers = response.headers().clone();
    if headers
        .get(reqwest::header::CONTENT_ENCODING)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|encoding| !encoding.trim().eq_ignore_ascii_case("identity"))
    {
        return Err(malformed_anthropic_stream_error(
            "Anthropic SSE response used content encoding despite an identity request",
        ));
    }
    let mut stream = response.bytes_stream().boxed();
    let mut replay_chunks = Vec::new();
    let mut replay_bytes = 0usize;
    let mut parse_buffer = String::new();
    let mut utf8_remainder = Vec::new();
    let mut validator = AnthropicStreamStartValidator::default();

    loop {
        let next = match request_timeout {
            Some(timeout) => {
                let remaining = timeout.saturating_sub(started_at.elapsed());
                if remaining.is_zero() {
                    return Err(stream_first_byte_timeout_error(timeout));
                }
                tokio::time::timeout(remaining, stream.next())
                    .await
                    .map_err(|_| stream_first_byte_timeout_error(timeout))?
            }
            None => stream.next().await,
        };

        let Some(chunk) = next else {
            if !utf8_remainder.is_empty() {
                return Err(empty_codex_anthropic_stream_error(
                    "Anthropic stream ended with incomplete UTF-8 before producing output",
                ));
            }
            if validator.inspect_sse_block(&parse_buffer)? {
                return Ok(LiveResponse::from_stream(
                    status,
                    headers,
                    futures::stream::iter(replay_chunks.into_iter().map(Ok)),
                ));
            }
            if let Some(result) = inspect_anthropic_json_document(&parse_buffer) {
                result?;
                return Ok(LiveResponse::from_stream(
                    status,
                    headers,
                    futures::stream::iter(replay_chunks.into_iter().map(Ok)),
                ));
            }
            return Err(empty_codex_anthropic_stream_error(
                "Anthropic stream ended before producing text, reasoning, or a tool call",
            ));
        };
        let chunk = chunk.map_err(|error| {
            ProxyError::ForwardFailed(format!(
                "failed while validating Anthropic stream start: {error}"
            ))
        })?;
        replay_bytes = replay_bytes.saturating_add(chunk.len());
        if replay_bytes > MAX_PRIME_BYTES {
            return Err(empty_codex_anthropic_stream_error(&format!(
                "Anthropic stream exceeded {MAX_PRIME_BYTES} bytes before producing text, reasoning, or a complete tool call"
            )));
        }
        append_utf8_strict(&mut parse_buffer, &mut utf8_remainder, &chunk)?;
        replay_chunks.push(chunk);

        while let Some(block) = super::sse::take_sse_block(&mut parse_buffer) {
            if validator.inspect_sse_block(&block)? {
                let replay = futures::stream::iter(replay_chunks.into_iter().map(Ok)).chain(stream);
                return Ok(LiveResponse::from_stream(status, headers, replay));
            }
        }
    }
}

pub(crate) struct AnthropicStreamStartValidator {
    blocks: HashMap<u64, AnthropicValidationBlock>,
    seen_blocks: HashSet<u64>,
    substantive_output: bool,
    reject_empty_terminal: bool,
    terminal_seen: bool,
}

impl Default for AnthropicStreamStartValidator {
    fn default() -> Self {
        Self {
            blocks: HashMap::new(),
            seen_blocks: HashSet::new(),
            substantive_output: false,
            reject_empty_terminal: true,
            terminal_seen: false,
        }
    }
}

impl AnthropicStreamStartValidator {
    pub(crate) fn for_streaming_conversion() -> Self {
        Self::default()
    }
}

enum AnthropicValidationBlock {
    Text,
    Thinking,
    RedactedThinking,
    ToolUse {
        start_input: Value,
        partial_json: String,
    },
}

impl AnthropicStreamStartValidator {
    pub(crate) fn inspect_sse_block(&mut self, raw: &str) -> Result<bool, ProxyError> {
        let mut named_event = None;
        let mut data_lines = Vec::new();
        for line in raw.lines() {
            if let Some(event) = super::sse::strip_sse_field(line, "event") {
                named_event = Some(event.trim());
            } else if let Some(data) = super::sse::strip_sse_field(line, "data") {
                data_lines.push(data);
            }
        }

        let named_event = named_event.filter(|event| !event.is_empty());
        if data_lines.is_empty() {
            if named_event == Some("error") {
                return Err(malformed_anthropic_stream_error(
                    "Anthropic error event did not contain JSON data",
                ));
            }
            return Ok(false);
        }

        let value: Value = serde_json::from_str(&data_lines.join("\n")).map_err(|_| {
            malformed_anthropic_stream_error("Anthropic SSE event contained invalid JSON data")
        })?;
        let data_event = value.get("type").and_then(Value::as_str);
        if named_event.is_some() && data_event.is_some() && named_event != data_event {
            return Err(malformed_anthropic_stream_error(
                "Anthropic SSE event name did not match its JSON type",
            ));
        }
        let event = named_event.or(data_event).unwrap_or("");

        if self.terminal_seen {
            return Err(malformed_anthropic_stream_error(
                "Anthropic stream emitted an event after message_stop",
            ));
        }

        if event == "error" || value.get("error").is_some_and(|error| !error.is_null()) {
            let error = value.get("error").unwrap_or(&value);
            let error_type = error
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("upstream_error");
            let message = error
                .get("message")
                .and_then(Value::as_str)
                .or_else(|| error.as_str())
                .unwrap_or("Anthropic upstream emitted an error before output");
            return Err(responses_upstream_error(error_type, message));
        }

        let substantive = match event {
            "content_block_start" => self.start_block(&value),
            "content_block_delta" => self.update_block(&value),
            "content_block_stop" => self.stop_block(&value),
            "message_stop" if !self.blocks.is_empty() => Err(malformed_anthropic_stream_error(
                "Anthropic stream completed with an open content block",
            )),
            "message_stop" if !self.substantive_output && self.reject_empty_terminal => {
                Err(empty_codex_anthropic_stream_error(
                    "Anthropic stream completed without text, reasoning, or a complete tool call",
                ))
            }
            "message_stop" => {
                self.terminal_seen = true;
                Ok(false)
            }
            _ => Ok(false),
        }?;
        self.substantive_output |= substantive;
        Ok(self.substantive_output)
    }

    fn start_block(&mut self, value: &Value) -> Result<bool, ProxyError> {
        let index = anthropic_event_index(value)?;
        if !self.seen_blocks.insert(index) {
            return Err(malformed_anthropic_stream_error(
                "Anthropic stream started the same content block twice",
            ));
        }
        let block = value.get("content_block").ok_or_else(|| {
            malformed_anthropic_stream_error("Anthropic content block start omitted content_block")
        })?;
        let block_type = block.get("type").and_then(Value::as_str).ok_or_else(|| {
            malformed_anthropic_stream_error("Anthropic content block start omitted its type")
        })?;
        let (state, substantive) = match block_type {
            "text" => {
                let text = block.get("text").and_then(Value::as_str).ok_or_else(|| {
                    malformed_anthropic_stream_error("Anthropic text block omitted text")
                })?;
                (AnthropicValidationBlock::Text, !text.is_empty())
            }
            "thinking" => {
                let thinking = block.get("thinking").and_then(Value::as_str).unwrap_or("");
                (AnthropicValidationBlock::Thinking, !thinking.is_empty())
            }
            "redacted_thinking" => {
                let data = block.get("data").and_then(Value::as_str).ok_or_else(|| {
                    malformed_anthropic_stream_error(
                        "Anthropic redacted thinking block omitted data",
                    )
                })?;
                (AnthropicValidationBlock::RedactedThinking, !data.is_empty())
            }
            "tool_use" if valid_anthropic_tool_use_start(block) => (
                AnthropicValidationBlock::ToolUse {
                    start_input: block.get("input").cloned().unwrap_or_else(|| json!({})),
                    partial_json: String::new(),
                },
                false,
            ),
            "tool_use" => {
                return Err(malformed_anthropic_stream_error(
                    "Anthropic tool use omitted a valid id, name, or input object",
                ));
            }
            _ => return Ok(false),
        };
        self.blocks.insert(index, state);
        Ok(substantive)
    }

    fn update_block(&mut self, value: &Value) -> Result<bool, ProxyError> {
        let index = anthropic_event_index(value)?;
        let delta = value.get("delta").ok_or_else(|| {
            malformed_anthropic_stream_error("Anthropic content block delta omitted delta")
        })?;
        let delta_type = delta.get("type").and_then(Value::as_str).ok_or_else(|| {
            malformed_anthropic_stream_error("Anthropic content block delta omitted its type")
        })?;
        let block = self.blocks.get_mut(&index).ok_or_else(|| {
            malformed_anthropic_stream_error(
                "Anthropic content block delta arrived before its block start",
            )
        })?;
        match (block, delta_type) {
            (AnthropicValidationBlock::Text, "text_delta") => delta
                .get("text")
                .and_then(Value::as_str)
                .map(|text| !text.is_empty())
                .ok_or_else(|| {
                    malformed_anthropic_stream_error("Anthropic text delta omitted text")
                }),
            (AnthropicValidationBlock::Thinking, "thinking_delta") => delta
                .get("thinking")
                .and_then(Value::as_str)
                .map(|thinking| !thinking.is_empty())
                .ok_or_else(|| {
                    malformed_anthropic_stream_error("Anthropic thinking delta omitted thinking")
                }),
            (AnthropicValidationBlock::Thinking, "signature_delta") => delta
                .get("signature")
                .and_then(Value::as_str)
                .map(|_| false)
                .ok_or_else(|| {
                    malformed_anthropic_stream_error("Anthropic signature delta omitted signature")
                }),
            (AnthropicValidationBlock::ToolUse { partial_json, .. }, "input_json_delta") => {
                let fragment = delta
                    .get("partial_json")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        malformed_anthropic_stream_error(
                            "Anthropic tool input delta omitted partial_json",
                        )
                    })?;
                partial_json.push_str(fragment);
                Ok(false)
            }
            (AnthropicValidationBlock::RedactedThinking, _) => {
                Err(malformed_anthropic_stream_error(
                    "Anthropic redacted thinking block emitted an unexpected delta",
                ))
            }
            _ => Err(malformed_anthropic_stream_error(
                "Anthropic content block delta did not match its block type",
            )),
        }
    }

    fn stop_block(&mut self, value: &Value) -> Result<bool, ProxyError> {
        let index = anthropic_event_index(value)?;
        let block = self.blocks.remove(&index).ok_or_else(|| {
            malformed_anthropic_stream_error(
                "Anthropic content block stopped before it was started",
            )
        })?;
        let AnthropicValidationBlock::ToolUse {
            start_input,
            partial_json,
        } = block
        else {
            return Ok(false);
        };
        let input = if partial_json.is_empty() {
            start_input
        } else {
            serde_json::from_str(&partial_json).map_err(|_| {
                malformed_anthropic_stream_error(
                    "Anthropic tool input deltas did not form valid JSON",
                )
            })?
        };
        if !input.is_object() {
            return Err(malformed_anthropic_stream_error(
                "Anthropic tool input did not form a JSON object",
            ));
        }
        Ok(true)
    }
}

fn anthropic_event_index(value: &Value) -> Result<u64, ProxyError> {
    value.get("index").and_then(Value::as_u64).ok_or_else(|| {
        malformed_anthropic_stream_error("Anthropic content block event omitted a valid index")
    })
}

fn append_utf8_strict(
    buffer: &mut String,
    remainder: &mut Vec<u8>,
    new_bytes: &[u8],
) -> Result<(), ProxyError> {
    let mut combined = std::mem::take(remainder);
    combined.extend_from_slice(new_bytes);
    match std::str::from_utf8(&combined) {
        Ok(valid) => buffer.push_str(valid),
        Err(error) if error.error_len().is_none() => {
            let valid = std::str::from_utf8(&combined[..error.valid_up_to()])
                .expect("UTF-8 prefix before an incomplete code point is valid");
            buffer.push_str(valid);
            *remainder = combined[error.valid_up_to()..].to_vec();
        }
        Err(_) => {
            return Err(malformed_anthropic_stream_error(
                "Anthropic SSE stream contained invalid UTF-8",
            ));
        }
    }
    Ok(())
}

fn malformed_anthropic_stream_error(message: &str) -> ProxyError {
    responses_upstream_error("service_unavailable_error", message)
}

fn inspect_anthropic_json_document(input: &str) -> Option<Result<(), ProxyError>> {
    let value = serde_json::from_str::<Value>(input.trim()).ok()?;
    if value.get("type").and_then(Value::as_str) == Some("error")
        || value.get("error").is_some_and(|error| !error.is_null())
    {
        let error = value.get("error").unwrap_or(&value);
        let error_type = error
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("upstream_error");
        let message = error
            .get("message")
            .and_then(Value::as_str)
            .or_else(|| error.as_str())
            .unwrap_or("Anthropic upstream emitted an error before output");
        return Some(Err(responses_upstream_error(error_type, message)));
    }
    if value.get("type").and_then(Value::as_str) != Some("message")
        || value.get("role").and_then(Value::as_str) != Some("assistant")
    {
        return Some(Err(malformed_anthropic_stream_error(
            "Anthropic JSON response was not an assistant message",
        )));
    }
    let Some(content) = value.get("content").and_then(Value::as_array) else {
        return Some(Err(malformed_anthropic_stream_error(
            "Anthropic assistant message omitted its content array",
        )));
    };
    let mut substantive = false;
    for block in content {
        let block_type = block.get("type").and_then(Value::as_str).ok_or_else(|| {
            malformed_anthropic_stream_error("Anthropic content block omitted its type")
        });
        let Ok(block_type) = block_type else {
            return Some(Err(block_type.unwrap_err()));
        };
        match block_type {
            "tool_use" => {
                if !valid_anthropic_tool_use(block) {
                    return Some(Err(malformed_anthropic_stream_error(
                        "Anthropic tool use omitted a valid id, name, or input object",
                    )));
                }
                substantive = true;
            }
            "text" => {
                let Some(text) = block.get("text").and_then(Value::as_str) else {
                    return Some(Err(malformed_anthropic_stream_error(
                        "Anthropic text block omitted text",
                    )));
                };
                substantive |= !text.is_empty();
            }
            "thinking" => {
                let Some(thinking) = block.get("thinking").and_then(Value::as_str) else {
                    return Some(Err(malformed_anthropic_stream_error(
                        "Anthropic thinking block omitted thinking",
                    )));
                };
                substantive |= !thinking.is_empty();
            }
            "redacted_thinking" => {
                let Some(data) = block.get("data").and_then(Value::as_str) else {
                    return Some(Err(malformed_anthropic_stream_error(
                        "Anthropic redacted thinking block omitted data",
                    )));
                };
                substantive |= !data.is_empty();
            }
            _ => {
                if !block.is_object() {
                    return Some(Err(malformed_anthropic_stream_error(
                        "Anthropic content block was not an object",
                    )));
                }
            }
        }
    }
    Some(if substantive {
        Ok(())
    } else {
        Err(empty_codex_anthropic_stream_error(
            "Anthropic response completed without text, reasoning, or a tool call",
        ))
    })
}

fn empty_codex_anthropic_stream_error(message: &str) -> ProxyError {
    responses_upstream_error("service_unavailable_error", message)
}

fn valid_anthropic_tool_use(block: &Value) -> bool {
    block
        .get("id")
        .and_then(Value::as_str)
        .is_some_and(|id| !id.trim().is_empty())
        && block
            .get("name")
            .and_then(Value::as_str)
            .is_some_and(|name| !name.trim().is_empty())
        && block.get("input").is_some_and(Value::is_object)
}

fn valid_anthropic_tool_use_start(block: &Value) -> bool {
    block
        .get("id")
        .and_then(Value::as_str)
        .is_some_and(|id| !id.is_empty())
        && block
            .get("name")
            .and_then(Value::as_str)
            .is_some_and(|name| !name.is_empty())
        && block.get("input").is_none_or(Value::is_object)
}

async fn validate_responses_stream_start(
    response: reqwest::Response,
    started_at: Instant,
    request_timeout: Option<Duration>,
    validate_until_terminal: bool,
    enforce_total_timeout: bool,
) -> Result<LiveResponse, ProxyError> {
    const MAX_PRIME_BYTES: usize = 256 * 1024;
    const MAX_TERMINAL_VALIDATION_BYTES: usize = 4 * 1024 * 1024;
    const TERMINAL_VALIDATION_TIMEOUT: Duration = Duration::from_secs(600);

    let status = response.status();
    let headers = response.headers().clone();
    let mut stream = response.bytes_stream().boxed();
    let mut replay_chunks = Vec::new();
    let mut replay_bytes = 0usize;
    let mut parse_buffer = String::new();
    let mut utf8_remainder = Vec::new();
    let mut first_chunk_received_at = None;

    loop {
        let timeout_window = match first_chunk_received_at {
            Some(first_chunk_received_at) if validate_until_terminal && !enforce_total_timeout => {
                Some((TERMINAL_VALIDATION_TIMEOUT, first_chunk_received_at, false))
            }
            _ => request_timeout.map(|request_timeout| (request_timeout, started_at, true)),
        };
        let next = match timeout_window {
            Some((timeout, timeout_started_at, is_first_byte_timeout)) => {
                let remaining_timeout = timeout.saturating_sub(timeout_started_at.elapsed());
                if remaining_timeout.is_zero() {
                    return Err(if is_first_byte_timeout {
                        stream_first_byte_timeout_error(timeout)
                    } else {
                        ProxyError::Timeout(format!(
                            "Responses terminal validation timed out after {}s",
                            timeout.as_secs()
                        ))
                    });
                }
                tokio::time::timeout(remaining_timeout, stream.next())
                    .await
                    .map_err(|_| {
                        if is_first_byte_timeout {
                            stream_first_byte_timeout_error(timeout)
                        } else {
                            ProxyError::Timeout(format!(
                                "Responses terminal validation timed out after {}s",
                                timeout.as_secs()
                            ))
                        }
                    })?
            }
            None => stream.next().await,
        };

        let Some(chunk) = next else {
            if let Some(outcome) = inspect_responses_json_document(&parse_buffer) {
                outcome?;
                return Ok(LiveResponse::from_stream(
                    status,
                    headers,
                    futures::stream::iter(replay_chunks.into_iter().map(Ok)),
                ));
            }
            if !parse_buffer.trim().is_empty() {
                if let Some(outcome) = inspect_responses_start_event(parse_buffer.trim()) {
                    outcome?;
                    return Ok(LiveResponse::from_stream(
                        status,
                        headers,
                        futures::stream::iter(replay_chunks.into_iter().map(Ok)),
                    ));
                }
            }
            return Err(ProxyError::ForwardFailed(
                "Responses stream ended before producing output or a terminal event".to_string(),
            ));
        };
        let chunk = chunk.map_err(|error| {
            ProxyError::ForwardFailed(format!(
                "failed while validating Responses stream start: {error}"
            ))
        })?;
        first_chunk_received_at.get_or_insert_with(Instant::now);
        replay_bytes = replay_bytes.saturating_add(chunk.len());
        if validate_until_terminal && replay_bytes > MAX_TERMINAL_VALIDATION_BYTES {
            return Err(ProxyError::ForwardFailed(format!(
                "Responses stream exceeded {MAX_TERMINAL_VALIDATION_BYTES} bytes before a terminal event"
            )));
        }
        super::sse::append_utf8_safe(&mut parse_buffer, &mut utf8_remainder, &chunk);
        replay_chunks.push(chunk);

        if let Some(outcome) = inspect_responses_json_document(&parse_buffer) {
            outcome?;
            let replay = futures::stream::iter(replay_chunks.into_iter().map(Ok));
            if validate_until_terminal {
                return Ok(LiveResponse::from_stream(status, headers, replay));
            }
            let replay = replay.chain(stream);
            return Ok(LiveResponse::from_stream(status, headers, replay));
        }

        while let Some(block) = super::sse::take_sse_block(&mut parse_buffer) {
            if let Some(outcome) = inspect_responses_start_event(&block) {
                outcome?;
                if !validate_until_terminal || responses_block_is_terminal_success(&block) {
                    let replay = futures::stream::iter(replay_chunks.into_iter().map(Ok));
                    if validate_until_terminal {
                        return Ok(LiveResponse::from_stream(status, headers, replay));
                    }
                    let replay = replay.chain(stream);
                    return Ok(LiveResponse::from_stream(status, headers, replay));
                }
            }
        }

        if validate_until_terminal && replay_bytes >= MAX_TERMINAL_VALIDATION_BYTES {
            return Err(ProxyError::ForwardFailed(format!(
                "Responses stream exceeded {MAX_TERMINAL_VALIDATION_BYTES} bytes before a terminal event"
            )));
        }
        if !validate_until_terminal && replay_bytes >= MAX_PRIME_BYTES {
            log::warn!(
                "Responses semantic stream priming exceeded {MAX_PRIME_BYTES} bytes; committing stream"
            );
            let replay = futures::stream::iter(replay_chunks.into_iter().map(Ok)).chain(stream);
            return Ok(LiveResponse::from_stream(status, headers, replay));
        }
    }
}

fn validate_responses_success_body(body: &[u8]) -> Result<(), ProxyError> {
    if let Some((error_type, message)) = responses_error_envelope(body) {
        return Err(responses_upstream_error(&error_type, &message));
    }
    Ok(())
}

fn validate_buffered_responses_body(body: &[u8]) -> Result<(), ProxyError> {
    if serde_json::from_slice::<Value>(body).is_ok() {
        return validate_responses_success_body(body);
    }

    let Ok(text) = std::str::from_utf8(body) else {
        return Ok(());
    };
    let mut buffer = text.to_string();
    while let Some(block) = super::sse::take_sse_block(&mut buffer) {
        if let Some(Err(error)) = inspect_responses_start_event(&block) {
            return Err(error);
        }
    }
    if let Some(Err(error)) = inspect_responses_start_event(buffer.trim()) {
        return Err(error);
    }
    Ok(())
}

fn validate_codex_anthropic_success_body(body: &[u8]) -> Result<(), ProxyError> {
    let text = std::str::from_utf8(body).map_err(|_| {
        empty_codex_anthropic_stream_error(
            "Anthropic returned a non-UTF-8 2xx body before producing output",
        )
    })?;
    inspect_anthropic_json_document(text).unwrap_or_else(|| {
        Err(empty_codex_anthropic_stream_error(
            "Anthropic returned an empty or invalid JSON 2xx body before producing output",
        ))
    })
}

fn responses_error_envelope(body: &[u8]) -> Option<(String, String)> {
    let value: Value = serde_json::from_slice(body).ok()?;
    let status = value.get("status").and_then(Value::as_str);
    let has_error = value.get("error").is_some_and(|error| !error.is_null());
    let is_error_envelope = value.get("type").and_then(Value::as_str) == Some("error");
    let has_error_code = value.get("code").and_then(Value::as_str).is_some();
    if !matches!(status, Some("failed" | "cancelled"))
        && !has_error
        && !is_error_envelope
        && !has_error_code
    {
        return None;
    }

    let error = value.get("error").unwrap_or(&value);
    let error_type = responses_error_type(&value, error, status);
    let message = error
        .get("message")
        .and_then(Value::as_str)
        .or_else(|| value.get("message").and_then(Value::as_str))
        .or_else(|| {
            error
                .as_str()
                .filter(|candidate| !looks_like_responses_error_type(candidate))
        })
        .filter(|message| !message.trim().is_empty())
        .unwrap_or(match status {
            Some("cancelled") => "response generation was cancelled",
            _ => "response generation failed",
        });
    Some((error_type.to_string(), message.to_string()))
}

fn inspect_responses_json_document(buffer: &str) -> Option<Result<(), ProxyError>> {
    let trimmed = buffer.trim();
    if !matches!(trimmed.as_bytes().first(), Some(b'{') | Some(b'[')) {
        return None;
    }
    let _: Value = serde_json::from_str(trimmed).ok()?;
    Some(validate_responses_success_body(trimmed.as_bytes()))
}

fn inspect_responses_start_event(block: &str) -> Option<Result<(), ProxyError>> {
    let mut named_event = None;
    let mut data_lines = Vec::new();
    for line in block.lines() {
        if let Some(event) = super::sse::strip_sse_field(line, "event") {
            named_event = Some(event.trim().to_string());
        } else if let Some(data) = super::sse::strip_sse_field(line, "data") {
            data_lines.push(data);
        }
    }
    if data_lines.is_empty() {
        return None;
    }
    let value: Value = match serde_json::from_str(&data_lines.join("\n")) {
        Ok(value) => value,
        Err(_) => return None,
    };
    let event = named_event
        .as_deref()
        .filter(|event| !event.is_empty())
        .or_else(|| value.get("type").and_then(Value::as_str))
        .unwrap_or("");

    let response = value.get("response").unwrap_or(&value);
    if matches!(
        response.get("status").and_then(Value::as_str),
        Some("failed" | "cancelled")
    ) || response.get("error").is_some_and(|error| !error.is_null())
        || response.get("code").and_then(Value::as_str).is_some()
    {
        let error = response.get("error").unwrap_or(response);
        let message = error
            .get("message")
            .and_then(Value::as_str)
            .or_else(|| error.as_str())
            .unwrap_or("Responses upstream failed before output");
        let error_type = responses_error_type(
            response,
            error,
            response.get("status").and_then(Value::as_str),
        );
        return Some(Err(responses_upstream_error(&error_type, message)));
    }

    match event {
        "response.failed" | "error" => {
            let error = response.get("error").unwrap_or(response);
            let message = error
                .get("message")
                .and_then(Value::as_str)
                .or_else(|| error.as_str())
                .unwrap_or("Responses upstream emitted an error before output");
            let error_type = responses_error_type(response, error, None);
            Some(Err(responses_upstream_error(&error_type, message)))
        }
        "response.output_text.delta"
        | "response.refusal.delta"
        | "response.function_call_arguments.delta"
        | "response.reasoning.delta"
        | "response.completed"
        | "response.incomplete" => Some(Ok(())),
        "response.output_item.added"
            if value.pointer("/item/type").and_then(Value::as_str) == Some("function_call") =>
        {
            Some(Ok(()))
        }
        _ => None,
    }
}

fn responses_error_type(envelope: &Value, error: &Value, fallback: Option<&str>) -> String {
    let error_type = error
        .get("type")
        .and_then(Value::as_str)
        .filter(|error_type| !is_responses_error_wrapper(error_type))
        .or_else(|| error.get("code").and_then(Value::as_str))
        .or_else(|| {
            envelope
                .get("error")
                .and_then(Value::as_str)
                .filter(|candidate| looks_like_responses_error_type(candidate))
        })
        .or_else(|| envelope.get("code").and_then(Value::as_str))
        .or_else(|| {
            envelope
                .get("type")
                .and_then(Value::as_str)
                .filter(|error_type| !is_responses_error_wrapper(error_type))
        })
        .or_else(|| fallback.filter(|error_type| !is_responses_error_wrapper(error_type)))
        .unwrap_or("upstream_error");
    error_type.to_string()
}

fn looks_like_responses_error_type(candidate: &str) -> bool {
    candidate.ends_with("_error") || candidate == "overloaded"
}

fn is_responses_error_wrapper(candidate: &str) -> bool {
    matches!(
        candidate,
        "error" | "response.failed" | "failed" | "cancelled"
    )
}

fn responses_block_is_terminal_success(block: &str) -> bool {
    let mut named_event = None;
    let mut data_lines = Vec::new();
    for line in block.lines() {
        if let Some(event) = super::sse::strip_sse_field(line, "event") {
            named_event = Some(event.trim());
        } else if let Some(data) = super::sse::strip_sse_field(line, "data") {
            data_lines.push(data);
        }
    }
    let value = serde_json::from_str::<Value>(&data_lines.join("\n")).ok();
    let event = named_event
        .filter(|event| !event.is_empty())
        .or_else(|| {
            value
                .as_ref()
                .and_then(|value| value.get("type"))
                .and_then(Value::as_str)
        })
        .unwrap_or("");
    if matches!(event, "response.completed" | "response.incomplete") {
        return true;
    }
    value
        .as_ref()
        .and_then(|value| value.get("response").unwrap_or(value).get("status"))
        .and_then(Value::as_str)
        .is_some_and(|status| matches!(status, "completed" | "incomplete"))
}

fn responses_upstream_error(error_type: &str, message: &str) -> ProxyError {
    let status = match error_type {
        "service_unavailable_error" | "overloaded_error" | "server_error" => 503,
        "rate_limit_error" => 429,
        "authentication_error" => 401,
        "permission_error" => 403,
        "invalid_request_error" => 400,
        _ => 502,
    };
    let body = serde_json::json!({
        "type": "error",
        "error": {
            "type": error_type,
            "message": message,
        }
    });
    ProxyError::UpstreamError {
        status,
        body: Some(body.to_string()),
    }
}

fn is_retryable_responses_pre_output_error(error: &ProxyError) -> bool {
    matches!(error, ProxyError::UpstreamError { status: 503, .. })
}

fn responses_retry_backoff(retry_index: u32) -> Duration {
    let multiplier = 1u32.checked_shl(retry_index.min(2)).unwrap_or(4);
    RESPONSES_RETRY_BACKOFF_BASE
        .saturating_mul(multiplier)
        .min(RESPONSES_RETRY_BACKOFF_MAX)
}

async fn wait_for_responses_retry(
    delay: Duration,
    started_at: Instant,
    request_timeout: Option<Duration>,
) -> Result<(), ProxyError> {
    if let Some(request_timeout) = request_timeout {
        let remaining = request_timeout.saturating_sub(started_at.elapsed());
        if remaining <= delay {
            return Err(request_timeout_error(request_timeout));
        }
    }

    tokio::time::sleep(delay).await;
    Ok(())
}

async fn read_buffered_response(
    response: reqwest::Response,
    started_at: Instant,
    request_timeout: Option<Duration>,
    max_body_bytes: Option<usize>,
    use_stream_timeout_error: bool,
) -> Result<BufferedResponse, ProxyError> {
    let status = response.status();
    let mut headers = response.headers().clone();
    if max_body_bytes.is_some()
        && headers
            .get(reqwest::header::CONTENT_ENCODING)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|encoding| !encoding.trim().eq_ignore_ascii_case("identity"))
    {
        return Err(malformed_anthropic_stream_error(
            "Anthropic JSON response used content encoding despite an identity request",
        ));
    }
    if let (Some(limit), Some(content_length)) = (max_body_bytes, response.content_length()) {
        if content_length > limit as u64 {
            return Err(codex_anthropic_json_body_limit_error(limit));
        }
    }
    let mut stream = response.bytes_stream();
    let mut body = Vec::new();
    loop {
        let next = match request_timeout {
            Some(request_timeout) => {
                let remaining_timeout = request_timeout.saturating_sub(started_at.elapsed());
                if remaining_timeout.is_zero() {
                    return Err(buffered_read_timeout_error(
                        request_timeout,
                        use_stream_timeout_error,
                    ));
                }
                tokio::time::timeout(remaining_timeout, stream.next())
                    .await
                    .map_err(|_| {
                        buffered_read_timeout_error(request_timeout, use_stream_timeout_error)
                    })?
            }
            None => stream.next().await,
        };
        let Some(chunk) = next else {
            break;
        };
        let chunk = chunk.map_err(|error| map_request_send_error(error, request_timeout))?;
        if let Some(limit) = max_body_bytes {
            if body.len().saturating_add(chunk.len()) > limit {
                return Err(codex_anthropic_json_body_limit_error(limit));
            }
        }
        body.extend_from_slice(&chunk);
    }
    let body = decode_buffered_response_body(&mut headers, Bytes::from(body));
    if max_body_bytes.is_some_and(|limit| body.len() > limit) {
        return Err(codex_anthropic_json_body_limit_error(
            max_body_bytes.expect("body limit checked above"),
        ));
    }

    Ok(BufferedResponse {
        status,
        headers,
        body,
    })
}

fn codex_anthropic_json_body_limit_error(limit: usize) -> ProxyError {
    empty_codex_anthropic_stream_error(&format!(
        "Anthropic JSON response exceeded the {limit}-byte validation limit before client commit"
    ))
}

fn buffered_read_timeout_error(
    request_timeout: Duration,
    use_stream_timeout_error: bool,
) -> ProxyError {
    if use_stream_timeout_error {
        stream_first_byte_timeout_error(request_timeout)
    } else {
        request_timeout_error(request_timeout)
    }
}

fn extract_upstream_error_message(body: &[u8]) -> Option<String> {
    if let Ok(json_body) = serde_json::from_slice::<Value>(body) {
        return [
            json_body.pointer("/error/message"),
            json_body.pointer("/message"),
            json_body.pointer("/detail"),
            json_body.pointer("/error"),
        ]
        .into_iter()
        .flatten()
        .find_map(|value| value.as_str().map(ToString::to_string));
    }

    std::str::from_utf8(body)
        .ok()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
}

fn upstream_error_body_from_bytes(body: &[u8]) -> Option<String> {
    if body.is_empty() {
        None
    } else {
        Some(String::from_utf8_lossy(body).into_owned())
    }
}

fn buffered_response_to_upstream_error(response: BufferedResponse) -> ProxyError {
    ProxyError::UpstreamError {
        status: response.status.as_u16(),
        body: upstream_error_body_from_bytes(&response.body),
    }
}

fn streaming_response_to_upstream_error(response: StreamingResponse) -> ProxyError {
    match response {
        StreamingResponse::Buffered(response) => buffered_response_to_upstream_error(response),
        StreamingResponse::Live(response) => ProxyError::UpstreamError {
            status: response.status().as_u16(),
            body: None,
        },
    }
}

fn clone_request(
    base_request: &reqwest::RequestBuilder,
) -> Result<reqwest::RequestBuilder, ProxyError> {
    base_request.try_clone().ok_or_else(|| {
        ProxyError::ForwardFailed("clone proxy request failed before retry".to_string())
    })
}

fn uses_internal_transport_retry(app_type: &AppType) -> bool {
    !matches!(app_type, AppType::Claude)
}

fn is_retryable_transport_error(error: &reqwest::Error) -> bool {
    error.is_timeout() || error.is_connect()
}

fn map_request_send_error(error: reqwest::Error, request_timeout: Option<Duration>) -> ProxyError {
    if error.is_timeout() {
        return match request_timeout {
            Some(request_timeout) => request_timeout_error(request_timeout),
            None => ProxyError::Timeout(error.to_string()),
        };
    }

    if error.is_connect() {
        return ProxyError::ForwardFailed(format!("connection failed: {error}"));
    }

    ProxyError::ForwardFailed(error.to_string())
}

fn request_timeout_error(request_timeout: Duration) -> ProxyError {
    ProxyError::Timeout(format!(
        "request timed out after {}s",
        request_timeout.as_secs()
    ))
}

fn stream_first_byte_timeout_error(request_timeout: Duration) -> ProxyError {
    let display_seconds = request_timeout
        .as_secs()
        .max(u64::from(!request_timeout.is_zero()));
    ProxyError::Timeout(format!("stream timeout after {}s", display_seconds))
}

fn classify_upstream_response(
    status: reqwest::StatusCode,
    rectifier_retried: bool,
    app_type: &AppType,
    provider: &Provider,
) -> AttemptDecision {
    if matches!(app_type, AppType::Codex)
        && provider.is_codex_official()
        && matches!(status.as_u16(), 401 | 403)
    {
        return AttemptDecision::NeutralRelease;
    }

    match status.as_u16() {
        400 | 422 if rectifier_retried => AttemptDecision::NeutralRelease,
        400 | 405 | 406 | 413 | 414 | 415 | 422 | 501 => AttemptDecision::NeutralRelease,
        _ => AttemptDecision::ProviderFailure,
    }
}

#[cfg(test)]
mod tests;
