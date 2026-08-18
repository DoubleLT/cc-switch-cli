use axum::http::HeaderMap;
use bytes::Bytes;
use futures::{stream::BoxStream, StreamExt};
use serde_json::Value;
use std::{
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

struct ResponsesRetryState {
    remaining: u32,
    used: u32,
    deadline_started_at: Option<Instant>,
}

impl ResponsesRetryState {
    fn new(app_type: &AppType, configured_retries: u32) -> Self {
        let remaining = if matches!(app_type, AppType::Claude) {
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

        let retry_started_at = if configured_timeout.is_some() {
            provider_started_at
        } else {
            Instant::now()
        };
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
        let mut responses_retry_state = ResponsesRetryState::new(app_type, options.max_retries);

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
                    &mut responses_retry_state,
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
        let mut responses_retry_state = ResponsesRetryState::new(app_type, options.max_retries);

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
                    &mut responses_retry_state,
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
                let attempt_started_at = if allow_transport_retry {
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
                                let buffered_response = read_streaming_error_response(
                                    response,
                                    attempt_started_at,
                                    request_timeout,
                                )
                                .await
                                .map_err(StreamingRequestError::AfterResponse)?;
                                validate_codex_anthropic_success_body(&buffered_response.body)
                                    .map_err(|error| {
                                        if rectifier_retried {
                                            StreamingRequestError::AfterResponse(error)
                                        } else {
                                            StreamingRequestError::BeforeResponse(error)
                                        }
                                    })?;
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
                            let buffered_response = read_streaming_error_response(
                                response,
                                attempt_started_at,
                                request_timeout,
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
                let attempt_started_at = if allow_transport_retry {
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
                        let status = response.status();
                        let mut response_headers = response.headers().clone();
                        let response_body = match request_timeout {
                            Some(request_timeout) => {
                                let remaining_timeout =
                                    request_timeout.saturating_sub(attempt_started_at.elapsed());
                                if remaining_timeout.is_zero() {
                                    return Err(BufferedRequestError::AfterResponse(
                                        request_timeout_error(request_timeout),
                                    ));
                                }
                                tokio::time::timeout(remaining_timeout, response.bytes())
                                    .await
                                    .map_err(|_| {
                                        BufferedRequestError::AfterResponse(request_timeout_error(
                                            request_timeout,
                                        ))
                                    })?
                                    .map_err(|error| {
                                        BufferedRequestError::AfterResponse(map_request_send_error(
                                            error,
                                            Some(request_timeout),
                                        ))
                                    })?
                            }
                            None => response.bytes().await.map_err(|error| {
                                BufferedRequestError::AfterResponse(map_request_send_error(
                                    error, None,
                                ))
                            })?,
                        };
                        let response_body =
                            decode_buffered_response_body(&mut response_headers, response_body);

                        let buffered_response = BufferedResponse {
                            status,
                            headers: response_headers,
                            body: response_body,
                        };

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
                            validate_codex_anthropic_success_body(&buffered_response.body)
                                .map_err(|error| {
                                    if rectifier_retried {
                                        BufferedRequestError::AfterResponse(error)
                                    } else {
                                        BufferedRequestError::BeforeResponse(error)
                                    }
                                })?;
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
    if let Some(message) = codex_anthropic_error_envelope_message(body) {
        return Err(ProxyError::TransformError(format!(
            "Anthropic upstream returned a 2xx error envelope: {message}"
        )));
    }
    Ok(())
}

fn codex_anthropic_error_envelope_message(body: &[u8]) -> Option<String> {
    let value: Value = serde_json::from_slice(body).ok()?;
    if value.get("type").and_then(Value::as_str) != Some("error") && value.get("error").is_none() {
        return None;
    }
    let error = value.get("error").unwrap_or(&value);
    let error_type = error.get("type").and_then(Value::as_str).unwrap_or("error");
    let message = error
        .get("message")
        .and_then(Value::as_str)
        .or_else(|| error.as_str())
        .map(str::to_string)
        .unwrap_or_else(|| error.to_string());
    Some(format!("{error_type}: {message}"))
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

async fn read_streaming_error_response(
    response: reqwest::Response,
    started_at: Instant,
    request_timeout: Option<Duration>,
) -> Result<BufferedResponse, ProxyError> {
    let status = response.status();
    let mut headers = response.headers().clone();
    let body = match request_timeout {
        Some(request_timeout) => {
            let remaining_timeout = request_timeout.saturating_sub(started_at.elapsed());
            if remaining_timeout.is_zero() {
                return Err(stream_first_byte_timeout_error(request_timeout));
            }

            tokio::time::timeout(remaining_timeout, response.bytes())
                .await
                .map_err(|_| stream_first_byte_timeout_error(request_timeout))?
                .map_err(|error| map_request_send_error(error, Some(request_timeout)))?
        }
        None => response
            .bytes()
            .await
            .map_err(|error| map_request_send_error(error, None))?,
    };
    let body = decode_buffered_response_body(&mut headers, body);

    Ok(BufferedResponse {
        status,
        headers,
        body,
    })
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
