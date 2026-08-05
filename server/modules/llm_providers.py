from __future__ import annotations

"""Reliable, reversible chat-provider abstraction for adviSU.

The three supported providers expose OpenAI-compatible chat-completion endpoints, so one small
LangChain ``BaseChatModel`` can serve RetrievalQA without coupling application code to a vendor
SDK. Provider responses are reduced to final answer text, token counts and safe operational
metadata. Raw responses, provider error bodies and reasoning details are never retained.
"""

import json
import random
import re
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Iterator, Mapping, Sequence
from urllib.parse import urlparse

import httpx
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field, PrivateAttr


SUPPORTED_PROVIDERS = frozenset({"groq", "mistral", "openrouter"})
_REASONING_KEYS = frozenset({"reasoning", "reasoning_details", "thinking", "analysis"})
_REASONING_BLOCK_RE = re.compile(
    r"<(?P<tag>think|analysis|reasoning)\b[^>]*>.*?(?:</(?P=tag)\s*>|$)\s*",
    re.IGNORECASE | re.DOTALL,
)


class ProviderConfigurationError(ValueError):
    """The selected provider configuration is invalid or incomplete."""


class LLMProviderError(RuntimeError):
    """Base error with a deliberately sanitized, stable public representation."""

    def __init__(
        self,
        provider: str,
        code: str,
        *,
        retryable: bool,
        status_code: int | None = None,
    ) -> None:
        self.provider = provider
        self.code = code
        self.retryable = retryable
        self.status_code = status_code
        status = f", HTTP {status_code}" if status_code is not None else ""
        super().__init__(f"LLM provider {provider!r} failed ({code}{status}).")


class ProviderAuthenticationError(LLMProviderError):
    pass


class ProviderRateLimitError(LLMProviderError):
    pass


class ProviderTimeoutError(LLMProviderError):
    pass


class ProviderUnavailableError(LLMProviderError):
    pass


class ProviderRequestError(LLMProviderError):
    pass


class CircuitOpenError(LLMProviderError):
    pass


@dataclass(frozen=True, repr=False)
class ProviderSpec:
    name: str
    api_key: str
    model: str
    base_url: str

    def __repr__(self) -> str:
        return (
            f"ProviderSpec(name={self.name!r}, api_key='[REDACTED]', "
            f"model={self.model!r}, base_url='[REDACTED]')"
        )


@dataclass(frozen=True, repr=False)
class ProviderSettings:
    provider: str = "groq"
    fallback_provider: str | None = None
    groq_api_key: str | None = None
    groq_model: str = "llama-3.3-70b-versatile"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    mistral_api_key: str | None = None
    mistral_model: str = "mistral-small-latest"
    mistral_base_url: str = "https://api.mistral.ai/v1"
    openrouter_api_key: str | None = None
    openrouter_model: str = "deepseek/deepseek-v4-pro"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_http_referer: str = ""
    openrouter_app_title: str = "adviSU"
    openrouter_reasoning_enabled: bool = False
    openrouter_reasoning_effort: str = "medium"
    timeout_seconds: float = 60.0
    max_retries: int = 2
    retry_base_seconds: float = 0.25
    max_retry_after_seconds: float = 10.0
    circuit_failure_threshold: int = 3
    circuit_recovery_seconds: float = 30.0
    max_tokens: int = 2048
    temperature: float = 0.0

    def __repr__(self) -> str:
        return (
            "ProviderSettings("
            f"provider={self.provider!r}, fallback_provider={self.fallback_provider!r}, "
            f"groq_model={self.groq_model!r}, mistral_model={self.mistral_model!r}, "
            f"openrouter_model={self.openrouter_model!r}, credentials='[REDACTED]')"
        )

    def validated(self) -> "ProviderSettings":
        provider = self.provider.strip().lower()
        fallback = (self.fallback_provider or "").strip().lower() or None
        if provider not in SUPPORTED_PROVIDERS:
            raise ProviderConfigurationError(
                f"Unsupported LLM_PROVIDER={provider!r}; expected groq, mistral, or openrouter."
            )
        if fallback not in {None, "groq"}:
            raise ProviderConfigurationError("LLM_FALLBACK_PROVIDER must be empty or 'groq'.")
        if fallback == provider:
            fallback = None
        if self.timeout_seconds <= 0:
            raise ProviderConfigurationError("LLM_TIMEOUT_SECONDS must be positive.")
        if not 0 <= self.max_retries <= 5:
            raise ProviderConfigurationError("LLM_MAX_RETRIES must be between 0 and 5.")
        if self.retry_base_seconds < 0 or self.max_retry_after_seconds < 0:
            raise ProviderConfigurationError("Provider retry delays cannot be negative.")
        if self.circuit_failure_threshold < 1 or self.circuit_recovery_seconds < 0:
            raise ProviderConfigurationError("Circuit-breaker values are invalid.")
        if self.max_tokens < 1:
            raise ProviderConfigurationError("LLM_MAX_TOKENS must be positive.")
        if not 0 <= self.temperature <= 2:
            raise ProviderConfigurationError("LLM_TEMPERATURE must be between 0 and 2.")
        if self.openrouter_reasoning_effort not in {"low", "medium", "high"}:
            raise ProviderConfigurationError(
                "OPENROUTER_REASONING_EFFORT must be low, medium, or high."
            )

        # Validate every provider that can actually receive a request. Do not require unused keys.
        names = [provider, *([fallback] if fallback else [])]
        for name in names:
            spec = self._raw_spec(name)
            if not spec.api_key:
                raise ProviderConfigurationError(f"Missing API key for selected provider {name!r}.")
            if not spec.model.strip():
                raise ProviderConfigurationError(f"Missing model name for provider {name!r}.")
            parsed = urlparse(spec.base_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ProviderConfigurationError(f"Invalid base URL for provider {name!r}.")
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ProviderConfigurationError(
                    f"Provider {name!r} base URL cannot contain credentials, query, or fragment."
                )

        # Return a normalized copy without mutating the frozen settings object.
        return ProviderSettings(
            **{
                **asdict(self),
                "provider": provider,
                "fallback_provider": fallback,
                "groq_base_url": self.groq_base_url.rstrip("/"),
                "mistral_base_url": self.mistral_base_url.rstrip("/"),
                "openrouter_base_url": self.openrouter_base_url.rstrip("/"),
            }
        )

    def _raw_spec(self, name: str) -> ProviderSpec:
        if name == "groq":
            return ProviderSpec(name, self.groq_api_key or "", self.groq_model, self.groq_base_url)
        if name == "mistral":
            return ProviderSpec(
                name, self.mistral_api_key or "", self.mistral_model, self.mistral_base_url
            )
        if name == "openrouter":
            return ProviderSpec(
                name,
                self.openrouter_api_key or "",
                self.openrouter_model,
                self.openrouter_base_url,
            )
        raise ProviderConfigurationError(f"Unsupported provider {name!r}.")

    def spec(self, name: str) -> ProviderSpec:
        return self._raw_spec(name.strip().lower())

    @property
    def provider_order(self) -> tuple[str, ...]:
        # Flat and de-duplicated by validation: at most one fallback, therefore no loops.
        return (
            (self.provider, self.fallback_provider)
            if self.fallback_provider
            else (self.provider,)
        )


@dataclass(frozen=True)
class ProviderTelemetry:
    requested_provider: str
    effective_provider: str
    requested_model: str
    effective_model: str
    fallback_used: bool
    provider_attempts: Mapping[str, int]
    error_codes: tuple[str, ...]
    circuit_state: str
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    cost_usd: float | None = None
    provider_request_id: str | None = None

    def safe_dict(self) -> dict[str, Any]:
        return asdict(self)


class CircuitBreaker:
    """Small thread-safe closed/open/half-open circuit breaker."""

    def __init__(self, failure_threshold: int, recovery_seconds: float, clock: Callable[[], float]):
        self.failure_threshold = failure_threshold
        self.recovery_seconds = recovery_seconds
        self.clock = clock
        self._state = "closed"
        self._failures = 0
        self._opened_at: float | None = None
        self._half_open_probe = False
        self._lock = threading.Lock()

    def before_request(self, provider: str) -> None:
        with self._lock:
            if self._state == "open":
                elapsed = self.clock() - float(self._opened_at or 0.0)
                if elapsed >= self.recovery_seconds:
                    self._state = "half_open"
                    self._half_open_probe = True
                    return
                raise CircuitOpenError(provider, "circuit_open", retryable=True)
            if self._state == "half_open":
                if self._half_open_probe:
                    raise CircuitOpenError(provider, "circuit_half_open", retryable=True)
                self._half_open_probe = True

    def record_success(self) -> None:
        with self._lock:
            self._state = "closed"
            self._failures = 0
            self._opened_at = None
            self._half_open_probe = False

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            self._half_open_probe = False
            if self._state == "half_open" or self._failures >= self.failure_threshold:
                self._state = "open"
                self._opened_at = self.clock()

    @property
    def state(self) -> str:
        with self._lock:
            return self._state


@dataclass
class _ProviderResponse:
    content: str
    usage: dict[str, int | float | None]
    request_id: str | None
    attempts: int


def settings_from_config() -> ProviderSettings:
    """Build and validate settings without copying secrets into logs or errors."""
    from modules import config

    # Startup validation is necessary but not sufficient: evaluation scripts, workers and tests
    # can construct provider settings without running the ASGI lifespan hook. Enforce the same
    # production activation gate at the actual construction boundary as well.
    config.validate_provider_activation()

    return ProviderSettings(
        provider=config.LLM_PROVIDER,
        fallback_provider=config.LLM_FALLBACK_PROVIDER or None,
        groq_api_key=config.GROQ_API_KEY,
        groq_model=config.GROQ_MODEL_NAME,
        groq_base_url=config.GROQ_BASE_URL,
        mistral_api_key=config.MISTRAL_API_KEY,
        mistral_model=config.MISTRAL_MODEL_NAME,
        mistral_base_url=config.MISTRAL_BASE_URL,
        openrouter_api_key=config.OPENROUTER_API_KEY,
        openrouter_model=config.OPENROUTER_MODEL_NAME,
        openrouter_base_url=config.OPENROUTER_BASE_URL,
        openrouter_http_referer=config.OPENROUTER_HTTP_REFERER,
        openrouter_app_title=config.OPENROUTER_APP_TITLE,
        openrouter_reasoning_enabled=config.OPENROUTER_REASONING_ENABLED,
        openrouter_reasoning_effort=config.OPENROUTER_REASONING_EFFORT,
        timeout_seconds=config.LLM_TIMEOUT_SECONDS,
        max_retries=config.LLM_MAX_RETRIES,
        retry_base_seconds=config.LLM_RETRY_BASE_SECONDS,
        max_retry_after_seconds=config.LLM_MAX_RETRY_AFTER_SECONDS,
        circuit_failure_threshold=config.LLM_CIRCUIT_FAILURE_THRESHOLD,
        circuit_recovery_seconds=config.LLM_CIRCUIT_RECOVERY_SECONDS,
        max_tokens=config.LLM_MAX_TOKENS,
        temperature=config.LLM_TEMPERATURE,
    ).validated()


class ReliableProviderChatModel(BaseChatModel):
    """OpenAI-compatible LangChain chat model with bounded reliability controls."""

    settings: ProviderSettings = Field(exclude=True, repr=False)
    transport: httpx.BaseTransport | None = Field(default=None, exclude=True, repr=False)
    sleep_fn: Callable[[float], None] = Field(default=time.sleep, exclude=True, repr=False)
    clock: Callable[[], float] = Field(default=time.monotonic, exclude=True, repr=False)
    jitter_fn: Callable[[], float] = Field(default=random.random, exclude=True, repr=False)

    _breakers: dict[str, CircuitBreaker] = PrivateAttr(default_factory=dict)
    _last_telemetry: ProviderTelemetry | None = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        self.settings = self.settings.validated()
        self._breakers = {
            name: CircuitBreaker(
                self.settings.circuit_failure_threshold,
                self.settings.circuit_recovery_seconds,
                self.clock,
            )
            for name in self.settings.provider_order
        }

    @property
    def _llm_type(self) -> str:
        return "advisu-reliable-provider"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        # Deliberately excludes URLs, headers and credentials.
        return {
            "requested_provider": self.settings.provider,
            "requested_model": self.settings.spec(self.settings.provider).model,
            "fallback_provider": self.settings.fallback_provider,
        }

    def get_last_telemetry(self) -> dict[str, Any] | None:
        """Return a safe copy; never raw provider payloads or reasoning content."""
        return self._last_telemetry.safe_dict() if self._last_telemetry else None

    @property
    def resolved_configuration(self) -> Mapping[str, Any]:
        """Safe generation settings that the confirmatory benchmark binds before execution."""

        return {
            "reasoning_enabled": bool(self.settings.openrouter_reasoning_enabled),
            "temperature": float(self.settings.temperature),
            "maximum_output_tokens": int(self.settings.max_tokens),
            "request_timeout_seconds": self.settings.timeout_seconds,
            "maximum_retries": int(self.settings.max_retries),
            "fallback_provider": self.settings.fallback_provider,
        }

    def circuit_states(self) -> dict[str, str]:
        return {name: breaker.state for name, breaker in self._breakers.items()}

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del run_manager
        payload_messages = [_message_payload(message) for message in messages]
        started = self.clock()
        deadline = started + self.settings.timeout_seconds
        attempts: dict[str, int] = {}
        errors: list[str] = []
        requested_spec = self.settings.spec(self.settings.provider)

        for provider_name in self.settings.provider_order:
            spec = self.settings.spec(provider_name)
            try:
                result = self._complete(
                    spec, payload_messages, stop=stop, deadline=deadline, **kwargs
                )
                attempts[provider_name] = result.attempts
                telemetry = self._telemetry(
                    requested_spec=requested_spec,
                    effective_spec=spec,
                    attempts=attempts,
                    errors=errors,
                    started=started,
                    usage=result.usage,
                    request_id=result.request_id,
                )
                self._last_telemetry = telemetry
                usage_metadata = _langchain_usage(result.usage)
                message = AIMessage(
                    content=result.content,
                    response_metadata={"provider_telemetry": telemetry.safe_dict()},
                    usage_metadata=usage_metadata,
                )
                return ChatResult(generations=[ChatGeneration(message=message)])
            except LLMProviderError as exc:
                attempts[provider_name] = int(getattr(exc, "attempts", 0) or 0)
                errors.append(exc.code)
                # Authentication and invalid-request failures are configuration defects, not
                # availability events. Never hide them behind a fallback answer.
                if not exc.retryable or provider_name == self.settings.provider_order[-1]:
                    self._last_telemetry = self._telemetry(
                        requested_spec=requested_spec,
                        effective_spec=spec,
                        attempts=attempts,
                        errors=errors,
                        started=started,
                        usage={},
                        request_id=None,
                    )
                    raise

        raise ProviderUnavailableError(
            self.settings.provider, "provider_chain_exhausted", retryable=True
        )

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        payload_messages = [_message_payload(message) for message in messages]
        started = self.clock()
        deadline = started + self.settings.timeout_seconds
        attempts: dict[str, int] = {}
        errors: list[str] = []
        requested_spec = self.settings.spec(self.settings.provider)

        for provider_name in self.settings.provider_order:
            spec = self.settings.spec(provider_name)
            emitted = False
            try:
                chunks, state = self._stream_complete(
                    spec, payload_messages, stop=stop, deadline=deadline, **kwargs
                )
                for text in chunks:
                    emitted = True
                    if run_manager:
                        run_manager.on_llm_new_token(text)
                    yield ChatGenerationChunk(message=AIMessageChunk(content=text))
                attempts[provider_name] = state["attempts"]
                telemetry = self._telemetry(
                    requested_spec=requested_spec,
                    effective_spec=spec,
                    attempts=attempts,
                    errors=errors,
                    started=started,
                    usage=state["usage"],
                    request_id=state["request_id"],
                )
                self._last_telemetry = telemetry
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        response_metadata={"provider_telemetry": telemetry.safe_dict()},
                        usage_metadata=_langchain_usage(state["usage"]),
                    )
                )
                return
            except LLMProviderError as exc:
                attempts[provider_name] = int(getattr(exc, "attempts", 0) or 0)
                errors.append(exc.code)
                # Once output is visible, switching providers would splice two answers together.
                if not exc.retryable or emitted or provider_name == self.settings.provider_order[-1]:
                    raise

    def _headers(self, spec: ProviderSpec) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {spec.api_key}",
            "Content-Type": "application/json",
        }
        if spec.name == "openrouter":
            if self.settings.openrouter_http_referer:
                headers["HTTP-Referer"] = self.settings.openrouter_http_referer
            if self.settings.openrouter_app_title:
                headers["X-Title"] = self.settings.openrouter_app_title
        return headers

    def _payload(
        self,
        spec: ProviderSpec,
        messages: list[dict[str, Any]],
        *,
        stream: bool,
        stop: Sequence[str] | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        requested_temperature = kwargs.get("temperature", self.settings.temperature)
        requested_max_tokens = kwargs.get("max_tokens", self.settings.max_tokens)
        try:
            temperature = float(requested_temperature)
            max_tokens = int(requested_max_tokens)
        except (TypeError, ValueError) as exc:
            raise ProviderConfigurationError("Invalid per-call generation override.") from exc
        if temperature != float(self.settings.temperature):
            raise ProviderConfigurationError(
                "Per-call temperature override is forbidden; use the resolved provider settings."
            )
        if max_tokens != int(self.settings.max_tokens):
            raise ProviderConfigurationError(
                "Per-call max_tokens override is forbidden; use the resolved provider settings."
            )
        payload: dict[str, Any] = {
            "model": spec.model,
            "messages": messages,
            "stream": stream,
            "temperature": float(self.settings.temperature),
            "max_tokens": int(self.settings.max_tokens),
        }
        if stop:
            payload["stop"] = list(stop)
        if spec.name == "openrouter" and self.settings.openrouter_reasoning_enabled:
            payload["reasoning"] = {
                "enabled": True,
                "effort": self.settings.openrouter_reasoning_effort,
                # Provider may use reasoning internally, but must omit it from the response body.
                "exclude": True,
            }
        return payload

    def _complete(
        self,
        spec: ProviderSpec,
        messages: list[dict[str, Any]],
        *,
        stop: Sequence[str] | None,
        deadline: float,
        **kwargs: Any,
    ) -> _ProviderResponse:
        breaker = self._breakers[spec.name]
        breaker.before_request(spec.name)
        payload = self._payload(spec, messages, stream=False, stop=stop, **kwargs)
        try:
            response, attempts = self._post_with_retries(spec, payload, deadline=deadline)
            data = _safe_json(response, spec.name)
            content = _extract_content(data)
            content = _sanitize_content(content, self._known_secrets())
            if not content:
                raise ProviderUnavailableError(spec.name, "empty_response", retryable=True)
            usage = _extract_usage(data)
            request_id = _safe_request_id(response, data)
            breaker.record_success()
            return _ProviderResponse(content, usage, request_id, attempts)
        except LLMProviderError as exc:
            if exc.retryable:
                breaker.record_failure()
            raise

    def _stream_complete(
        self,
        spec: ProviderSpec,
        messages: list[dict[str, Any]],
        *,
        stop: Sequence[str] | None,
        deadline: float,
        **kwargs: Any,
    ) -> tuple[Iterator[str], dict[str, Any]]:
        breaker = self._breakers[spec.name]
        breaker.before_request(spec.name)
        state: dict[str, Any] = {"usage": {}, "request_id": None, "attempts": 0}
        payload = self._payload(spec, messages, stream=True, stop=stop, **kwargs)

        def iterator() -> Iterator[str]:
            emitted_content = False
            try:
                sanitizer = _StreamingSanitizer(self._known_secrets())
                with self._open_stream_with_retries(
                    spec, payload, state, deadline=deadline
                ) as response:
                    state["request_id"] = _safe_header(response.headers.get("x-request-id"))
                    for line in response.iter_lines():
                        # httpx's read timeout is inactivity-based and resets after every chunk.
                        # Re-check the shared wall-clock deadline so a trickle stream cannot keep
                        # the request alive indefinitely.
                        self._remaining_seconds(spec.name, deadline)
                        if not line or not line.startswith("data:"):
                            continue
                        raw = line[5:].strip()
                        if raw == "[DONE]":
                            break
                        try:
                            event = json.loads(raw)
                        except (TypeError, json.JSONDecodeError):
                            continue
                        # Never copy reasoning/reasoning_details into chunks or state.
                        text = _extract_delta_content(event)
                        if text:
                            safe_text = sanitizer.feed(text)
                            if safe_text:
                                emitted_content = True
                                yield safe_text
                        usage = _extract_usage(event)
                        if usage:
                            state["usage"] = usage
                        if not state["request_id"]:
                            state["request_id"] = _safe_string(event.get("id"))
                final_text = sanitizer.finish()
                if final_text:
                    emitted_content = True
                    yield final_text
                if not emitted_content:
                    raise ProviderUnavailableError(spec.name, "empty_response", retryable=True)
                breaker.record_success()
            except LLMProviderError as exc:
                if exc.retryable:
                    breaker.record_failure()
                raise
            except (httpx.TimeoutException, httpx.RequestError):
                breaker.record_failure()
                exc = ProviderUnavailableError(spec.name, "stream_interrupted", retryable=True)
                exc.attempts = int(state["attempts"])
                raise exc from None

        return iterator(), state

    def _post_with_retries(
        self, spec: ProviderSpec, payload: dict[str, Any], *, deadline: float
    ) -> tuple[httpx.Response, int]:
        attempts = 0
        while True:
            attempts += 1
            response: httpx.Response | None = None
            try:
                remaining = self._remaining_seconds(spec.name, deadline)
                with httpx.Client(
                    transport=self.transport,
                    timeout=remaining,
                ) as client:
                    response = client.post(
                        f"{spec.base_url}/chat/completions",
                        headers=self._headers(spec),
                        json=payload,
                    )
            except httpx.TimeoutException:
                exc: LLMProviderError = ProviderTimeoutError(
                    spec.name, "timeout", retryable=True
                )
            except httpx.RequestError:
                exc = ProviderUnavailableError(spec.name, "network_error", retryable=True)
            else:
                exc = _error_for_response(spec.name, response)
                if exc is None:
                    return response, attempts

            exc.attempts = attempts
            if not exc.retryable or attempts > self.settings.max_retries:
                raise exc
            delay = min(
                self._retry_delay(response, attempts),
                self._remaining_seconds(spec.name, deadline),
            )
            if delay > 0:
                self.sleep_fn(delay)

    def _open_stream_with_retries(
        self,
        spec: ProviderSpec,
        payload: dict[str, Any],
        state: dict[str, Any],
        *,
        deadline: float,
    ):
        # ``httpx.stream`` cannot be returned after leaving its context manager, so build a
        # context manager whose __enter__ performs the bounded initial-response retries.
        model = self

        class _StreamContext:
            client: httpx.Client | None = None
            context: Any = None

            def __enter__(self) -> httpx.Response:
                attempts = 0
                while True:
                    attempts += 1
                    state["attempts"] = attempts
                    remaining = model._remaining_seconds(spec.name, deadline)
                    self.client = httpx.Client(
                        transport=model.transport,
                        timeout=remaining,
                    )
                    try:
                        self.context = self.client.stream(
                            "POST",
                            f"{spec.base_url}/chat/completions",
                            headers=model._headers(spec),
                            json=payload,
                        )
                        response = self.context.__enter__()
                    except httpx.TimeoutException:
                        self.client.close()
                        exc: LLMProviderError = ProviderTimeoutError(
                            spec.name, "timeout", retryable=True
                        )
                        response = None
                    except httpx.RequestError:
                        self.client.close()
                        exc = ProviderUnavailableError(spec.name, "network_error", retryable=True)
                        response = None
                    else:
                        exc = _error_for_response(spec.name, response)
                        if exc is None:
                            return response
                        self.context.__exit__(None, None, None)
                        self.client.close()
                    exc.attempts = attempts
                    if not exc.retryable or attempts > model.settings.max_retries:
                        raise exc
                    delay = min(
                        model._retry_delay(response, attempts),
                        model._remaining_seconds(spec.name, deadline),
                    )
                    if delay > 0:
                        model.sleep_fn(delay)

            def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
                if self.context is not None:
                    self.context.__exit__(exc_type, exc, tb)
                if self.client is not None:
                    self.client.close()

        return _StreamContext()

    def _retry_delay(self, response: httpx.Response | None, attempts: int) -> float:
        retry_after = _retry_after_seconds(response)
        if retry_after is not None:
            return min(max(retry_after, 0.0), self.settings.max_retry_after_seconds)
        base = min(
            self.settings.retry_base_seconds * (2 ** max(attempts - 1, 0)),
            self.settings.max_retry_after_seconds,
        )
        # Bounded jitter prevents synchronized retry storms. Explicit provider Retry-After values
        # above remain exact (subject only to the configured maximum and request deadline).
        return base * (0.75 + 0.5 * min(max(float(self.jitter_fn()), 0.0), 1.0))

    def _remaining_seconds(self, provider: str, deadline: float) -> float:
        remaining = deadline - self.clock()
        if remaining <= 0:
            raise ProviderTimeoutError(provider, "request_deadline_exceeded", retryable=True)
        return max(remaining, 0.001)

    def verify_exact_model_available(self) -> None:
        """Fail closed when OpenRouter's authenticated catalogue lacks the configured model."""
        spec = self.settings.spec(self.settings.provider)
        if spec.name != "openrouter":
            return
        try:
            with httpx.Client(transport=self.transport, timeout=self.settings.timeout_seconds) as client:
                response = client.get(
                    f"{spec.base_url}/models",
                    headers=self._headers(spec),
                )
        except httpx.TimeoutException:
            raise ProviderTimeoutError(spec.name, "model_preflight_timeout", retryable=True) from None
        except httpx.RequestError:
            raise ProviderUnavailableError(
                spec.name, "model_preflight_network_error", retryable=True
            ) from None
        error = _error_for_response(spec.name, response)
        if error is not None:
            raise error
        data = _safe_json(response, spec.name)
        rows = data.get("data")
        available = {
            str(row.get("id") or "")
            for row in rows
            if isinstance(rows, list) and isinstance(row, Mapping)
        } if isinstance(rows, list) else set()
        if spec.model not in available:
            raise ProviderConfigurationError(
                f"Configured OpenRouter model {spec.model!r} is unavailable; no substitute is allowed."
            )

    def _known_secrets(self) -> tuple[str, ...]:
        return tuple(
            secret
            for secret in (
                self.settings.groq_api_key,
                self.settings.mistral_api_key,
                self.settings.openrouter_api_key,
            )
            if secret and len(secret) >= 8
        )

    def _telemetry(
        self,
        *,
        requested_spec: ProviderSpec,
        effective_spec: ProviderSpec,
        attempts: Mapping[str, int],
        errors: Sequence[str],
        started: float,
        usage: Mapping[str, int | float | None],
        request_id: str | None,
    ) -> ProviderTelemetry:
        return ProviderTelemetry(
            requested_provider=requested_spec.name,
            effective_provider=effective_spec.name,
            requested_model=requested_spec.model,
            effective_model=effective_spec.model,
            fallback_used=requested_spec.name != effective_spec.name,
            provider_attempts=dict(attempts),
            error_codes=tuple(errors),
            circuit_state=self._breakers[effective_spec.name].state,
            latency_ms=round(max(self.clock() - started, 0.0) * 1000.0, 2),
            prompt_tokens=_optional_int(usage.get("prompt_tokens")),
            completion_tokens=_optional_int(usage.get("completion_tokens")),
            total_tokens=_optional_int(usage.get("total_tokens")),
            reasoning_tokens=_optional_int(usage.get("reasoning_tokens")),
            cost_usd=_optional_float(usage.get("cost_usd")),
            provider_request_id=request_id,
        )


def build_chat_model(
    settings: ProviderSettings | None = None,
    *,
    transport: httpx.BaseTransport | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    jitter_fn: Callable[[], float] = random.random,
) -> ReliableProviderChatModel:
    return ReliableProviderChatModel(
        settings=(settings or settings_from_config()).validated(),
        transport=transport,
        sleep_fn=sleep_fn,
        clock=clock,
        jitter_fn=jitter_fn,
    )


def _message_payload(message: BaseMessage) -> dict[str, Any]:
    role = {
        "system": "system",
        "human": "user",
        "ai": "assistant",
        "tool": "tool",
        "function": "function",
    }.get(message.type, "user")
    payload: dict[str, Any] = {"role": role, "content": _safe_message_content(message.content)}
    if getattr(message, "name", None):
        payload["name"] = message.name
    if role == "tool" and getattr(message, "tool_call_id", None):
        payload["tool_call_id"] = message.tool_call_id
    # Do not forward additional_kwargs from AI messages: providers may put reasoning_details there.
    return payload


def _safe_message_content(value: Any) -> Any:
    """Drop reasoning blocks from any prior assistant message before conversation continuation."""
    if isinstance(value, Mapping):
        block_type = str(value.get("type") or "").lower()
        if block_type in {"reasoning", "reasoning_content", "thinking", "analysis"}:
            return None
        return {
            key: _safe_message_content(item)
            for key, item in value.items()
            if str(key).lower() not in _REASONING_KEYS
        }
    if isinstance(value, list):
        return [cleaned for item in value if (cleaned := _safe_message_content(item)) is not None]
    return value


def _error_for_response(provider: str, response: httpx.Response) -> LLMProviderError | None:
    status = response.status_code
    if 200 <= status < 300:
        return None
    if status in {401, 403}:
        return ProviderAuthenticationError(
            provider, "authentication_failed", retryable=False, status_code=status
        )
    if status == 429:
        return ProviderRateLimitError(provider, "rate_limited", retryable=True, status_code=status)
    if status in {408, 409, 425} or status >= 500:
        return ProviderUnavailableError(
            provider, "temporarily_unavailable", retryable=True, status_code=status
        )
    return ProviderRequestError(provider, "invalid_request", retryable=False, status_code=status)


def _safe_json(response: httpx.Response, provider: str) -> dict[str, Any]:
    try:
        data = response.json()
    except (TypeError, ValueError, json.JSONDecodeError):
        raise ProviderUnavailableError(provider, "invalid_json", retryable=True) from None
    if not isinstance(data, dict):
        raise ProviderUnavailableError(provider, "invalid_response", retryable=True)
    return data


def _extract_content(data: Mapping[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0] if isinstance(choices[0], Mapping) else {}
    message = first.get("message") if isinstance(first, Mapping) else {}
    content = message.get("content") if isinstance(message, Mapping) else ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text") or "")
            for part in content
            if isinstance(part, Mapping) and part.get("type") in {"text", "output_text"}
        )
    return ""


def _extract_delta_content(data: Mapping[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        return ""
    delta = choices[0].get("delta")
    if not isinstance(delta, Mapping):
        return ""
    content = delta.get("content")
    return content if isinstance(content, str) else ""


def _extract_usage(data: Mapping[str, Any]) -> dict[str, int | float | None]:
    usage = data.get("usage")
    if not isinstance(usage, Mapping):
        return {}
    completion_details = usage.get("completion_tokens_details")
    reasoning_tokens = (
        completion_details.get("reasoning_tokens")
        if isinstance(completion_details, Mapping)
        else None
    )
    return {
        "prompt_tokens": _optional_int(usage.get("prompt_tokens") or usage.get("input_tokens")),
        "completion_tokens": _optional_int(
            usage.get("completion_tokens") or usage.get("output_tokens")
        ),
        "total_tokens": _optional_int(usage.get("total_tokens")),
        "reasoning_tokens": _optional_int(reasoning_tokens),
        "cost_usd": _optional_float(usage.get("cost")),
    }


def _langchain_usage(usage: Mapping[str, int | float | None]) -> dict[str, int] | None:
    prompt = _optional_int(usage.get("prompt_tokens"))
    completion = _optional_int(usage.get("completion_tokens"))
    total = _optional_int(usage.get("total_tokens"))
    if prompt is None and completion is None and total is None:
        return None
    prompt = prompt or 0
    completion = completion or 0
    return {
        "input_tokens": prompt,
        "output_tokens": completion,
        "total_tokens": total if total is not None else prompt + completion,
    }


def _sanitize_content(content: str, secrets: Sequence[str], *, strip: bool = True) -> str:
    # Some reasoning models emit a legacy <think> block even when structured reasoning exclusion
    # was requested. Drop it defensively and never retain the provider's reasoning fields.
    # The end-of-string alternative deliberately drops an unclosed reasoning block. A malformed
    # provider response must lose trailing user-visible text rather than expose hidden reasoning.
    clean = _REASONING_BLOCK_RE.sub("", content or "")
    for secret in secrets:
        clean = clean.replace(secret, "[REDACTED]")
    return clean.strip() if strip else clean


class _StreamingSanitizer:
    """Stateful redaction for tags and secrets split across provider chunks.

    Holding a bounded suffix prevents a partial ``<think>`` tag or credential prefix from being
    emitted before the following chunk makes its meaning clear. Unclosed reasoning blocks are
    discarded at end-of-stream.
    """

    _OPEN_PREFIXES = ("<think", "<analysis", "<reasoning")
    _CLOSE_BY_OPEN = {
        "<think": "</think>",
        "<analysis": "</analysis>",
        "<reasoning": "</reasoning>",
    }

    def __init__(self, secrets: Sequence[str]) -> None:
        self._secrets = tuple(secret for secret in secrets if secret)
        self._tag_tail = ""
        self._visible_tail = ""
        self._in_reasoning = False
        self._close_marker: str | None = None
        self._secret_reserve = max((len(secret) for secret in self._secrets), default=1) - 1

    @staticmethod
    def _partial_suffix_length(value: str, marker: str) -> int:
        folded, target = value.casefold(), marker.casefold()
        maximum = min(len(value), len(marker) - 1)
        for size in range(maximum, 0, -1):
            if folded.endswith(target[:size]):
                return size
        return 0

    @classmethod
    def _partial_open_suffix_length(cls, value: str) -> int:
        return max(cls._partial_suffix_length(value, marker) for marker in cls._OPEN_PREFIXES)

    @staticmethod
    def _first_marker(value: str, markers: Sequence[str]) -> tuple[int, str] | None:
        folded = value.casefold()
        matches = [
            (folded.find(marker.casefold()), marker)
            for marker in markers
            if folded.find(marker.casefold()) >= 0
        ]
        return min(matches, key=lambda match: match[0]) if matches else None

    def _visible(self, text: str, *, final: bool = False) -> str:
        self._visible_tail += text
        for secret in self._secrets:
            self._visible_tail = self._visible_tail.replace(secret, "[REDACTED]")
        if final or self._secret_reserve <= 0:
            emitted, self._visible_tail = self._visible_tail, ""
            return emitted
        if len(self._visible_tail) <= self._secret_reserve:
            return ""
        emitted = self._visible_tail[:-self._secret_reserve]
        self._visible_tail = self._visible_tail[-self._secret_reserve:]
        return emitted

    def feed(self, text: str) -> str:
        data = self._tag_tail + str(text or "")
        self._tag_tail = ""
        visible_parts: list[str] = []
        while data:
            markers = (self._close_marker,) if self._in_reasoning else self._OPEN_PREFIXES
            match = self._first_marker(data, tuple(marker for marker in markers if marker))
            if match is not None:
                index, marker = match
                if not self._in_reasoning:
                    visible_parts.append(data[:index])
                    tag_end = data.find(">", index + len(marker))
                    if tag_end < 0:
                        # Hold an attribute-bearing opening tag until a later chunk completes it.
                        # If the stream ends first, finish() drops the held tag and its tail.
                        self._tag_tail = data[index:]
                        break
                    self._close_marker = self._CLOSE_BY_OPEN[marker]
                    data = data[tag_end + 1:]
                else:
                    self._close_marker = None
                    data = data[index + len(marker):]
                self._in_reasoning = not self._in_reasoning
                continue
            keep = (
                self._partial_suffix_length(data, str(self._close_marker))
                if self._in_reasoning
                else self._partial_open_suffix_length(data)
            )
            stable = data[:-keep] if keep else data
            if not self._in_reasoning:
                visible_parts.append(stable)
            self._tag_tail = data[-keep:] if keep else ""
            break
        return self._visible("".join(visible_parts))

    def finish(self) -> str:
        folded_tail = self._tag_tail.casefold()
        is_partial_or_open = any(
            marker.casefold().startswith(folded_tail)
            or folded_tail.startswith(marker.casefold())
            for marker in self._OPEN_PREFIXES
        )
        if not self._in_reasoning and not is_partial_or_open:
            self._visible_tail += self._tag_tail
        self._tag_tail = ""
        return self._visible("", final=True)


def _retry_after_seconds(response: httpx.Response | None) -> float | None:
    if response is None:
        return None
    value = response.headers.get("retry-after")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        try:
            target = parsedate_to_datetime(value)
            if target.tzinfo is None:
                target = target.replace(tzinfo=timezone.utc)
            return max((target - datetime.now(timezone.utc)).total_seconds(), 0.0)
        except (TypeError, ValueError, OverflowError):
            return None


def _safe_request_id(response: httpx.Response, data: Mapping[str, Any]) -> str | None:
    return _safe_header(response.headers.get("x-request-id")) or _safe_string(data.get("id"))


def _safe_header(value: str | None) -> str | None:
    return _safe_string(value)


def _safe_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    # Request ids are telemetry labels, not a channel for arbitrary provider data.
    clean = re.sub(r"[^A-Za-z0-9._:-]", "", value)[:128]
    return clean or None


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError, OverflowError):
        return None


def _optional_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError, OverflowError):
        return None


def strip_reasoning_fields(value: Any) -> Any:
    """Public defense-in-depth helper used by tests and future persistence boundaries."""
    if isinstance(value, Mapping):
        return {
            key: strip_reasoning_fields(item)
            for key, item in value.items()
            if str(key).lower() not in _REASONING_KEYS
        }
    if isinstance(value, list):
        return [strip_reasoning_fields(item) for item in value]
    return value
