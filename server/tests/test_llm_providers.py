from __future__ import annotations

import json

import httpx
import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from modules.llm_providers import (
    CircuitOpenError,
    ProviderAuthenticationError,
    ProviderConfigurationError,
    ProviderSettings,
    _sanitize_content,
    build_chat_model,
    strip_reasoning_fields,
)


OPENROUTER_SECRET = "or-test-secret-123456789"
GROQ_SECRET = "groq-test-secret-987654321"


def settings(**overrides) -> ProviderSettings:
    values = {
        "provider": "openrouter",
        "openrouter_api_key": OPENROUTER_SECRET,
        "openrouter_model": "deepseek/deepseek-v4-pro",
        "openrouter_base_url": "https://openrouter.test/v1",
        "groq_api_key": GROQ_SECRET,
        "groq_base_url": "https://groq.test/openai/v1",
        "max_retries": 0,
        "retry_base_seconds": 0,
        "circuit_failure_threshold": 3,
    }
    values.update(overrides)
    return ProviderSettings(**values)


def success_response(content: str = "Final answer", *, request_id: str = "req-1") -> httpx.Response:
    return httpx.Response(
        200,
        headers={"x-request-id": request_id},
        json={
            "id": request_id,
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": content,
                    "reasoning": "private chain of thought",
                    "reasoning_details": [{"text": f"hidden {OPENROUTER_SECRET}"}],
                }
            }],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
                "completion_tokens_details": {"reasoning_tokens": 4},
                "cost": 0.0012,
            },
        },
    )


def test_openrouter_uses_exact_model_and_returns_safe_telemetry():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["payload"] = json.loads(request.content)
        seen["authorization"] = request.headers["authorization"]
        return success_response(
            f"<think>private work</think>\nVerified answer; credential {OPENROUTER_SECRET}"
        )

    model = build_chat_model(settings(), transport=httpx.MockTransport(handler))
    answer = model.invoke([HumanMessage(content="Question")])

    assert answer.content == "Verified answer; credential [REDACTED]"
    assert seen["payload"]["model"] == "deepseek/deepseek-v4-pro"
    assert seen["payload"]["stream"] is False
    assert model.resolved_configuration["temperature"] == seen["payload"]["temperature"]
    assert model.resolved_configuration["maximum_output_tokens"] == seen["payload"]["max_tokens"]
    assert model.resolved_configuration["request_timeout_seconds"] == model.settings.timeout_seconds
    assert model.resolved_configuration["fallback_provider"] is None
    assert seen["authorization"] == f"Bearer {OPENROUTER_SECRET}"
    telemetry = model.get_last_telemetry()
    assert telemetry == answer.response_metadata["provider_telemetry"]
    assert telemetry["requested_provider"] == telemetry["effective_provider"] == "openrouter"
    assert telemetry["prompt_tokens"] == 11
    assert telemetry["reasoning_tokens"] == 4
    assert telemetry["cost_usd"] == pytest.approx(0.0012)
    safe_output = json.dumps({"answer": answer.content, "telemetry": telemetry, "model": repr(model)})
    assert OPENROUTER_SECRET not in safe_output
    assert "private chain of thought" not in safe_output
    assert "reasoning_details" not in safe_output
    assert OPENROUTER_SECRET not in json.dumps(model.model_dump(), default=str)


def test_per_call_generation_overrides_cannot_diverge_from_resolved_configuration():
    def unexpected_network(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"network must not be reached: {request.url.host}")

    model = build_chat_model(settings(), transport=httpx.MockTransport(unexpected_network))
    with pytest.raises(ProviderConfigurationError, match="temperature override is forbidden"):
        model.invoke("Question", temperature=0.7)
    with pytest.raises(ProviderConfigurationError, match="max_tokens override is forbidden"):
        model._payload(
            model.settings.spec("openrouter"),
            [{"role": "user", "content": "Question"}],
            stream=False,
            stop=None,
            max_tokens=model.settings.max_tokens + 1,
        )


def test_reasoning_request_is_excluded_and_response_reasoning_is_not_persisted():
    payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return success_response()

    model = build_chat_model(
        settings(openrouter_reasoning_enabled=True, openrouter_reasoning_effort="high"),
        transport=httpx.MockTransport(handler),
    )
    answer = model.invoke("Solve carefully")

    assert payloads[0]["reasoning"] == {"enabled": True, "effort": "high", "exclude": True}
    assert answer.content == "Final answer"
    assert "reasoning" not in answer.additional_kwargs
    assert strip_reasoning_fields({
        "content": "ok",
        "reasoning_details": [{"text": "secret"}],
        "nested": {"thinking": "secret", "safe": 1},
    }) == {"content": "ok", "nested": {"safe": 1}}


def test_prior_assistant_reasoning_blocks_are_not_forwarded():
    payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return success_response()

    model = build_chat_model(settings(), transport=httpx.MockTransport(handler))
    model.invoke([
        HumanMessage(content="first"),
        AIMessage(content=[
            {"type": "reasoning", "reasoning": "private"},
            {"type": "text", "text": "public answer"},
        ], additional_kwargs={"reasoning_details": [{"text": "private"}]}),
        HumanMessage(content="continue"),
    ])
    forwarded = json.dumps(payloads[0]["messages"])
    assert "private" not in forwarded
    assert "reasoning_details" not in forwarded
    assert "public answer" in forwarded


def test_retry_after_is_bounded_and_retry_count_is_reported():
    calls = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"retry-after": "99"}, json={"error": "do not log"})
        return success_response()

    model = build_chat_model(
        settings(max_retries=1, max_retry_after_seconds=2),
        transport=httpx.MockTransport(handler),
        sleep_fn=sleeps.append,
    )
    assert model.invoke("retry").content == "Final answer"
    assert calls == 2
    assert sleeps == [2]
    assert model.get_last_telemetry()["provider_attempts"] == {"openrouter": 2}


def test_explicit_groq_fallback_is_single_hop_and_telemetry_names_effective_provider():
    hosts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        if request.url.host == "openrouter.test":
            return httpx.Response(503, json={"error": f"must stay private {OPENROUTER_SECRET}"})
        return success_response("Groq fallback answer", request_id="groq-request")

    model = build_chat_model(
        settings(fallback_provider="groq"),
        transport=httpx.MockTransport(handler),
    )
    assert model.invoke("fallback").content == "Groq fallback answer"
    assert hosts == ["openrouter.test", "groq.test"]
    telemetry = model.get_last_telemetry()
    assert telemetry["requested_provider"] == "openrouter"
    assert telemetry["effective_provider"] == "groq"
    assert telemetry["fallback_used"] is True
    assert telemetry["provider_attempts"] == {"openrouter": 1, "groq": 1}
    assert telemetry["error_codes"] == ("temporarily_unavailable",)
    assert OPENROUTER_SECRET not in json.dumps(telemetry)


def test_circuit_opens_after_threshold_and_blocks_network_until_recovery():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, json={"error": "down"})

    model = build_chat_model(
        settings(circuit_failure_threshold=2, circuit_recovery_seconds=60),
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(Exception):
        model.invoke("one")
    with pytest.raises(Exception):
        model.invoke("two")
    assert model.circuit_states() == {"openrouter": "open"}
    with pytest.raises(CircuitOpenError):
        model.invoke("three")
    assert calls == 2


def test_provider_errors_are_typed_and_never_include_body_or_secret():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"error": f"credential={OPENROUTER_SECRET}", "reasoning_details": "private"},
        )

    model = build_chat_model(settings(), transport=httpx.MockTransport(handler))
    with pytest.raises(ProviderAuthenticationError) as caught:
        model.invoke("authenticate")
    rendered = f"{caught.value!s} {caught.value!r} {model!r}"
    assert OPENROUTER_SECRET not in rendered
    assert "reasoning_details" not in rendered
    assert "credential=" not in rendered


def test_authentication_failure_is_never_hidden_by_fallback():
    hosts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        return httpx.Response(401, json={"error": "bad credential"})

    model = build_chat_model(
        settings(fallback_provider="groq"),
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(ProviderAuthenticationError):
        model.invoke("authenticate")
    assert hosts == ["openrouter.test"]


def test_empty_provider_response_counts_as_failure():
    def handler(request: httpx.Request) -> httpx.Response:
        return success_response("")

    model = build_chat_model(settings(), transport=httpx.MockTransport(handler))
    with pytest.raises(Exception, match="empty_response"):
        model.invoke("empty")


def test_non_stream_reasoning_tags_fail_closed_when_closed_or_unclosed():
    assert _sanitize_content("Safe <think>private</think> answer", ()) == "Safe answer"
    assert _sanitize_content("Safe <analysis>private</analysis> answer", ()) == "Safe answer"
    assert _sanitize_content("Safe <reasoning>private without close", ()) == "Safe"


def test_streaming_ignores_reasoning_events_and_reports_usage():
    stream_body = "\n".join([
        'data: {"choices":[{"delta":{"reasoning_details":[{"text":"private"}]}}]}',
        'data: {"choices":[{"delta":{"content":"Hello "}}]}',
        'data: {"choices":[{"delta":{"content":"world"}}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}',
        "data: [DONE]",
        "",
    ])

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(200, text=stream_body, headers={"content-type": "text/event-stream"})

    model = build_chat_model(settings(), transport=httpx.MockTransport(handler))
    chunks = list(model.stream("hello"))
    assert "".join(str(chunk.content) for chunk in chunks) == "Hello world"
    assert all("private" not in str(chunk) for chunk in chunks)
    assert model.get_last_telemetry()["total_tokens"] == 5


def test_streaming_redacts_reasoning_tags_and_credentials_split_across_chunks():
    parts = ["Safe ", "<thi", "nk>private", "</thi", "nk>answer ", OPENROUTER_SECRET[:9], OPENROUTER_SECRET[9:]]
    stream_body = "\n".join(
        [f'data: {{"choices":[{{"delta":{{"content":{json.dumps(part)}}}}}]}}' for part in parts]
        + ["data: [DONE]", ""]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=stream_body, headers={"content-type": "text/event-stream"})

    model = build_chat_model(settings(), transport=httpx.MockTransport(handler))
    rendered = "".join(str(chunk.content) for chunk in model.stream("hello"))
    assert rendered == "Safe answer [REDACTED]"
    assert "private" not in rendered and OPENROUTER_SECRET not in rendered


def test_streaming_redacts_analysis_and_unclosed_reasoning_blocks():
    parts = ["Safe ", "<ana", "lysis>private", "</analysis>answer ", "<reasoning>", "hidden"]
    stream_body = "\n".join(
        [f'data: {{"choices":[{{"delta":{{"content":{json.dumps(part)}}}}}]}}' for part in parts]
        + ["data: [DONE]", ""]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=stream_body, headers={"content-type": "text/event-stream"})

    model = build_chat_model(settings(), transport=httpx.MockTransport(handler))
    rendered = "".join(str(chunk.content) for chunk in model.stream("hello"))
    assert rendered == "Safe answer "
    assert "private" not in rendered and "hidden" not in rendered


def test_streaming_redacts_attribute_bearing_reasoning_tags_across_chunks():
    parts = ["Safe ", "<think data-", "scope='private'>hidden", "</think>", "answer"]
    stream_body = "\n".join(
        [f'data: {{"choices":[{{"delta":{{"content":{json.dumps(part)}}}}}]}}' for part in parts]
        + ["data: [DONE]", ""]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=stream_body, headers={"content-type": "text/event-stream"})

    model = build_chat_model(settings(), transport=httpx.MockTransport(handler))
    rendered = "".join(str(chunk.content) for chunk in model.stream("hello"))
    assert rendered == "Safe answer"
    assert "hidden" not in rendered


def test_streaming_trickle_cannot_exceed_total_wall_clock_deadline():
    instants = iter((0.0, 0.0, 61.0, 61.0))

    def clock() -> float:
        return next(instants, 61.0)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text='data: {"choices":[{"delta":{"content":"late"}}]}\n\ndata: [DONE]\n',
            headers={"content-type": "text/event-stream"},
        )

    model = build_chat_model(
        settings(timeout_seconds=60),
        transport=httpx.MockTransport(handler),
        clock=clock,
    )
    with pytest.raises(Exception, match="request_deadline_exceeded"):
        list(model.stream("hello"))


def test_exact_openrouter_model_preflight_fails_closed_without_substitution():
    def available(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "deepseek/deepseek-v4-pro"}]})

    model = build_chat_model(settings(), transport=httpx.MockTransport(available))
    model.verify_exact_model_available()

    def missing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "some/other-model"}]})

    missing_model = build_chat_model(settings(), transport=httpx.MockTransport(missing))
    with pytest.raises(ProviderConfigurationError, match="no substitute"):
        missing_model.verify_exact_model_available()


def test_fallback_cannot_extend_original_request_deadline():
    instant = [0.0]
    hosts: list[str] = []

    def clock() -> float:
        return instant[0]

    def handler(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        instant[0] = 61.0
        return httpx.Response(503, json={"error": "temporary"})

    model = build_chat_model(
        settings(fallback_provider="groq", timeout_seconds=60),
        transport=httpx.MockTransport(handler),
        clock=clock,
    )
    with pytest.raises(Exception, match="request_deadline_exceeded"):
        model.invoke("deadline")
    assert hosts == ["openrouter.test"]


def test_configuration_validation_rejects_missing_key_invalid_provider_and_fallback_loop():
    with pytest.raises(ProviderConfigurationError):
        ProviderSettings(provider="openrouter").validated()
    with pytest.raises(ProviderConfigurationError):
        ProviderSettings(provider="unknown", groq_api_key=GROQ_SECRET).validated()
    with pytest.raises(ProviderConfigurationError):
        settings(fallback_provider="mistral").validated()

    # Selecting Groq while also naming it as fallback normalizes to one provider, never recursion.
    normalized = settings(provider="groq", fallback_provider="groq").validated()
    assert normalized.provider_order == ("groq",)


def test_model_remains_compatible_with_current_retrievalqa_chain(monkeypatch):
    class StaticRetriever(BaseRetriever):
        documents: list[Document] = Field(default_factory=list)

        def _get_relevant_documents(self, query: str) -> list[Document]:
            return self.documents

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert "Official fact" in payload["messages"][0]["content"]
        return success_response("Grounded final answer")

    model = build_chat_model(settings(), transport=httpx.MockTransport(handler))
    from modules import llm

    monkeypatch.setattr(llm, "_build_llm", lambda: model)
    chain = llm.get_llm_chain(
        StaticRetriever(documents=[Document(page_content="Official fact")]),
        intent="ders_ayrintisi",
        language="en",
    )
    result = chain.invoke({"query": "What is the fact?"})
    assert result["result"] == "Grounded final answer"
