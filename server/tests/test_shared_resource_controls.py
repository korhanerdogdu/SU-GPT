from __future__ import annotations

import statistics
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import modules.resource_controls as controls
from modules.rate_limit_backends import (
    AdmissionPolicy,
    AdmissionRequest,
    BackendTimeoutError,
    BackendUnavailableError,
    RateLimitBackend,
    RedisRateLimitBackend,
    SharedInMemoryRateLimitBackend,
    UsageReservation,
    build_rate_limit_backend,
)
from modules.resource_controls import (
    PrivacyKeyring,
    ResourceController,
    ResourceLimitError,
    privacy_identifier,
    usd_to_microusd,
)


KEY_V1 = b"v1-test-only-hmac-key-material-32b"
KEY_V2 = b"v2-test-only-hmac-key-material-32b"


def keyring(version: str = "v1", *, previous: dict[str, bytes] | None = None) -> PrivacyKeyring:
    secret = KEY_V1 if version == "v1" else KEY_V2
    return PrivacyKeyring.from_mapping(
        primary_version=version,
        primary_secret=secret,
        previous=previous,
    )


def controllers(
    *,
    backend: SharedInMemoryRateLimitBackend | None = None,
    lease_ttl_seconds: float | None = None,
    usage_ttl_seconds: float | None = None,
) -> tuple[ResourceController, ResourceController, SharedInMemoryRateLimitBackend]:
    shared = backend or SharedInMemoryRateLimitBackend()
    options = {
        "backend": shared,
        "keyring": keyring(),
        "lease_ttl_seconds": lease_ttl_seconds,
        "usage_ttl_seconds": usage_ttl_seconds,
    }
    return ResourceController(**options), ResourceController(**options), shared


def test_backend_protocol_and_existing_begin_finish_contract_are_preserved():
    first, _second, backend = controllers()
    assert isinstance(backend, RateLimitBackend)
    admission = first.begin(username="student", client_address="192.0.2.10", now=1_000)
    state = first.finish(username="student", total_tokens=17, cost_usd="0.000041", now=1_001)
    assert admission.admission_id
    assert state.token_count == 17
    assert state.spend_microusd == 41
    assert state.spend_usd == pytest.approx(0.000041)
    assert state.active_concurrency == 0


def test_two_controller_workers_share_atomic_request_windows(monkeypatch):
    monkeypatch.setattr(controls, "PER_USER_REQUESTS_PER_MINUTE", 1)
    first, second, _backend = controllers()
    first.begin(username="same", client_address="192.0.2.1", now=1_000)
    first.finish(username="same", now=1_000)
    with pytest.raises(ResourceLimitError, match="user_rate_limit"):
        second.begin(username="same", client_address="198.51.100.2", now=1_001)


def test_login_account_and_network_limits_are_shared_between_workers(monkeypatch):
    monkeypatch.setattr(controls, "LOGIN_ATTEMPTS_PER_MINUTE", 1)
    first, second, _backend = controllers()
    first.check_login_attempt(username="first", client_address="192.0.2.1", now=1_000)
    with pytest.raises(ResourceLimitError, match="login_network_rate_limit"):
        second.check_login_attempt(username="second", client_address="192.0.2.1", now=1_001)


def test_independent_operation_windows_are_shared_and_return_retry_after():
    first, second, _backend = controllers()
    first.check_operation(
        "course-review",
        identity="student",
        client_address="192.0.2.1",
        limit=1,
        now=1_000,
    )
    with pytest.raises(ResourceLimitError, match="course-review_account_rate_limit") as error:
        second.check_operation(
            "course-review",
            identity="student",
            client_address="198.51.100.2",
            limit=1,
            now=1_001,
        )
    assert error.value.retry_after_seconds == 60


def test_daily_quota_retry_after_reaches_next_utc_day(monkeypatch):
    monkeypatch.setattr(controls, "DAILY_PROVIDER_REQUEST_QUOTA", 1)
    first, second, _backend = controllers()
    first.begin(
        username="student",
        client_address="192.0.2.1",
        now=86_390,
        reservation=UsageReservation(provider_requests=1),
    )
    with pytest.raises(ResourceLimitError, match="daily_provider_request_quota") as error:
        second.begin(username="student", client_address="192.0.2.2", now=86_391)
    assert error.value.retry_after_seconds == 9
    first.finish(username="student", provider_requests=1, now=86_392)


def test_concurrency_admission_is_atomic_and_denial_does_not_consume_rate_window(monkeypatch):
    monkeypatch.setattr(controls, "MAX_CONCURRENT_REQUESTS", 1)
    monkeypatch.setattr(controls, "PER_USER_REQUESTS_PER_MINUTE", 1)
    monkeypatch.setattr(controls, "PER_IP_REQUESTS_PER_MINUTE", 10)
    first, second, _backend = controllers()
    first.begin(username="one", client_address="192.0.2.1", now=1_000)
    with pytest.raises(ResourceLimitError, match="concurrency_limit"):
        second.begin(username="two", client_address="192.0.2.2", now=1_001)
    first.finish(username="one", now=1_002)
    # The rejected attempt did not partially append to user two's rate window.
    second.begin(username="two", client_address="192.0.2.2", now=1_003)
    second.finish(username="two", now=1_004)


def test_simultaneous_worker_admission_has_exactly_one_winner(monkeypatch):
    monkeypatch.setattr(controls, "MAX_CONCURRENT_REQUESTS", 1)
    first, second, backend = controllers()
    barrier = threading.Barrier(2)
    results: list[tuple[str, ResourceController, object]] = []
    result_lock = threading.Lock()

    def enter(controller: ResourceController, username: str, address: str) -> None:
        barrier.wait()
        try:
            value: object = controller.begin(
                username=username,
                client_address=address,
                now=1_000,
            )
            outcome = "accepted"
        except ResourceLimitError as exc:
            value = exc.reason
            outcome = "denied"
        with result_lock:
            results.append((outcome, controller, value))

    threads = [
        threading.Thread(target=enter, args=(first, "one", "192.0.2.1")),
        threading.Thread(target=enter, args=(second, "two", "192.0.2.2")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(item[0] for item in results) == ["accepted", "denied"]
    assert next(item[2] for item in results if item[0] == "denied") == "concurrency_limit"
    assert len(backend._leases) == 1  # exact atomic boundary, inspected only in this backend test
    winner = next(item for item in results if item[0] == "accepted")
    winner[1].finish(username=None, admission=winner[2], now=1_001)


def test_request_window_expires_at_sixty_second_boundary(monkeypatch):
    monkeypatch.setattr(controls, "PER_USER_REQUESTS_PER_MINUTE", 1)
    first, second, _backend = controllers()
    first.begin(username="same", client_address="192.0.2.1", now=1_000)
    first.finish(username="same", now=1_000)
    second.begin(username="same", client_address="192.0.2.1", now=1_060)
    second.finish(username="same", now=1_060)


def test_exact_provider_token_and_microdollar_reservations_are_reconciled(monkeypatch):
    monkeypatch.setattr(controls, "DAILY_PROVIDER_REQUEST_QUOTA", 2)
    monkeypatch.setattr(controls, "DAILY_TOKEN_QUOTA", 100)
    monkeypatch.setattr(controls, "DAILY_SPEND_LIMIT_USD", 0.0001)  # exactly 100 microdollars
    monkeypatch.setattr(controls, "PER_USER_REQUESTS_PER_MINUTE", 20)
    first, second, _backend = controllers()

    first.begin(
        username="student",
        client_address="192.0.2.1",
        now=1_000,
        reservation=UsageReservation(provider_requests=1, tokens=80, cost_microusd=70),
    )
    with pytest.raises(ResourceLimitError, match="daily_token_quota"):
        second.begin(
            username="student",
            client_address="192.0.2.2",
            now=1_001,
            reservation=UsageReservation(tokens=21),
        )

    state = first.finish(
        username="student",
        provider_requests=1,
        total_tokens=50,
        cost_microusd=40,
        now=1_002,
    )
    assert (state.provider_requests, state.token_count, state.spend_microusd) == (1, 50, 40)

    second.begin(
        username="student",
        client_address="192.0.2.2",
        now=1_003,
        reservation=UsageReservation(provider_requests=1, tokens=50, cost_microusd=60),
    )
    # Missing usage retains the conservative reservation instead of silently accounting zero.
    state = second.finish(username="student", now=1_004)
    assert (state.provider_requests, state.token_count, state.spend_microusd) == (2, 100, 100)
    with pytest.raises(ResourceLimitError, match="daily_provider_request_quota"):
        first.begin(username="student", client_address="192.0.2.3", now=1_005)


def test_microdollar_conversion_is_decimal_bounded_and_rejects_invalid_values():
    assert usd_to_microusd("0.1") == 100_000
    assert usd_to_microusd("0.0000014") == 1
    assert usd_to_microusd("0.0000015") == 2
    with pytest.raises(ValueError):
        usd_to_microusd("nan")
    with pytest.raises(ValueError):
        usd_to_microusd("-0.01")


def test_reconciliation_is_idempotent_and_never_double_charges():
    first, _second, _backend = controllers()
    admission = first.begin(username="student", client_address="192.0.2.1", now=1_000)
    first.finish(
        username="student",
        admission=admission,
        total_tokens=10,
        cost_microusd=5,
        now=1_001,
    )
    repeated = first.finish(
        username="student",
        admission=admission,
        total_tokens=999,
        cost_microusd=999,
        now=1_002,
    )
    assert repeated.token_count == 10 and repeated.spend_microusd == 5


def test_expired_lease_releases_concurrency_and_can_still_reconcile(monkeypatch):
    monkeypatch.setattr(controls, "MAX_CONCURRENT_REQUESTS", 1)
    first, second, backend = controllers(lease_ttl_seconds=1)
    admission = first.begin(username="one", client_address="192.0.2.1", now=1_000)
    with pytest.raises(ResourceLimitError, match="concurrency_limit"):
        second.begin(username="two", client_address="192.0.2.2", now=1_000.5)
    second.begin(username="two", client_address="192.0.2.2", now=1_001.1)
    assert backend.usage(
        user_keys=keyring().identifiers("user:two", purpose="request-user"),
        day="1970-01-01",
        now_ms=1_001_100,
    ).active_concurrency == 1
    state = first.finish(
        username="one",
        admission=admission,
        total_tokens=7,
        now=1_001.2,
    )
    assert state.token_count == 7
    second.finish(username="two", now=1_001.3)


def test_usage_and_internal_state_expire_and_cleanup():
    first, _second, backend = controllers(usage_ttl_seconds=1)
    first.begin(username="student", client_address="192.0.2.1", now=1_000)
    first.finish(username="student", total_tokens=9, now=1_000)
    assert first.usage(username="student", now=1_000.5).token_count == 9
    assert backend.cleanup(now_ms=1_001_100) > 0
    assert first.usage(username="student", now=1_001.1).token_count == 0


class FailingBackend:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def __getattr__(self, _name: str):
        def fail(*_args, **_kwargs):
            raise self.error

        return fail


@pytest.mark.parametrize(
    "error,reason",
    [
        (BackendTimeoutError("timeout"), "resource_backend_timeout"),
        (controls.BackendUnavailableError("offline"), "resource_backend_unavailable"),
    ],
)
def test_backend_timeout_and_unavailability_fail_closed_without_local_fallback(error, reason):
    controller = ResourceController(backend=FailingBackend(error), keyring=keyring())
    with pytest.raises(ResourceLimitError, match=reason):
        controller.begin(username="student", client_address="192.0.2.1", now=1_000)


def test_versioned_hmac_rotation_preserves_old_window_and_separates_purposes(monkeypatch):
    monkeypatch.setattr(controls, "PER_USER_REQUESTS_PER_MINUTE", 1)
    backend = SharedInMemoryRateLimitBackend()
    old = ResourceController(backend=backend, keyring=keyring("v1"))
    old.begin(username="student", client_address="192.0.2.1", now=1_000)
    old.finish(username="student", now=1_000)

    rotated_ring = keyring("v2", previous={"v1": KEY_V1})
    rotated = ResourceController(backend=backend, keyring=rotated_ring)
    with pytest.raises(ResourceLimitError, match="user_rate_limit"):
        rotated.begin(username="student", client_address="192.0.2.2", now=1_001)

    identifiers = rotated_ring.identifiers("192.0.2.1", purpose="request-network")
    assert identifiers[0].startswith("v2:") and identifiers[1].startswith("v1:")
    assert "192.0.2.1" not in repr(identifiers)
    assert privacy_identifier(
        "same", purpose="request-user", keyring=rotated_ring
    ) != privacy_identifier("same", purpose="request-network", keyring=rotated_ring)
    assert "primary_secret" not in repr(rotated_ring)


class RecordingRedisClient:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    def eval(self, script: str, _key_count: int, *parts):
        if self.error:
            raise self.error
        self.calls.append((script, parts))
        if script == RedisRateLimitBackend._ADMIT_SCRIPT:
            return [1, "accepted", 1_120_000]
        if script == RedisRateLimitBackend._RECONCILE_SCRIPT:
            return [1, 1, 12, 3]
        if script == RedisRateLimitBackend._USAGE_SCRIPT:
            return [1, 1, 12, 3, 0, 0, 0, 0]
        if script == RedisRateLimitBackend._WINDOW_SCRIPT:
            return [1]
        return [1, 0]

    def scan_iter(self, **_kwargs):
        return iter(())

    def delete(self, *_keys):
        return 0


def test_redis_adapter_uses_atomic_scripts_and_receives_only_hmac_identities():
    client = RecordingRedisClient()
    backend = RedisRateLimitBackend(client, namespace="test:rate:v1")
    controller = ResourceController(backend=backend, keyring=keyring())
    controller.begin(
        username="raw-student",
        client_address="192.0.2.44",
        now=1_000,
        reservation=UsageReservation(provider_requests=1, tokens=12, cost_microusd=3),
    )
    state = controller.finish(username="raw-student", now=1_001)
    assert (state.provider_requests, state.token_count, state.spend_microusd) == (1, 12, 3)
    serialized = repr(client.calls)
    assert "raw-student" not in serialized and "192.0.2.44" not in serialized
    assert any(call[0] == RedisRateLimitBackend._ADMIT_SCRIPT for call in client.calls)
    assert any(call[0] == RedisRateLimitBackend._RECONCILE_SCRIPT for call in client.calls)


def test_redis_adapter_timeout_is_explicit_and_never_falls_back():
    backend = RedisRateLimitBackend(RecordingRedisClient(error=TimeoutError("offline")))
    request = AdmissionRequest(
        user_keys=("v1:user",),
        network_keys=("v1:network",),
        day="1970-01-01",
        now_ms=1_000_000,
        usage_expires_at_ms=2_000_000,
        policy=AdmissionPolicy(
            per_user_requests=1,
            per_network_requests=1,
            daily_provider_requests=1,
            daily_tokens=1,
            daily_cost_microusd=1,
            max_concurrent_requests=1,
        ),
    )
    with pytest.raises(BackendTimeoutError):
        backend.admit(request)


def test_backend_factory_requires_explicit_valid_configuration():
    assert isinstance(build_rate_limit_backend("memory"), SharedInMemoryRateLimitBackend)
    with pytest.raises(BackendUnavailableError, match="RATE_LIMIT_REDIS_URL"):
        build_rate_limit_backend("redis", redis_url="")
    with pytest.raises(ValueError, match="memory.*redis"):
        build_rate_limit_backend("unknown")


def test_shared_memory_admission_overhead_p50_p95(monkeypatch):
    monkeypatch.setattr(controls, "PER_USER_REQUESTS_PER_MINUTE", 10_000)
    monkeypatch.setattr(controls, "PER_IP_REQUESTS_PER_MINUTE", 10_000)
    monkeypatch.setattr(controls, "DAILY_PROVIDER_REQUEST_QUOTA", 10_000)
    monkeypatch.setattr(controls, "DAILY_TOKEN_QUOTA", 10_000_000)
    monkeypatch.setattr(controls, "MAX_CONCURRENT_REQUESTS", 10)
    controller = ResourceController(
        backend=SharedInMemoryRateLimitBackend(),
        keyring=keyring(),
    )
    samples_ms: list[float] = []
    for index in range(1_000):
        started = time.perf_counter_ns()
        controller.begin(
            username="benchmark",
            client_address="192.0.2.1",
            now=1_000 + index / 10_000,
        )
        controller.finish(username="benchmark", now=1_000 + index / 10_000)
        samples_ms.append((time.perf_counter_ns() - started) / 1_000_000)
    ordered = sorted(samples_ms)
    p50 = statistics.median(ordered)
    p95 = ordered[int(len(ordered) * 0.95) - 1]
    print(f"shared-memory admission+reconcile overhead: p50={p50:.4f}ms p95={p95:.4f}ms")
    assert p50 < 5.0
    assert p95 < 10.0
