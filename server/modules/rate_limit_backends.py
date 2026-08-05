"""Atomic storage backends for request admission and exact usage accounting.

The in-memory implementation can be shared by multiple ``ResourceController`` instances in one
process.  The Redis-compatible implementation is intentionally opt-in and never falls back to
local state: a selected shared backend that is unavailable must fail closed.

All time and money values crossing the backend boundary are integers (milliseconds and
micro-dollars).  Raw usernames and network addresses must be HMACed before they reach this module.
"""

from __future__ import annotations

import json
import secrets
import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Iterable, Protocol, runtime_checkable


MAX_SAFE_LUA_INTEGER = 9_007_199_254_740_991


class BackendError(RuntimeError):
    """Base class for storage failures that must not trigger a local fallback."""


class BackendUnavailableError(BackendError):
    pass


class BackendTimeoutError(BackendError):
    pass


class BackendStateError(BackendError):
    pass


class BackendLimitExceeded(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _nonnegative_integer(value: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    if value > MAX_SAFE_LUA_INTEGER:
        raise ValueError(f"{field} exceeds the exact Redis Lua integer range")
    return value


@dataclass(frozen=True)
class UsageReservation:
    provider_requests: int = 0
    tokens: int = 0
    cost_microusd: int = 0

    def __post_init__(self) -> None:
        _nonnegative_integer(self.provider_requests, "provider_requests")
        _nonnegative_integer(self.tokens, "tokens")
        _nonnegative_integer(self.cost_microusd, "cost_microusd")


@dataclass(frozen=True)
class AdmissionPolicy:
    per_user_requests: int
    per_network_requests: int
    daily_provider_requests: int | None
    daily_tokens: int | None
    daily_cost_microusd: int | None
    max_concurrent_requests: int
    window_ms: int = 60_000
    lease_ttl_ms: int = 120_000

    def __post_init__(self) -> None:
        for name in (
            "per_user_requests",
            "per_network_requests",
            "max_concurrent_requests",
            "window_ms",
            "lease_ttl_ms",
        ):
            if _nonnegative_integer(getattr(self, name), name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("daily_provider_requests", "daily_tokens", "daily_cost_microusd"):
            value = getattr(self, name)
            if value is not None and _nonnegative_integer(value, name) < 1:
                raise ValueError(f"{name} must be positive when enabled")


@dataclass(frozen=True)
class AdmissionRequest:
    user_keys: tuple[str, ...]
    network_keys: tuple[str, ...]
    day: str
    now_ms: int
    usage_expires_at_ms: int
    policy: AdmissionPolicy
    reservation: UsageReservation = UsageReservation()
    admission_id: str = ""

    def __post_init__(self) -> None:
        if not self.user_keys or not self.network_keys:
            raise ValueError("user_keys and network_keys cannot be empty")
        if not self.day:
            raise ValueError("day cannot be empty")
        _nonnegative_integer(self.now_ms, "now_ms")
        _nonnegative_integer(self.usage_expires_at_ms, "usage_expires_at_ms")
        if self.usage_expires_at_ms <= self.now_ms:
            raise ValueError("usage_expires_at_ms must be in the future")


@dataclass(frozen=True)
class AdmissionLease:
    admission_id: str
    user_keys: tuple[str, ...]
    day: str
    expires_at_ms: int
    reservation: UsageReservation


@dataclass(frozen=True)
class ReconciliationRequest:
    admission_id: str | None
    user_keys: tuple[str, ...]
    day: str
    now_ms: int
    usage_expires_at_ms: int
    actual: UsageReservation

    def __post_init__(self) -> None:
        if not self.user_keys or not self.day:
            raise ValueError("reconciliation requires user_keys and day")
        _nonnegative_integer(self.now_ms, "now_ms")
        _nonnegative_integer(self.usage_expires_at_ms, "usage_expires_at_ms")
        if self.usage_expires_at_ms <= self.now_ms:
            raise ValueError("usage_expires_at_ms must be in the future")


@dataclass(frozen=True)
class UsageSnapshot:
    provider_requests: int
    tokens: int
    cost_microusd: int
    reserved_provider_requests: int = 0
    reserved_tokens: int = 0
    reserved_cost_microusd: int = 0
    active_concurrency: int = 0


@runtime_checkable
class RateLimitBackend(Protocol):
    def admit(self, request: AdmissionRequest) -> AdmissionLease: ...

    def check_rate_limits(
        self,
        *,
        windows: tuple[tuple[str, tuple[str, ...], int], ...],
        now_ms: int,
        window_ms: int,
        event_id: str,
    ) -> None: ...

    def reconcile(self, request: ReconciliationRequest) -> UsageSnapshot: ...

    def usage(
        self,
        *,
        user_keys: tuple[str, ...],
        day: str,
        now_ms: int,
    ) -> UsageSnapshot: ...

    def cleanup(self, *, now_ms: int) -> int: ...

    def reset(self) -> None: ...


@dataclass
class _UsageRecord:
    provider_requests: int = 0
    tokens: int = 0
    cost_microusd: int = 0
    expires_at_ms: int = 0


@dataclass(frozen=True)
class _LeaseRecord:
    lease: AdmissionLease
    usage_expires_at_ms: int


class SharedInMemoryRateLimitBackend:
    """Thread-safe state that can be shared across controller/worker objects in one process."""

    def __init__(self, *, settled_ttl_ms: int = 86_400_000) -> None:
        self._lock = threading.RLock()
        self._windows: dict[tuple[str, str], deque[int]] = defaultdict(deque)
        self._usage: dict[tuple[str, str], _UsageRecord] = {}
        self._leases: dict[str, _LeaseRecord] = {}
        self._expired_leases: dict[str, _LeaseRecord] = {}
        self._settled: dict[str, int] = {}
        self._settled_ttl_ms = max(1, int(settled_ttl_ms))

    @staticmethod
    def _unique(values: Iterable[str]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(str(value) for value in values if str(value)))

    def _cleanup_locked(self, now_ms: int) -> int:
        removed = 0
        for key, window in list(self._windows.items()):
            while window and window[0] <= now_ms - 60_000:
                window.popleft()
                removed += 1
            if not window:
                self._windows.pop(key, None)
        for key, record in list(self._usage.items()):
            if record.expires_at_ms <= now_ms:
                self._usage.pop(key, None)
                removed += 1
        for admission_id, record in list(self._leases.items()):
            if record.lease.expires_at_ms <= now_ms:
                self._leases.pop(admission_id, None)
                self._expired_leases[admission_id] = record
                removed += 1
        for admission_id, record in list(self._expired_leases.items()):
            if record.usage_expires_at_ms <= now_ms:
                self._expired_leases.pop(admission_id, None)
                removed += 1
        for admission_id, expires_at_ms in list(self._settled.items()):
            if expires_at_ms <= now_ms:
                self._settled.pop(admission_id, None)
                removed += 1
        return removed

    def _window_count(self, kind: str, keys: tuple[str, ...], cutoff_ms: int) -> int:
        maximum = 0
        for identity in self._unique(keys):
            window = self._windows[(kind, identity)]
            while window and window[0] <= cutoff_ms:
                window.popleft()
            maximum = max(maximum, len(window))
        return maximum

    def _actual_totals(self, day: str, user_keys: tuple[str, ...]) -> UsageReservation:
        records = [
            self._usage.get((day, identity), _UsageRecord())
            for identity in self._unique(user_keys)
        ]
        return UsageReservation(
            provider_requests=sum(record.provider_requests for record in records),
            tokens=sum(record.tokens for record in records),
            cost_microusd=sum(record.cost_microusd for record in records),
        )

    def _reserved_totals(self, day: str, user_keys: tuple[str, ...]) -> UsageReservation:
        aliases = set(self._unique(user_keys))
        reservations = [
            record.lease.reservation
            for record in self._leases.values()
            if record.lease.day == day and set(record.lease.user_keys) & aliases
        ]
        return UsageReservation(
            provider_requests=sum(item.provider_requests for item in reservations),
            tokens=sum(item.tokens for item in reservations),
            cost_microusd=sum(item.cost_microusd for item in reservations),
        )

    def _snapshot(self, day: str, user_keys: tuple[str, ...]) -> UsageSnapshot:
        actual = self._actual_totals(day, user_keys)
        reserved = self._reserved_totals(day, user_keys)
        return UsageSnapshot(
            provider_requests=actual.provider_requests,
            tokens=actual.tokens,
            cost_microusd=actual.cost_microusd,
            reserved_provider_requests=reserved.provider_requests,
            reserved_tokens=reserved.tokens,
            reserved_cost_microusd=reserved.cost_microusd,
            active_concurrency=len(self._leases),
        )

    @staticmethod
    def _check_quota(current: int, incoming: int, limit: int | None, reason: str) -> None:
        if limit is not None and (current >= limit or current + incoming > limit):
            raise BackendLimitExceeded(reason)

    def admit(self, request: AdmissionRequest) -> AdmissionLease:
        admission_id = request.admission_id or secrets.token_hex(16)
        if not admission_id or any(char in admission_id for char in "\r\n\x00"):
            raise ValueError("invalid admission_id")
        user_keys = self._unique(request.user_keys)
        network_keys = self._unique(request.network_keys)
        with self._lock:
            self._cleanup_locked(request.now_ms)
            cutoff = request.now_ms - request.policy.window_ms
            if self._window_count("user", user_keys, cutoff) >= request.policy.per_user_requests:
                raise BackendLimitExceeded("user_rate_limit")
            if (
                self._window_count("network", network_keys, cutoff)
                >= request.policy.per_network_requests
            ):
                raise BackendLimitExceeded("network_rate_limit")
            if len(self._leases) >= request.policy.max_concurrent_requests:
                raise BackendLimitExceeded("concurrency_limit")

            actual = self._actual_totals(request.day, user_keys)
            reserved = self._reserved_totals(request.day, user_keys)
            self._check_quota(
                actual.provider_requests + reserved.provider_requests,
                request.reservation.provider_requests,
                request.policy.daily_provider_requests,
                "daily_provider_request_quota",
            )
            self._check_quota(
                actual.tokens + reserved.tokens,
                request.reservation.tokens,
                request.policy.daily_tokens,
                "daily_token_quota",
            )
            self._check_quota(
                actual.cost_microusd + reserved.cost_microusd,
                request.reservation.cost_microusd,
                request.policy.daily_cost_microusd,
                "daily_budget_cutoff",
            )

            # Mutate only after every admission condition passes.
            self._windows[("user", user_keys[0])].append(request.now_ms)
            self._windows[("network", network_keys[0])].append(request.now_ms)
            lease = AdmissionLease(
                admission_id=admission_id,
                user_keys=user_keys,
                day=request.day,
                expires_at_ms=request.now_ms + request.policy.lease_ttl_ms,
                reservation=request.reservation,
            )
            self._leases[admission_id] = _LeaseRecord(lease, request.usage_expires_at_ms)
            return lease

    def check_rate_limits(
        self,
        *,
        windows: tuple[tuple[str, tuple[str, ...], int], ...],
        now_ms: int,
        window_ms: int,
        event_id: str,
    ) -> None:
        with self._lock:
            self._cleanup_locked(now_ms)
            cutoff = now_ms - window_ms
            normalized: list[tuple[str, tuple[str, ...], int]] = []
            for kind, keys, limit in windows:
                aliases = self._unique(keys)
                if not aliases or limit < 1:
                    raise ValueError("rate-limit windows require identities and positive limits")
                if self._window_count(kind, aliases, cutoff) >= limit:
                    raise BackendLimitExceeded(f"{kind}_rate_limit")
                normalized.append((kind, aliases, limit))
            for kind, aliases, _limit in normalized:
                self._windows[(kind, aliases[0])].append(now_ms)

    def reconcile(self, request: ReconciliationRequest) -> UsageSnapshot:
        with self._lock:
            self._cleanup_locked(request.now_ms)
            if request.admission_id and request.admission_id in self._settled:
                return self._snapshot(request.day, request.user_keys)

            record: _LeaseRecord | None = None
            if request.admission_id:
                record = self._leases.pop(request.admission_id, None)
                if record is None:
                    record = self._expired_leases.pop(request.admission_id, None)
                if record is None:
                    raise BackendStateError("unknown or expired admission")
                day = record.lease.day
                user_keys = record.lease.user_keys
            else:
                day = request.day
                user_keys = self._unique(request.user_keys)

            usage_key = (day, user_keys[0])
            usage = self._usage.setdefault(usage_key, _UsageRecord())
            usage.provider_requests += request.actual.provider_requests
            usage.tokens += request.actual.tokens
            usage.cost_microusd += request.actual.cost_microusd
            usage.expires_at_ms = max(usage.expires_at_ms, request.usage_expires_at_ms)
            if request.admission_id:
                self._settled[request.admission_id] = request.now_ms + self._settled_ttl_ms
            return self._snapshot(day, user_keys)

    def usage(
        self,
        *,
        user_keys: tuple[str, ...],
        day: str,
        now_ms: int,
    ) -> UsageSnapshot:
        with self._lock:
            self._cleanup_locked(now_ms)
            return self._snapshot(day, self._unique(user_keys))

    def cleanup(self, *, now_ms: int) -> int:
        with self._lock:
            return self._cleanup_locked(now_ms)

    def reset(self) -> None:
        with self._lock:
            self._windows.clear()
            self._usage.clear()
            self._leases.clear()
            self._expired_leases.clear()
            self._settled.clear()


class RedisRateLimitBackend:
    """Minimal synchronous Redis-compatible adapter backed by atomic Lua scripts.

    The supplied client must expose ``eval``. ``from_url`` additionally requires the optional
    ``redis`` package.  Every error is converted to an explicit backend error; this class never
    constructs or consults an in-memory fallback.
    """

    _ADMIT_SCRIPT = r"""
local now = tonumber(ARGV[1])
local cutoff = tonumber(ARGV[2])
local user_limit = tonumber(ARGV[3])
local network_limit = tonumber(ARGV[4])
local provider_limit = tonumber(ARGV[5])
local token_limit = tonumber(ARGV[6])
local cost_limit = tonumber(ARGV[7])
local concurrency_limit = tonumber(ARGV[8])
local lease_id = ARGV[9]
local lease_expiry = tonumber(ARGV[10])
local reserve_provider = tonumber(ARGV[11])
local reserve_tokens = tonumber(ARGV[12])
local reserve_cost = tonumber(ARGV[13])
local usage_ttl = tonumber(ARGV[14])
local user_windows = cjson.decode(ARGV[15])
local network_windows = cjson.decode(ARGV[16])
local usage_keys = cjson.decode(ARGV[17])
local pending_keys = cjson.decode(ARGV[18])
local pending_member = ARGV[19]

local function max_window(keys)
  local maximum = 0
  for _, key in ipairs(keys) do
    redis.call('ZREMRANGEBYSCORE', key, '-inf', cutoff)
    local count = redis.call('ZCARD', key)
    if count > maximum then maximum = count end
  end
  return maximum
end

if max_window(user_windows) >= user_limit then return {0, 'user_rate_limit'} end
if max_window(network_windows) >= network_limit then return {0, 'network_rate_limit'} end
redis.call('ZREMRANGEBYSCORE', KEYS[4], '-inf', now)
if redis.call('ZCARD', KEYS[4]) >= concurrency_limit then return {0, 'concurrency_limit'} end

local actual_provider, actual_tokens, actual_cost = 0, 0, 0
for _, key in ipairs(usage_keys) do
  actual_provider = actual_provider + tonumber(redis.call('HGET', key, 'provider_requests') or '0')
  actual_tokens = actual_tokens + tonumber(redis.call('HGET', key, 'tokens') or '0')
  actual_cost = actual_cost + tonumber(redis.call('HGET', key, 'cost_microusd') or '0')
end
local pending_provider, pending_tokens, pending_cost = 0, 0, 0
for _, key in ipairs(pending_keys) do
  redis.call('ZREMRANGEBYSCORE', key, '-inf', now)
  for _, member in ipairs(redis.call('ZRANGE', key, 0, -1)) do
    local p, t, c = string.match(member, '^(%d+):(%d+):(%d+):')
    pending_provider = pending_provider + tonumber(p or '0')
    pending_tokens = pending_tokens + tonumber(t or '0')
    pending_cost = pending_cost + tonumber(c or '0')
  end
end
if provider_limit >= 0 and (actual_provider + pending_provider >= provider_limit or
  actual_provider + pending_provider + reserve_provider > provider_limit) then
  return {0, 'daily_provider_request_quota'}
end
if token_limit >= 0 and (actual_tokens + pending_tokens >= token_limit or
  actual_tokens + pending_tokens + reserve_tokens > token_limit) then
  return {0, 'daily_token_quota'}
end
if cost_limit >= 0 and (actual_cost + pending_cost >= cost_limit or
  actual_cost + pending_cost + reserve_cost > cost_limit) then
  return {0, 'daily_budget_cutoff'}
end

redis.call('ZADD', KEYS[1], now, lease_id)
redis.call('ZADD', KEYS[2], now, lease_id)
redis.call('ZADD', KEYS[4], lease_expiry, lease_id)
redis.call('ZADD', KEYS[6], lease_expiry, pending_member)
redis.call('HSET', KEYS[5], 'pending_member', pending_member, 'day', ARGV[20],
  'user_keys', ARGV[17], 'usage_keys', ARGV[17], 'pending_keys', ARGV[18])
redis.call('PEXPIRE', KEYS[1], usage_ttl)
redis.call('PEXPIRE', KEYS[2], usage_ttl)
redis.call('PEXPIRE', KEYS[3], usage_ttl)
redis.call('PEXPIRE', KEYS[4], usage_ttl)
redis.call('PEXPIRE', KEYS[5], usage_ttl)
redis.call('PEXPIRE', KEYS[6], usage_ttl)
return {1, lease_id, lease_expiry}
"""

    _WINDOW_SCRIPT = r"""
local now = tonumber(ARGV[1])
local cutoff = tonumber(ARGV[2])
local event_id = ARGV[3]
local groups = cjson.decode(ARGV[4])
for _, group in ipairs(groups) do
  local maximum = 0
  for _, key in ipairs(group.keys) do
    redis.call('ZREMRANGEBYSCORE', key, '-inf', cutoff)
    local count = redis.call('ZCARD', key)
    if count > maximum then maximum = count end
  end
  if maximum >= tonumber(group.limit) then return {0, group.kind .. '_rate_limit'} end
end
for _, group in ipairs(groups) do
  redis.call('ZADD', group.keys[1], now, event_id .. ':' .. group.kind)
  redis.call('PEXPIRE', group.keys[1], tonumber(ARGV[5]))
end
return {1}
"""

    _RECONCILE_SCRIPT = r"""
local now = tonumber(ARGV[1])
local actual_provider = tonumber(ARGV[2])
local actual_tokens = tonumber(ARGV[3])
local actual_cost = tonumber(ARGV[4])
local usage_ttl = tonumber(ARGV[5])
local settled_ttl = tonumber(ARGV[6])
local usage_keys = cjson.decode(ARGV[7])
local pending_keys = cjson.decode(ARGV[8])
local admission_id = ARGV[9]
if redis.call('EXISTS', KEYS[5]) == 0 then
  if redis.call('EXISTS', KEYS[2]) == 0 then return {0, 'unknown_or_expired_admission'} end
  local pending_member = redis.call('HGET', KEYS[2], 'pending_member')
  if pending_member then
    for _, key in ipairs(pending_keys) do redis.call('ZREM', key, pending_member) end
  end
  redis.call('ZREM', KEYS[1], admission_id)
  redis.call('HINCRBY', KEYS[3], 'provider_requests', actual_provider)
  redis.call('HINCRBY', KEYS[3], 'tokens', actual_tokens)
  redis.call('HINCRBY', KEYS[3], 'cost_microusd', actual_cost)
  redis.call('PEXPIRE', KEYS[3], usage_ttl)
  redis.call('SET', KEYS[5], '1', 'PX', settled_ttl)
  redis.call('DEL', KEYS[2])
end
local provider, tokens, cost = 0, 0, 0
for _, key in ipairs(usage_keys) do
  provider = provider + tonumber(redis.call('HGET', key, 'provider_requests') or '0')
  tokens = tokens + tonumber(redis.call('HGET', key, 'tokens') or '0')
  cost = cost + tonumber(redis.call('HGET', key, 'cost_microusd') or '0')
end
return {1, provider, tokens, cost}
"""

    _USAGE_SCRIPT = r"""
local now = tonumber(ARGV[1])
local usage_keys = cjson.decode(ARGV[2])
local pending_keys = cjson.decode(ARGV[3])
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now)
local provider, tokens, cost = 0, 0, 0
for _, key in ipairs(usage_keys) do
  provider = provider + tonumber(redis.call('HGET', key, 'provider_requests') or '0')
  tokens = tokens + tonumber(redis.call('HGET', key, 'tokens') or '0')
  cost = cost + tonumber(redis.call('HGET', key, 'cost_microusd') or '0')
end
local pending_provider, pending_tokens, pending_cost = 0, 0, 0
for _, key in ipairs(pending_keys) do
  redis.call('ZREMRANGEBYSCORE', key, '-inf', now)
  for _, member in ipairs(redis.call('ZRANGE', key, 0, -1)) do
    local p, t, c = string.match(member, '^(%d+):(%d+):(%d+):')
    pending_provider = pending_provider + tonumber(p or '0')
    pending_tokens = pending_tokens + tonumber(t or '0')
    pending_cost = pending_cost + tonumber(c or '0')
  end
end
return {1, provider, tokens, cost, pending_provider, pending_tokens, pending_cost,
  redis.call('ZCARD', KEYS[1])}
"""

    def __init__(self, client: Any, *, namespace: str = "advisu:rate-limit:v1") -> None:
        if client is None or not callable(getattr(client, "eval", None)):
            raise ValueError("a synchronous Redis-compatible client with eval() is required")
        clean_namespace = str(namespace).strip().strip(":")
        if not clean_namespace or any(char.isspace() for char in clean_namespace):
            raise ValueError("namespace must be non-empty and contain no whitespace")
        self._client = client
        self._namespace = clean_namespace

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        namespace: str = "advisu:rate-limit:v1",
        socket_timeout_seconds: float = 0.25,
    ) -> "RedisRateLimitBackend":
        if not str(url).strip():
            raise ValueError("Redis URL is required")
        try:
            import redis  # type: ignore[import-not-found]
        except ImportError as exc:
            raise BackendUnavailableError(
                "Redis rate-limit backend selected but the optional redis package is unavailable"
            ) from exc
        client = redis.Redis.from_url(
            url,
            socket_timeout=max(0.001, float(socket_timeout_seconds)),
            socket_connect_timeout=max(0.001, float(socket_timeout_seconds)),
            decode_responses=False,
        )
        return cls(client, namespace=namespace)

    def _key(self, *parts: str) -> str:
        return ":".join((self._namespace, *parts))

    @staticmethod
    def _decode(value: Any) -> str:
        return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)

    def _eval(self, script: str, keys: list[str], args: list[Any]) -> list[Any]:
        try:
            result = self._client.eval(script, len(keys), *keys, *args)
        except Exception as exc:
            name = type(exc).__name__.casefold()
            if "timeout" in name:
                raise BackendTimeoutError("rate-limit backend timed out") from exc
            raise BackendUnavailableError("rate-limit backend unavailable") from exc
        if not isinstance(result, (list, tuple)) or not result:
            raise BackendStateError("rate-limit backend returned a malformed result")
        return list(result)

    def _raise_if_denied(self, result: list[Any]) -> None:
        if int(result[0]) == 1:
            return
        reason = self._decode(result[1]) if len(result) > 1 else "backend_denied"
        if reason == "unknown_or_expired_admission":
            raise BackendStateError(reason)
        raise BackendLimitExceeded(reason)

    def _window_keys(self, kind: str, identities: tuple[str, ...]) -> list[str]:
        return [self._key("window", kind, identity) for identity in dict.fromkeys(identities)]

    def _usage_keys(self, day: str, identities: tuple[str, ...]) -> list[str]:
        return [self._key("usage", day, identity) for identity in dict.fromkeys(identities)]

    def _pending_keys(self, day: str, identities: tuple[str, ...]) -> list[str]:
        return [self._key("pending", day, identity) for identity in dict.fromkeys(identities)]

    def admit(self, request: AdmissionRequest) -> AdmissionLease:
        admission_id = request.admission_id or secrets.token_hex(16)
        user_windows = self._window_keys("user", request.user_keys)
        network_windows = self._window_keys("network", request.network_keys)
        usage_keys = self._usage_keys(request.day, request.user_keys)
        pending_keys = self._pending_keys(request.day, request.user_keys)
        pending_member = (
            f"{request.reservation.provider_requests}:{request.reservation.tokens}:"
            f"{request.reservation.cost_microusd}:{admission_id}"
        )
        lease_expiry = request.now_ms + request.policy.lease_ttl_ms
        active_key = self._key("active")
        lease_key = self._key("lease", admission_id)
        result = self._eval(
            self._ADMIT_SCRIPT,
            [
                user_windows[0],
                network_windows[0],
                usage_keys[0],
                active_key,
                lease_key,
                pending_keys[0],
            ],
            [
                request.now_ms,
                request.now_ms - request.policy.window_ms,
                request.policy.per_user_requests,
                request.policy.per_network_requests,
                request.policy.daily_provider_requests
                if request.policy.daily_provider_requests is not None
                else -1,
                request.policy.daily_tokens if request.policy.daily_tokens is not None else -1,
                request.policy.daily_cost_microusd
                if request.policy.daily_cost_microusd is not None
                else -1,
                request.policy.max_concurrent_requests,
                admission_id,
                lease_expiry,
                request.reservation.provider_requests,
                request.reservation.tokens,
                request.reservation.cost_microusd,
                max(1, request.usage_expires_at_ms - request.now_ms),
                json.dumps(user_windows, separators=(",", ":")),
                json.dumps(network_windows, separators=(",", ":")),
                json.dumps(usage_keys, separators=(",", ":")),
                json.dumps(pending_keys, separators=(",", ":")),
                pending_member,
                request.day,
            ],
        )
        self._raise_if_denied(result)
        return AdmissionLease(
            admission_id=admission_id,
            user_keys=request.user_keys,
            day=request.day,
            expires_at_ms=lease_expiry,
            reservation=request.reservation,
        )

    def check_rate_limits(
        self,
        *,
        windows: tuple[tuple[str, tuple[str, ...], int], ...],
        now_ms: int,
        window_ms: int,
        event_id: str,
    ) -> None:
        groups = [
            {"kind": kind, "keys": self._window_keys(kind, keys), "limit": limit}
            for kind, keys, limit in windows
        ]
        expiry_ttl = max(window_ms * 2, 120_000)
        result = self._eval(
            self._WINDOW_SCRIPT,
            [],
            [
                now_ms,
                now_ms - window_ms,
                event_id,
                json.dumps(groups, separators=(",", ":")),
                expiry_ttl,
            ],
        )
        self._raise_if_denied(result)

    def reconcile(self, request: ReconciliationRequest) -> UsageSnapshot:
        if not request.admission_id:
            raise BackendStateError("Redis reconciliation requires an admission ID")
        usage_keys = self._usage_keys(request.day, request.user_keys)
        pending_keys = self._pending_keys(request.day, request.user_keys)
        result = self._eval(
            self._RECONCILE_SCRIPT,
            [
                self._key("active"),
                self._key("lease", request.admission_id),
                usage_keys[0],
                pending_keys[0],
                self._key("settled", request.admission_id),
            ],
            [
                request.now_ms,
                request.actual.provider_requests,
                request.actual.tokens,
                request.actual.cost_microusd,
                max(1, request.usage_expires_at_ms - request.now_ms),
                86_400_000,
                json.dumps(usage_keys, separators=(",", ":")),
                json.dumps(pending_keys, separators=(",", ":")),
                request.admission_id,
            ],
        )
        self._raise_if_denied(result)
        current = self.usage(user_keys=request.user_keys, day=request.day, now_ms=request.now_ms)
        return UsageSnapshot(
            provider_requests=int(result[1]),
            tokens=int(result[2]),
            cost_microusd=int(result[3]),
            reserved_provider_requests=current.reserved_provider_requests,
            reserved_tokens=current.reserved_tokens,
            reserved_cost_microusd=current.reserved_cost_microusd,
            active_concurrency=current.active_concurrency,
        )

    def usage(
        self,
        *,
        user_keys: tuple[str, ...],
        day: str,
        now_ms: int,
    ) -> UsageSnapshot:
        usage_keys = self._usage_keys(day, user_keys)
        pending_keys = self._pending_keys(day, user_keys)
        result = self._eval(
            self._USAGE_SCRIPT,
            [self._key("active")],
            [
                now_ms,
                json.dumps(usage_keys, separators=(",", ":")),
                json.dumps(pending_keys, separators=(",", ":")),
            ],
        )
        self._raise_if_denied(result)
        return UsageSnapshot(
            provider_requests=int(result[1]),
            tokens=int(result[2]),
            cost_microusd=int(result[3]),
            reserved_provider_requests=int(result[4]),
            reserved_tokens=int(result[5]),
            reserved_cost_microusd=int(result[6]),
            active_concurrency=int(result[7]),
        )

    def cleanup(self, *, now_ms: int) -> int:
        result = self._eval(
            "local n=redis.call('ZREMRANGEBYSCORE',KEYS[1],'-inf',ARGV[1]); return {1,n}",
            [self._key("active")],
            [now_ms],
        )
        self._raise_if_denied(result)
        return int(result[1])

    def reset(self) -> None:
        try:
            keys = list(self._client.scan_iter(match=f"{self._namespace}:*", count=500))
            for start in range(0, len(keys), 500):
                batch = keys[start : start + 500]
                if batch:
                    self._client.delete(*batch)
        except Exception as exc:
            name = type(exc).__name__.casefold()
            if "timeout" in name:
                raise BackendTimeoutError("rate-limit backend timed out") from exc
            raise BackendUnavailableError("rate-limit backend unavailable") from exc


def build_rate_limit_backend(
    backend: str,
    *,
    redis_url: str | None = None,
    redis_namespace: str = "advisu:rate-limit:v1",
    redis_timeout_seconds: float = 0.25,
) -> RateLimitBackend:
    """Build the explicitly selected backend without any cross-backend fallback."""

    selected = str(backend or "").strip().casefold()
    if selected in {"memory", "in_memory", "in-memory"}:
        return SharedInMemoryRateLimitBackend()
    if selected == "redis":
        if not str(redis_url or "").strip():
            raise BackendUnavailableError(
                "Redis rate-limit backend selected but RATE_LIMIT_REDIS_URL is missing"
            )
        return RedisRateLimitBackend.from_url(
            str(redis_url),
            namespace=redis_namespace,
            socket_timeout_seconds=redis_timeout_seconds,
        )
    raise ValueError("RATE_LIMIT_BACKEND must be 'memory' or 'redis'")
