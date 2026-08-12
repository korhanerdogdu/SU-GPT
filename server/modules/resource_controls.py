"""Repository-native admission, quota, concurrency, and exact cost controls.

``ResourceController`` keeps the original ``begin``/``finish`` API while delegating all mutable
state to a pluggable atomic backend.  Controllers in different worker objects can share the same
backend.  A selected backend failure always fails closed; no Redis-to-local fallback exists.
"""

from __future__ import annotations

import contextvars
import hashlib
import hmac
import os
import re
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Mapping

from modules.rate_limit_backends import (
    AdmissionLease,
    AdmissionPolicy,
    AdmissionRequest,
    BackendLimitExceeded,
    BackendStateError,
    BackendTimeoutError,
    BackendUnavailableError,
    RateLimitBackend,
    ReconciliationRequest,
    SharedInMemoryRateLimitBackend,
    UsageReservation,
    UsageSnapshot,
    build_rate_limit_backend,
)


def _positive_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _positive_float(name: str, default: float) -> float:
    try:
        return max(0.0, float(os.getenv(name, str(default))))
    except ValueError:
        return default


def _nonnegative_int(name: str, default: int) -> int:
    try:
        return max(0, int(os.getenv(name, str(default))))
    except ValueError:
        return default


MAX_INPUT_CHARS = _positive_int("MAX_INPUT_CHARS", 8_000)
MAX_OUTPUT_TOKENS = _positive_int("MAX_OUTPUT_TOKENS", 1_024)
MAX_UPLOAD_BYTES = _positive_int("MAX_UPLOAD_BYTES", 10 * 1024 * 1024)
MAX_UPLOAD_FILES = _positive_int("MAX_UPLOAD_FILES", 10)
MAX_UPLOAD_TOTAL_BYTES = _positive_int("MAX_UPLOAD_TOTAL_BYTES", 25 * 1024 * 1024)
REQUEST_TIMEOUT_SECONDS = _positive_float("REQUEST_TIMEOUT_SECONDS", 60.0)
PER_USER_REQUESTS_PER_MINUTE = _positive_int("PER_USER_REQUESTS_PER_MINUTE", 30)
PER_IP_REQUESTS_PER_MINUTE = _positive_int("PER_IP_REQUESTS_PER_MINUTE", 60)
LOGIN_ATTEMPTS_PER_MINUTE = _positive_int("LOGIN_ATTEMPTS_PER_MINUTE", 5)
STREAM_REQUESTS_PER_MINUTE = _positive_int("STREAM_REQUESTS_PER_MINUTE", 20)
UPLOAD_REQUESTS_PER_MINUTE = _positive_int("UPLOAD_REQUESTS_PER_MINUTE", 10)
COURSE_REVIEW_REQUESTS_PER_MINUTE = _positive_int(
    "COURSE_REVIEW_REQUESTS_PER_MINUTE", 6
)
CONVERSATION_CREATIONS_PER_MINUTE = _positive_int(
    "CONVERSATION_CREATIONS_PER_MINUTE", 10
)
DAILY_PROVIDER_REQUEST_QUOTA = _positive_int("DAILY_PROVIDER_REQUEST_QUOTA", 20)
DAILY_TOKEN_QUOTA = _positive_int("DAILY_TOKEN_QUOTA", 100_000)
# No product cost budget exists in the repository. Zero means disabled until the owner sets one;
# evaluation must report this acceptance gate as unresolved rather than inventing a dollar limit.
DAILY_SPEND_LIMIT_USD = _positive_float("DAILY_SPEND_LIMIT_USD", 0.0)
PROVIDER_REQUEST_COST_RESERVATION_MICROUSD = _nonnegative_int(
    "PROVIDER_REQUEST_COST_RESERVATION_MICROUSD", 0
)
if DAILY_SPEND_LIMIT_USD and PROVIDER_REQUEST_COST_RESERVATION_MICROUSD < 1:
    raise RuntimeError(
        "A positive daily spend limit requires PROVIDER_REQUEST_COST_RESERVATION_MICROUSD"
    )
SOFT_BUDGET_FRACTION = min(0.99, _positive_float("SOFT_BUDGET_FRACTION", 0.8))
MAX_CONCURRENT_REQUESTS = _positive_int("MAX_CONCURRENT_REQUESTS", 4)
RESOURCE_RESERVATION_TTL_SECONDS = _positive_float(
    "RESOURCE_RESERVATION_TTL_SECONDS", max(120.0, REQUEST_TIMEOUT_SECONDS * 2)
)

MICRODOLLARS_PER_DOLLAR = 1_000_000
_KEY_VERSION_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,31}$")


class ResourceLimitError(RuntimeError):
    def __init__(self, reason: str, *, retry_after_seconds: int | None = None):
        super().__init__(reason)
        self.reason = reason
        self.retry_after_seconds = retry_after_seconds


def usd_to_microusd(value: int | float | str | Decimal | None) -> int:
    """Convert a USD amount once at the application boundary using decimal arithmetic."""

    if value is None:
        return 0
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("cost_usd must be a finite non-negative decimal") from exc
    if not amount.is_finite() or amount < 0:
        raise ValueError("cost_usd must be a finite non-negative decimal")
    return int((amount * MICRODOLLARS_PER_DOLLAR).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def microusd_to_usd(value: int) -> float:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("cost_microusd must be a non-negative integer")
    return float(Decimal(value) / MICRODOLLARS_PER_DOLLAR)


@dataclass(frozen=True)
class PrivacyKeyring:
    """Versioned HMAC keys used only to derive opaque backend identities.

    Previous keys make a coordinated rotation recognize quota/window state written under an older
    version.  Deployments should update workers together: a worker that knows only the old key
    cannot discover traffic already written solely under the new version.
    """

    primary_version: str
    primary_secret: bytes = field(repr=False)
    previous_keys: tuple[tuple[str, bytes], ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        entries = ((self.primary_version, self.primary_secret), *self.previous_keys)
        versions: set[str] = set()
        for version, secret in entries:
            if not _KEY_VERSION_RE.fullmatch(str(version)):
                raise ValueError("privacy key versions must be lowercase stable identifiers")
            if version in versions:
                raise ValueError("privacy key versions must be unique")
            versions.add(version)
            if not isinstance(secret, bytes) or len(secret) < 16:
                raise ValueError("privacy HMAC keys must contain at least 16 bytes")

    @classmethod
    def from_environment(cls) -> "PrivacyKeyring":
        raw_primary = os.getenv("ABUSE_HASH_SECRET", "")
        primary = raw_primary.encode() if raw_primary else secrets.token_bytes(32)
        version = os.getenv("ABUSE_HASH_SECRET_VERSION", "v1").strip().lower() or "v1"
        previous: list[tuple[str, bytes]] = []
        raw_previous = os.getenv("ABUSE_HASH_PREVIOUS_SECRETS", "").strip()
        for item in (part.strip() for part in raw_previous.split(",") if part.strip()):
            if "=" not in item:
                raise ValueError("ABUSE_HASH_PREVIOUS_SECRETS entries must use version=secret")
            old_version, old_secret = item.split("=", 1)
            previous.append((old_version.strip().lower(), old_secret.encode()))
        return cls(version, primary, tuple(previous))

    @classmethod
    def from_mapping(
        cls,
        *,
        primary_version: str,
        primary_secret: bytes,
        previous: Mapping[str, bytes] | None = None,
    ) -> "PrivacyKeyring":
        return cls(primary_version, primary_secret, tuple((previous or {}).items()))

    @staticmethod
    def _identifier(version: str, secret: bytes, raw: str | None, purpose: str) -> str:
        message = f"{purpose}\x00{str(raw or 'unknown')}".encode("utf-8")
        digest = hmac.new(secret, message, hashlib.sha256).hexdigest()[:32]
        return f"{version}:{digest}"

    def identifiers(self, raw: str | None, *, purpose: str) -> tuple[str, ...]:
        if not purpose or any(char in purpose for char in "\r\n\x00"):
            raise ValueError("a stable privacy-identifier purpose is required")
        return tuple(
            self._identifier(version, secret, raw, purpose)
            for version, secret in ((self.primary_version, self.primary_secret), *self.previous_keys)
        )


_DEFAULT_KEYRING = PrivacyKeyring.from_environment()


def privacy_identifier(
    raw: str | None,
    *,
    purpose: str = "generic",
    keyring: PrivacyKeyring | None = None,
) -> str:
    """Return the active versioned HMAC identifier; never retain the raw value."""

    return (keyring or _DEFAULT_KEYRING).identifiers(raw, purpose=purpose)[0]


@dataclass(frozen=True)
class UsageState:
    token_count: int
    spend_usd: float
    soft_warning: bool
    provider_requests: int = 0
    spend_microusd: int = 0
    reserved_provider_requests: int = 0
    reserved_tokens: int = 0
    reserved_spend_microusd: int = 0
    active_concurrency: int = 0


def _to_milliseconds(now: float) -> int:
    try:
        value = Decimal(str(now))
    except InvalidOperation as exc:
        raise ValueError("now must be a finite non-negative timestamp") from exc
    if not value.is_finite() or value < 0:
        raise ValueError("now must be a finite non-negative timestamp")
    return int(value * 1_000)


class ResourceController:
    def __init__(
        self,
        *,
        backend: RateLimitBackend | None = None,
        keyring: PrivacyKeyring | None = None,
        lease_ttl_seconds: float | None = None,
        usage_ttl_seconds: float | None = None,
    ) -> None:
        self.backend: RateLimitBackend = backend or SharedInMemoryRateLimitBackend()
        self.keyring = keyring or _DEFAULT_KEYRING
        self.lease_ttl_seconds = (
            RESOURCE_RESERVATION_TTL_SECONDS
            if lease_ttl_seconds is None
            else max(0.001, float(lease_ttl_seconds))
        )
        self.usage_ttl_seconds = (
            None if usage_ttl_seconds is None else max(0.001, float(usage_ttl_seconds))
        )
        self._admission_stack: contextvars.ContextVar[tuple[AdmissionLease, ...]] = (
            contextvars.ContextVar(f"resource_admissions_{id(self)}", default=())
        )

    @staticmethod
    def _day(now: float) -> str:
        return datetime.fromtimestamp(now, tz=timezone.utc).date().isoformat()

    def _usage_expiry_ms(self, now: float) -> int:
        if self.usage_ttl_seconds is not None:
            return _to_milliseconds(now + self.usage_ttl_seconds)
        instant = datetime.fromtimestamp(now, tz=timezone.utc)
        next_midnight = datetime.combine(
            instant.date() + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc
        )
        # Keep yesterday's record for one additional day for reconciliation/idempotency.
        return int((next_midnight + timedelta(days=1)).timestamp() * 1_000)

    def _identifiers(self, raw: str | None, purpose: str) -> tuple[str, ...]:
        return self.keyring.identifiers(raw, purpose=purpose)

    @staticmethod
    def _daily_cost_limit() -> int | None:
        value = usd_to_microusd(DAILY_SPEND_LIMIT_USD)
        return value or None

    def _policy(self) -> AdmissionPolicy:
        return AdmissionPolicy(
            per_user_requests=PER_USER_REQUESTS_PER_MINUTE,
            per_network_requests=PER_IP_REQUESTS_PER_MINUTE,
            daily_provider_requests=DAILY_PROVIDER_REQUEST_QUOTA,
            daily_tokens=DAILY_TOKEN_QUOTA,
            daily_cost_microusd=self._daily_cost_limit(),
            max_concurrent_requests=MAX_CONCURRENT_REQUESTS,
            lease_ttl_ms=max(1, int(self.lease_ttl_seconds * 1_000)),
        )

    @staticmethod
    def _translate_backend_error(exc: Exception, *, now: float | None = None) -> ResourceLimitError:
        if isinstance(exc, BackendLimitExceeded):
            if exc.reason.startswith("daily_"):
                instant = datetime.fromtimestamp(time.time() if now is None else now, tz=timezone.utc)
                next_midnight = datetime.combine(
                    instant.date() + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc
                )
                retry_after = max(1, int((next_midnight - instant).total_seconds()))
            elif exc.reason == "concurrency_limit":
                retry_after = 1
            else:
                retry_after = 60
            return ResourceLimitError(exc.reason, retry_after_seconds=retry_after)
        if isinstance(exc, BackendTimeoutError):
            return ResourceLimitError("resource_backend_timeout")
        if isinstance(exc, BackendUnavailableError):
            return ResourceLimitError("resource_backend_unavailable")
        if isinstance(exc, BackendStateError):
            return ResourceLimitError("resource_backend_state_error")
        return ResourceLimitError("resource_backend_error")

    def begin(
        self,
        *,
        username: str | None,
        client_address: str | None,
        now: float | None = None,
        reservation: UsageReservation | None = None,
        provider_requests: int = 0,
        reserved_tokens: int = 0,
        reserved_cost_microusd: int = 0,
    ) -> AdmissionLease:
        instant = time.time() if now is None else now
        if reservation is not None and any(
            value for value in (provider_requests, reserved_tokens, reserved_cost_microusd)
        ):
            raise ValueError("pass either reservation or individual reservation values")
        requested = reservation or UsageReservation(
            provider_requests=provider_requests,
            tokens=reserved_tokens,
            cost_microusd=reserved_cost_microusd,
        )
        request = AdmissionRequest(
            user_keys=self._identifiers(f"user:{username or 'anonymous'}", "request-user"),
            network_keys=self._identifiers(
                f"network:{client_address or 'unknown'}", "request-network"
            ),
            day=self._day(instant),
            now_ms=_to_milliseconds(instant),
            usage_expires_at_ms=self._usage_expiry_ms(instant),
            policy=self._policy(),
            reservation=requested,
            admission_id=secrets.token_hex(16),
        )
        try:
            admission = self.backend.admit(request)
        except (BackendLimitExceeded, BackendTimeoutError, BackendUnavailableError, BackendStateError) as exc:
            raise self._translate_backend_error(exc, now=instant) from exc
        self._admission_stack.set((*self._admission_stack.get(), admission))
        return admission

    def check_login_attempt(
        self, *, username: str | None, client_address: str | None, now: float | None = None
    ) -> None:
        """Atomically bound guesses by both claimed account and coarse network identity."""

        instant = time.time() if now is None else now
        now_ms = _to_milliseconds(instant)
        windows = (
            (
                "login_account",
                self._identifiers(
                    f"login-account:{str(username or '').strip().casefold()}", "login-account"
                ),
                LOGIN_ATTEMPTS_PER_MINUTE,
            ),
            (
                "login_network",
                self._identifiers(
                    f"login-network:{client_address or 'unknown'}", "login-network"
                ),
                LOGIN_ATTEMPTS_PER_MINUTE,
            ),
        )
        try:
            self.backend.check_rate_limits(
                windows=windows,
                now_ms=now_ms,
                window_ms=60_000,
                event_id=secrets.token_hex(16),
            )
        except (BackendLimitExceeded, BackendTimeoutError, BackendUnavailableError, BackendStateError) as exc:
            raise self._translate_backend_error(exc, now=instant) from exc

    def check_operation(
        self,
        kind: str,
        *,
        identity: str | None,
        client_address: str | None,
        limit: int,
        now: float | None = None,
    ) -> None:
        """Apply an independent account and network window to a non-chat operation."""

        clean_kind = re.sub(r"[^a-z0-9_-]", "", str(kind).strip().casefold())
        if not clean_kind or isinstance(limit, bool) or int(limit) < 1:
            raise ValueError("operation rate limits require a stable kind and positive limit")
        instant = time.time() if now is None else now
        windows = (
            (
                f"{clean_kind}_account",
                self._identifiers(
                    f"{clean_kind}-account:{identity or 'anonymous'}",
                    f"{clean_kind}-account",
                ),
                int(limit),
            ),
            (
                f"{clean_kind}_network",
                self._identifiers(
                    f"{clean_kind}-network:{client_address or 'unknown'}",
                    f"{clean_kind}-network",
                ),
                int(limit),
            ),
        )
        try:
            self.backend.check_rate_limits(
                windows=windows,
                now_ms=_to_milliseconds(instant),
                window_ms=60_000,
                event_id=secrets.token_hex(16),
            )
        except (BackendLimitExceeded, BackendTimeoutError, BackendUnavailableError, BackendStateError) as exc:
            raise self._translate_backend_error(exc, now=instant) from exc

    def _take_admission(self, admission: AdmissionLease | str | None) -> AdmissionLease | None:
        stack = list(self._admission_stack.get())
        if admission is None:
            selected = stack.pop() if stack else None
        else:
            admission_id = admission.admission_id if isinstance(admission, AdmissionLease) else admission
            selected = next(
                (item for item in reversed(stack) if item.admission_id == admission_id),
                admission if isinstance(admission, AdmissionLease) else None,
            )
            stack = [item for item in stack if item.admission_id != admission_id]
        self._admission_stack.set(tuple(stack))
        return selected

    @staticmethod
    def _usage_state(snapshot: UsageSnapshot) -> UsageState:
        cost_limit = usd_to_microusd(DAILY_SPEND_LIMIT_USD)
        warning = bool(
            snapshot.tokens + snapshot.reserved_tokens
            >= DAILY_TOKEN_QUOTA * SOFT_BUDGET_FRACTION
            or (
                cost_limit
                and snapshot.cost_microusd + snapshot.reserved_cost_microusd
                >= cost_limit * SOFT_BUDGET_FRACTION
            )
        )
        return UsageState(
            token_count=snapshot.tokens,
            spend_usd=microusd_to_usd(snapshot.cost_microusd),
            soft_warning=warning,
            provider_requests=snapshot.provider_requests,
            spend_microusd=snapshot.cost_microusd,
            reserved_provider_requests=snapshot.reserved_provider_requests,
            reserved_tokens=snapshot.reserved_tokens,
            reserved_spend_microusd=snapshot.reserved_cost_microusd,
            active_concurrency=snapshot.active_concurrency,
        )

    def finish(
        self,
        *,
        username: str | None,
        total_tokens: int | None = None,
        cost_usd: float | str | Decimal | None = None,
        now: float | None = None,
        admission: AdmissionLease | str | None = None,
        provider_requests: int | None = None,
        cost_microusd: int | None = None,
    ) -> UsageState:
        instant = time.time() if now is None else now
        if cost_usd is not None and cost_microusd is not None:
            raise ValueError("pass either cost_usd or cost_microusd")
        if total_tokens is not None and (
            isinstance(total_tokens, bool) or not isinstance(total_tokens, int) or total_tokens < 0
        ):
            raise ValueError("total_tokens must be a non-negative integer")
        if cost_microusd is not None and (
            isinstance(cost_microusd, bool)
            or not isinstance(cost_microusd, int)
            or cost_microusd < 0
        ):
            raise ValueError("cost_microusd must be a non-negative integer")
        selected = self._take_admission(admission)
        if admission is not None and selected is None:
            raise ResourceLimitError("resource_admission_missing")
        reserve = selected.reservation if selected else UsageReservation()
        actual = UsageReservation(
            provider_requests=(
                reserve.provider_requests if provider_requests is None else provider_requests
            ),
            tokens=reserve.tokens if total_tokens is None else total_tokens,
            cost_microusd=(
                reserve.cost_microusd
                if cost_usd is None and cost_microusd is None
                else (
                    usd_to_microusd(cost_usd)
                    if cost_microusd is None
                    else cost_microusd
                )
            ),
        )
        user_keys = (
            selected.user_keys
            if selected
            else self._identifiers(f"user:{username or 'anonymous'}", "request-user")
        )
        day = selected.day if selected else self._day(instant)
        try:
            snapshot = self.backend.reconcile(
                ReconciliationRequest(
                    admission_id=selected.admission_id if selected else None,
                    user_keys=user_keys,
                    day=day,
                    now_ms=_to_milliseconds(instant),
                    usage_expires_at_ms=self._usage_expiry_ms(instant),
                    actual=actual,
                )
            )
        except (BackendLimitExceeded, BackendTimeoutError, BackendUnavailableError, BackendStateError) as exc:
            raise self._translate_backend_error(exc) from exc
        return self._usage_state(snapshot)

    def usage(self, *, username: str | None, now: float | None = None) -> UsageState:
        instant = time.time() if now is None else now
        keys = self._identifiers(f"user:{username or 'anonymous'}", "request-user")
        try:
            snapshot = self.backend.usage(
                user_keys=keys,
                day=self._day(instant),
                now_ms=_to_milliseconds(instant),
            )
        except (BackendLimitExceeded, BackendTimeoutError, BackendUnavailableError, BackendStateError) as exc:
            raise self._translate_backend_error(exc) from exc
        return self._usage_state(snapshot)

    def cleanup(self, *, now: float | None = None) -> int:
        instant = time.time() if now is None else now
        try:
            return self.backend.cleanup(now_ms=_to_milliseconds(instant))
        except (BackendLimitExceeded, BackendTimeoutError, BackendUnavailableError, BackendStateError) as exc:
            raise self._translate_backend_error(exc) from exc

    def reset_for_tests(self) -> None:
        try:
            self.backend.reset()
        except (BackendTimeoutError, BackendUnavailableError, BackendStateError) as exc:
            raise self._translate_backend_error(exc) from exc
        self._admission_stack.set(())


controller = ResourceController(
    backend=build_rate_limit_backend(
        os.getenv("RATE_LIMIT_BACKEND", "memory"),
        redis_url=os.getenv("RATE_LIMIT_REDIS_URL", ""),
        redis_namespace=os.getenv("RATE_LIMIT_REDIS_NAMESPACE", "advisu:rate-limit:v1"),
        redis_timeout_seconds=_positive_float("RATE_LIMIT_REDIS_TIMEOUT_SECONDS", 0.25),
    ),
    keyring=PrivacyKeyring.from_environment(),
)
