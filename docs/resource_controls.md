# Shared rate limits and provider quotas

## Runtime selection

`RATE_LIMIT_BACKEND=memory` is the local-development default. Its state is thread-safe and may be
shared by controllers inside one process, but it is not a multi-process production control.
Production configuration validation requires `RATE_LIMIT_BACKEND=redis` and a non-empty
`RATE_LIMIT_REDIS_URL`. Selecting Redis never falls back to memory: timeout, unavailable, and
malformed-state failures fail closed with a sanitized 503 response.

The current multi-key layout targets standalone Redis (or an explicitly configured single hash
slot). It is not declared Redis Cluster compatible: cluster mode can return `CROSSSLOT` until all
script keys are migrated to one reviewed hash tag.

The Redis adapter uses one Lua operation for each admission/window update and one Lua operation
for usage reconciliation. Increment, expiry, concurrency lease, and quota reservation therefore
do not have an increment/expire race. No Redis infrastructure is deployed by this repository.

## Privacy and rotation

Backend keys contain purpose-separated HMAC identifiers, never plaintext usernames, account IDs,
emails, or network addresses. `ABUSE_HASH_SECRET_VERSION` is part of each identifier. During a
coordinated rotation, `ABUSE_HASH_PREVIOUS_SECRETS=old_version=old_secret` lets new workers inspect
old windows while writing under the new primary version. Workers must be updated together; remove
the previous key only after every old window, lease, and daily quota record has expired.

Rate windows expire after 60 seconds. Concurrency reservations have a bounded lease (default 120
seconds). Daily usage expires after the following UTC day, allowing late idempotent reconciliation.
Settled admission IDs are retained temporarily so retries cannot double charge usage.

## Protected operations

Account and network windows are independent for login, chat, streaming, uploads, conversation
creation, and course-review submission/deletion/aggregation/moderation. File count, per-file
bytes, and total request bytes remain enforced at the upload validation boundary. Signup is
disabled and has no backend route.

Every login attempt is counted before credential verification. Consequently both failed and
successful attempts consume the account and network window; a successful login does not erase
failed-attempt history. This prevents an attacker from resetting the counter with one known
credential.

## Provider budget accounting

Chat admission reserves one provider request and a conservative token upper bound atomically.
Deterministic/refused requests reconcile to zero provider usage. Provider attempts reconcile to
reported tokens/cost when available; errors retain their conservative token reservation. Costs
cross the backend boundary only as integer microdollars, never floating-point totals.

`DAILY_SPEND_LIMIT_USD=0` means no owner-approved spend limit exists and the cost gate is disabled.
This is an explicit external blocker, not a passing budget gate. Once an owner sets a positive
limit, production configuration also requires a positive, conservatively sized
`PROVIDER_REQUEST_COST_RESERVATION_MICROUSD`; admission reserves it before the call and
reconciliation replaces it with actual cost. Missing reservation configuration fails startup.

429 responses include `Retry-After`: 60 seconds for fixed windows, one second for concurrency,
and the remaining seconds until the next UTC day for daily quotas. Backend-health failures use a
sanitized 503 and reveal no backend type, topology, namespace, or key.
