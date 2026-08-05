# Course Review Data Policy

## Purpose and default state

The course-review feature measures courses, not people. It is disabled by default and must be
explicitly enabled by the application after the product owner has approved the notice, retention
period, moderation staffing, and HMAC-key management. It is not connected to WhatsApp exports,
private student groups, instructor-review records, or historical private messages.

## Persistent v1 data accepted

Each submission identifies one course and contains only five 1–5 integer ratings:

- difficulty
- workload
- learning value
- organization
- overall satisfaction

An optional course-focused comment may be submitted. Persistent v1 does **not** accept or publish
subgroups; this prevents complementary-group differencing and avoids collecting unnecessary
demographic attributes. Instructor/professor rating fields, instructor names, author names,
usernames, student IDs, e-mail addresses, and telephone numbers are not accepted. Explicit consent
is mandatory. The server requires the current policy version rather than trusting an arbitrary
client version, and stores that version together with a UTC consent timestamp.

## Duplicate prevention without stored identity

The caller supplies an authenticated author token only for the duration of `submit`. The store
computes a course-scoped HMAC-SHA-256 value over the normalized token using a secret held outside
source control and a versioned namespace (`course-review-author-v1`). The raw token is never stored.

The digest is stored only as a private field in the persistence envelope of the same course-review
document. It is never present in the public/domain serialization, moderation history, aggregate,
log, or API response. Keeping the digest and ratings in one document allows MongoDB's unique
`(courseCode, authorDigest)` index to make the insert atomic without requiring a multi-document
transaction. Application code must not perform a check-then-insert duplicate query.

Production keys must contain at least 32 random bytes. Key rotation requires a documented digest
migration that preserves the one-author-per-course guarantee; merely changing the secret would
permit duplicates under a new digest namespace.

The digest is pseudonymous security data, not anonymous analytics data. Access to the collection
and its indexes must therefore remain restricted. Consent withdrawal uses a hard delete addressed
by the authenticated user's recomputed digest and course code. The complete document—including
digest, sanitized comment, and embedded moderation history—is deleted. No raw author identifier or
pseudonymous tombstone is retained by this component.

## Text privacy and moderation

Comments are normalized and checked before storage. The deterministic filter detects and redacts:

- e-mail addresses, telephone numbers, and labelled student IDs;
- names supplied from an authorized blocked-name/instructor roster;
- instructor-role terms such as “professor”, “instructor”, and “hoca”;
- conservative likely-full-name patterns.

Content matching those controls is stored only in redacted form and rejected. The original text is
not retained by this service. A comment with no detected sensitive content is still `pending` until
a human moderator approves it. Ratings-only submissions may be approved automatically because they
contain no free text. Privacy-rejected content cannot later be approved.

Name detection is intentionally conservative and imperfect. The authorized name roster must be
kept current, and human moderators must review all comments before publication or aggregation.
Moderators must reject overlooked personal or instructor-targeted content rather than copying it to
notes. Free-text moderator notes are prohibited. The store accepts only fixed reason codes and uses
a compare-and-set transition from `pending` to one terminal state. Its embedded history contains
only review ID, course code where applicable, prior/new state, reason codes, and timestamp; it
contains no author token, digest, comment, or subgroup.

## Aggregation and suppression

Only approved reviews contribute to statistics. The database aggregation pipeline applies the
course/state filter, computes the aggregate, and removes results below the threshold before rows
reach application code. No average or exact small count is returned until at least 10 eligible
reviews exist for the course. Persistent v1 offers no subgroup input or subgroup aggregate.

Published aggregates contain only the course code, approved-review count after the threshold, and
the five rounded averages. A deletion that drops the eligible population below 10 immediately
suppresses the aggregate again. Results must not be presented as official university judgments or
as claims about an instructor.

## Dedicated storage and startup contract

Persistent reviews require a new, course-only collection. The legacy instructor-review collection,
private chat data, WhatsApp exports, retrieval corpus, and vector index must never be read, joined,
copied, or written by this feature. Required indexes are:

- unique `reviewId`;
- unique `(courseCode, authorDigest)`;
- `(courseCode, moderationState)` for approved-only aggregation.

The store exposes `ensure_indexes()`. When the feature flag is enabled, the application startup
owner must validate the persistence policy and HMAC secret, call this method, and abort startup if
any required index cannot be created or verified. Database exceptions are converted to safe typed
errors; connection strings, credentials, queries, digests, and backend details must not be returned
to clients.

## Retention, access, and deletion before production

The persistence boundary now supplies the atomic storage primitives, but it does not enable the
feature or decide deployment policy. Before traffic is enabled, the product owner must define:

1. review and redacted-comment retention periods;
2. access roles for pending comments and the private digest field;
3. authenticated deletion and consent-withdrawal routing;
4. HMAC-secret storage, backup, and rotation/migration;
5. moderator staffing, training, and escalation;
6. a user-facing privacy notice and legal basis;
7. rate limits and abuse-response procedures;
8. startup wiring that verifies the dedicated collection and indexes.

The embedded moderation history is deleted with the review. If policy later requires an immutable
audit trail after deletion, it must be designed separately with an approved retention/legal basis
and transactional guarantees; it must not silently retain a user-linked digest. Until all decisions
above are implemented and tested, the feature must remain disabled.
