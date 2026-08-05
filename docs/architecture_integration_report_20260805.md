# Architecture integration report — 2026-08-05

## Objective

Two divergent implementations were evaluated under the same protocol and reduced to one runtime
architecture. Selection was based on measured behavior, security properties, reproducibility, and
consistency rather than implementation origin.

The resulting implementation is bound to commit:

`e66b45f`

## Same-protocol comparison

The input comparison used 42 frozen prompts: 22 attacks and 20 benign cases. Both individual
methods and their ordered union ran over the exact same rows.

| Method | Recall | Precision | F1 | False-refusal rate |
|---|---:|---:|---:|---:|
| Prompt/data guardrails alone | 77.27% | 100% | 0.8718 | 0% |
| Category-specific content classifier alone | 22.73% | 100% | 0.3704 | 0% |
| Selected ordered union | **100%** | **100%** | **1.0000** | **0%** |

The methods are complementary. The content classifier covers self-harm, directed violence, sexual
harassment, and profanity cases that the extraction-oriented guardrails did not cover. The broader
guardrails cover prompt injection, harmful content generation, secret extraction, encoded attacks,
retrieval poisoning, citation fabrication, hidden reasoning, and output leakage.

## Selected architecture

| Concern | Selected implementation | Reason |
|---|---|---|
| Crisis and abuse | Category-specific deterministic classifier before all later processing | Appropriate crisis response and additional measured attack recall |
| Prompt/data attacks | Independent input, retrieval, and output guardrails | Broader surfaces and most of the frozen attack coverage |
| Overall safety | Ordered union of both methods | 100% recall and no new false refusal on the frozen set |
| Daily allowance UI | Visible remaining-count meter | Makes resource availability predictable |
| Daily allowance backend | HMAC-keyed atomic resource controller | Avoids plaintext identities and supports shared Redis state |
| Question accounting | Actual provider calls, not every deterministic request | Local verified operations and refused requests should not consume model quota |
| Intent identifiers | Canonical English identifiers after one translation boundary | Clear internal vocabulary without invalidating trained artifacts |
| Frozen provider prompt | Historical label conversion only at prompt boundary | Preserves preregistered prompt hash and comparison validity |
| Frontend localization | One typed locale context and resource catalogue | Prevents two conflicting interface-language sources |
| Login experience | Branded split layout integrated with existing auth/theme/locale contracts | Better visual hierarchy without duplicating application state |
| Provider layer | Provider-neutral reliable adapter | Bounded retries, deadlines, exact-model checks, sanitized errors and reversible experiments |
| Academic decisions | Deterministic engines plus confidence abstention | Prevents model-authored eligibility, credit, prerequisite, and schedule decisions |

## Parallel implementations removed

The following concepts were not retained as separate runtime systems:

- A second frontend language context and translation dictionary
- A plaintext username-based Mongo question counter
- A quota path that fails open when shared state is unavailable
- A duplicate server message catalogue
- Private-chat and named instructor-review ingestion

Useful product behavior from those implementations was moved into the selected abstractions instead
of keeping two sources of truth.

## Additional integration repairs

1. Public ingestion validates the original filename before adding its content digest. Embedded
   private markers such as `abc123-whatsapp.txt` are also rejected.
2. Provider activation is enforced inside settings construction as well as ASGI startup, covering
   workers and evaluation scripts.
3. English questions containing Turkish proper names no longer switch the answer to Turkish.
4. The usage endpoint requires a bearer token and same-user/administrator authorization.
5. The daily default is 15 actual provider-backed requests; deterministic and refused paths
   reconcile to zero.

## Final verification

| Verification | Result |
|---|---:|
| Python tests | 333 passed |
| Frontend production build | passed |
| Deterministic security v2 | 52/52 |
| Layered input safety | 42/42 decisions correct |
| Language routing | TR 100/100, EN 100/100, mixed 20/20 |
| Bounded real `/ask/` endpoint security | 12/12 |

Endpoint evidence is stored in:

`outputs/security_endpoint/unified-v1-20260805/`

Its manifest binds the runner, checksummed dataset, configuration, implementation commit, and the
SHA-256 of an empty working-tree diff. Raw prompts, raw responses, secrets, and reasoning content
are excluded.

## Limits

- The endpoint run uses synthetic in-memory provider responses and is not a comprehensive
  penetration test.
- Test count is a coverage indicator, not a quality metric by itself.
- Live-provider quality claims remain valid only for the exact preregistered split and settings.
- AWS deployment remains deferred and was not changed.
