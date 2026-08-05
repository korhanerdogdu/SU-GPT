from __future__ import annotations

"""
Intent -> retrieval policy mapping (roadmap section 21.1).

Turns the detected intent + the student's academic profile into (a) a Chroma metadata
filter that scopes retrieval to the correct data_role / program / curriculum_term BEFORE
ranking, and (b) a missing-profile-field / missing-data check so an *authoritative*
graduation audit is only attempted when the exact official requirement file exists.

Kept free of MongoDB imports so it can be unit-tested in isolation.
"""

from dataclasses import dataclass, field

from modules import curriculum_registry, intents


@dataclass(frozen=True)
class RetrievalPolicy:
    intent: str
    data_roles: tuple[str, ...]              # allowed data_role metadata values
    required_profile_fields: tuple[str, ...] = ()   # profile keys needed for an authoritative answer
    scope_program: bool = False              # constrain retrieval to profile.major
    scope_curriculum_term: bool = False      # constrain retrieval to profile.curriculum_term
    authoritative: bool = False              # needs the exact official file (else safe limitation)


_POLICIES: dict[str, RetrievalPolicy] = {
    intents.GRADUATION_STATUS: RetrievalPolicy(
        intents.GRADUATION_STATUS, ("curriculum_requirement",),
        required_profile_fields=("major", "curriculum_term"),
        scope_program=True, scope_curriculum_term=True, authoritative=True,
    ),
    intents.COURSE_RECOMMENDATION: RetrievalPolicy(
        intents.COURSE_RECOMMENDATION, ("curriculum_requirement",),
        required_profile_fields=("major", "curriculum_term"),
        scope_program=True, scope_curriculum_term=True,
    ),
    intents.COURSE_DETAIL: RetrievalPolicy(intents.COURSE_DETAIL, ("curriculum_requirement",)),
    intents.STUDY_PLAN: RetrievalPolicy(intents.STUDY_PLAN, ("curriculum_requirement",)),
    intents.MAJOR_SELECTION: RetrievalPolicy(intents.MAJOR_SELECTION, ("curriculum_requirement",)),
    intents.SPECIALIZATION: RetrievalPolicy(intents.SPECIALIZATION, ("curriculum_requirement",)),
    intents.MINOR: RetrievalPolicy(intents.MINOR, ("minor_requirement",)),
}


def policy_for(intent: str) -> RetrievalPolicy | None:
    return _POLICIES.get(intent)


def _and(clauses: list[dict]) -> dict | None:
    clauses = [c for c in clauses if c]
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def build_metadata_filter(intent: str, profile: dict | None) -> dict | None:
    """Chroma `where` scoping retrieval by data_role (+ program/curriculum_term when known)."""
    policy = policy_for(intent)
    if not policy:
        return None
    profile = profile or {}
    clauses: list[dict] = []
    if len(policy.data_roles) == 1:
        clauses.append({"data_role": policy.data_roles[0]})
    else:
        clauses.append({"data_role": {"$in": list(policy.data_roles)}})
    if policy.scope_program and profile.get("major"):
        clauses.append({"program": str(profile["major"]).strip().upper()})
    if policy.scope_curriculum_term and profile.get("curriculum_term"):
        clauses.append({"curriculum_term": str(profile["curriculum_term"]).strip()})
    return _and(clauses)


@dataclass(frozen=True)
class ProfileGate:
    ok: bool
    missing_fields: tuple[str, ...] = ()
    data_unavailable: bool = False
    message: str = ""


def check_profile(intent: str, profile: dict | None) -> ProfileGate:
    """For authoritative intents, verify required profile fields + that the exact file exists."""
    policy = policy_for(intent)
    if not policy or not policy.authoritative:
        return ProfileGate(ok=True)
    profile = profile or {}
    missing = tuple(f for f in policy.required_profile_fields if not profile.get(f))
    if missing:
        return ProfileGate(
            ok=False, missing_fields=missing,
            message=(
                "To give you a reliable graduation audit I need your academic profile first. "
                "Please set your major and curriculum term (missing: "
                + ", ".join(missing) + ")."
            ),
        )
    program = str(profile.get("major")).strip().upper()
    term = str(profile.get("curriculum_term")).strip()
    if not curriculum_registry.has_major_curriculum(program, term):
        return ProfileGate(
            ok=False, data_unavailable=True,
            message=(
                f"Official degree requirement data for {program} curriculum {term} is not "
                "available in the system. I can give general course and program information, "
                "but I cannot produce a reliable graduation audit for that exact curriculum."
            ),
        )
    return ProfileGate(ok=True)
