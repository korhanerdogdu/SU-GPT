from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main as main_module
from modules.auth import issue_token
from modules.transcript_parser import ParsedTranscript, TranscriptCourse


def _headers(username: str, role: str = "student") -> dict[str, str]:
    token, _ = issue_token(username, role)
    return {"Authorization": f"Bearer {token}"}


def _fake_parsed() -> ParsedTranscript:
    return ParsedTranscript(
        attempts=[],
        courses=[
            TranscriptCourse(
                course_code="CS 201", title="Introduction to Computing", grade="A-",
                su_credits=3.0, ects=6.0, term="Fall 2022-2023", counts_in_gpa=True,
            ),
        ],
        gpa=3.7,
        total_su_credits=3.0,
        total_ects=6.0,
        warnings=[],
    )


def test_transcript_upload_rejects_non_pdf_content_type():
    client = TestClient(main_module.app)
    response = client.post(
        "/users/student/transcript",
        headers=_headers("student"),
        files={"file": ("notes.txt", b"not a pdf", "text/plain")},
    )
    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


def test_transcript_upload_rejects_oversized_file():
    client = TestClient(main_module.app)
    oversized = b"%PDF-1.4\n" + b"0" * (15 * 1024 * 1024 + 1)
    response = client.post(
        "/users/student/transcript",
        headers=_headers("student"),
        files={"file": ("big.pdf", oversized, "application/pdf")},
    )
    assert response.status_code == 400
    assert "too large" in response.json()["detail"].lower()


def test_transcript_upload_rejects_unparseable_pdf():
    client = TestClient(main_module.app)
    response = client.post(
        "/users/student/transcript",
        headers=_headers("student"),
        files={"file": ("garbage.pdf", b"not actually a pdf file", "application/pdf")},
    )
    assert response.status_code in (422,)


def test_transcript_upload_success_applies_courses_and_returns_summary(monkeypatch):
    parsed = _fake_parsed()
    monkeypatch.setattr(main_module.transcript_parser, "parse_transcript", lambda data: parsed)

    saved_calls = []

    async def fake_save_transcript(username, *, filename, content_type, pdf_bytes, parsed):
        saved_calls.append((username, filename, content_type))
        return {
            "id": "t1", "filename": filename, "uploaded_at": "2026-08-06T00:00:00Z",
            "course_count": len(parsed.courses), "gpa": parsed.gpa,
            "total_su_credits": parsed.total_su_credits, "total_ects": parsed.total_ects,
            "warnings": parsed.warnings,
        }

    applied_calls = []

    async def fake_apply_transcript_courses(username, courses):
        applied_calls.append((username, [c.course_code for c in courses]))
        return {"matched": ["CS 201"], "unmatched": [], "courses": [{"code": "CS 201", "grade": "A-"}]}

    monkeypatch.setattr(main_module, "save_transcript", fake_save_transcript)
    monkeypatch.setattr(main_module, "apply_transcript_courses", fake_apply_transcript_courses)

    client = TestClient(main_module.app)
    response = client.post(
        "/users/student/transcript",
        headers=_headers("student"),
        files={"file": ("transcript.pdf", b"%PDF-1.4\nfake but content-type is right", "application/pdf")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["gpa"] == 3.7
    assert body["matched_course_codes"] == ["CS 201"]
    assert body["unmatched_course_codes"] == []
    assert saved_calls == [("student", "transcript.pdf", "application/pdf")]
    assert applied_calls == [("student", ["CS 201"])]


def test_transcript_upload_updates_profile_major_from_official_program_line(monkeypatch):
    # Regression: an internal-transfer transcript's LATEST "Program :" line is the student's
    # current, authoritative major -- upload must update a stale/unset profile.major to match,
    # not leave the student's advising scoped to a program they already transferred out of.
    parsed = _fake_parsed()
    parsed.program_hint = "Computer Science and Engineering (FENS)"
    monkeypatch.setattr(main_module.transcript_parser, "parse_transcript", lambda data: parsed)

    async def fake_save_transcript(username, *, filename, content_type, pdf_bytes, parsed):
        return {
            "id": "t1", "filename": filename, "uploaded_at": "2026-08-06T00:00:00Z",
            "course_count": len(parsed.courses), "gpa": parsed.gpa,
            "total_su_credits": parsed.total_su_credits, "total_ects": parsed.total_ects,
            "warnings": parsed.warnings,
        }

    async def fake_apply_transcript_courses(username, courses):
        return {"matched": ["CS 201"], "unmatched": [], "courses": [{"code": "CS 201", "grade": "A-"}]}

    async def fake_get_academic_profile(username):
        return {"major": "MAN", "degree_code": None, "admission_term": None,
                "curriculum_term": None, "academic_year": None, "minor_codes": [],
                "profile_status": "confirmed"}

    set_calls = []

    async def fake_set_academic_profile(username, profile):
        set_calls.append((username, profile))
        return {**await fake_get_academic_profile(username), **profile}

    monkeypatch.setattr(main_module, "save_transcript", fake_save_transcript)
    monkeypatch.setattr(main_module, "apply_transcript_courses", fake_apply_transcript_courses)
    monkeypatch.setattr(main_module, "get_academic_profile", fake_get_academic_profile)
    monkeypatch.setattr(main_module, "set_academic_profile", fake_set_academic_profile)

    client = TestClient(main_module.app)
    response = client.post(
        "/users/student/transcript",
        headers=_headers("student"),
        files={"file": ("transcript.pdf", b"%PDF-1.4\nfake but content-type is right", "application/pdf")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["profile_major_update"] == {"from": "MAN", "to": "CS"}
    assert set_calls == [("student", {"major": "CS"})]


def test_transcript_upload_leaves_profile_untouched_when_program_hint_unresolved(monkeypatch):
    # Fail-closed: an unrecognised or missing "Program :" line must never guess a major.
    parsed = _fake_parsed()
    parsed.program_hint = "Some Unknown Program (XYZ)"
    monkeypatch.setattr(main_module.transcript_parser, "parse_transcript", lambda data: parsed)

    async def fake_save_transcript(username, *, filename, content_type, pdf_bytes, parsed):
        return {
            "id": "t1", "filename": filename, "uploaded_at": "2026-08-06T00:00:00Z",
            "course_count": len(parsed.courses), "gpa": parsed.gpa,
            "total_su_credits": parsed.total_su_credits, "total_ects": parsed.total_ects,
            "warnings": parsed.warnings,
        }

    async def fake_apply_transcript_courses(username, courses):
        return {"matched": ["CS 201"], "unmatched": [], "courses": [{"code": "CS 201", "grade": "A-"}]}

    def boom(*_args, **_kwargs):
        raise AssertionError("profile must not be touched when the program hint is unresolved")

    monkeypatch.setattr(main_module, "save_transcript", fake_save_transcript)
    monkeypatch.setattr(main_module, "apply_transcript_courses", fake_apply_transcript_courses)
    monkeypatch.setattr(main_module, "get_academic_profile", boom)
    monkeypatch.setattr(main_module, "set_academic_profile", boom)

    client = TestClient(main_module.app)
    response = client.post(
        "/users/student/transcript",
        headers=_headers("student"),
        files={"file": ("transcript.pdf", b"%PDF-1.4\nfake but content-type is right", "application/pdf")},
    )
    assert response.status_code == 200
    assert response.json()["profile_major_update"] is None


def test_transcript_upload_requires_matching_username_authorization():
    client = TestClient(main_module.app)
    response = client.post(
        "/users/other-student/transcript",
        headers=_headers("student"),  # token for "student", path says "other-student"
        files={"file": ("transcript.pdf", b"%PDF-1.4\n", "application/pdf")},
    )
    assert response.status_code in (401, 403)
