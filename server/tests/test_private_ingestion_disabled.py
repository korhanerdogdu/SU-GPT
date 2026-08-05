from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.load_vectorstore import (
    canonical_document_type,
    ingest_file_paths,
    ingest_text_source,
    validate_public_ingest_path,
    validate_public_ingest_source,
)
from modules.source_indexer import _iter_source_files
from scripts import bulk_ingest


@pytest.mark.parametrize(
    "kind",
    [
        "review",
        "whatsapp",
        "telegram",
        "discord",
        "private_chat",
        "private-chat",
        "instructor_review",
        "instructor-review",
    ],
)
def test_private_or_person_review_document_types_fail_closed(kind):
    with pytest.raises(ValueError, match="permanently disabled"):
        canonical_document_type(kind)


def test_private_review_cannot_reach_file_or_text_ingestion(tmp_path):
    path = tmp_path / "private.txt"
    path.write_text("synthetic private content", encoding="utf-8")
    with pytest.raises(ValueError, match="permanently disabled"):
        ingest_file_paths([str(path)], document_type="review")
    with pytest.raises(ValueError, match="permanently disabled"):
        ingest_text_source(
            text="synthetic private content",
            source_name="private.txt",
            document_type="whatsapp",
        )


def test_bulk_ingest_has_no_private_review_target(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["bulk_ingest.py", "--target", "reviews"])
    with pytest.raises(SystemExit):
        bulk_ingest.parse_args()

    monkeypatch.setattr(sys, "argv", ["bulk_ingest.py"])
    args = bulk_ingest.parse_args()
    assert args.target == "all"
    assert not hasattr(args, "reviews_dir")


@pytest.mark.parametrize(
    "directory",
    [
        "reviews",
        "whatsapp",
        "private_chats",
        "private-chats",
        "instructor_reviews",
        "telegram-export",
        "discord",
    ],
)
def test_bulk_course_ingest_rejects_private_directory_variants(tmp_path, directory):
    private_dir = tmp_path / directory
    private_dir.mkdir()
    (private_dir / "export.txt").write_text("synthetic private content", encoding="utf-8")
    with pytest.raises(ValueError, match="permanently disabled"):
        bulk_ingest.iter_supported_files(str(tmp_path), exclude_dirs=set())


def test_ingest_path_policy_rejects_root_escape_symlink(tmp_path):
    source_root = tmp_path / "public"
    source_root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("synthetic outside content", encoding="utf-8")
    link = source_root / "course.txt"
    link.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        validate_public_ingest_path(link, allowed_root=source_root)


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_name", "telegram-export.txt"),
        ("source_name", "discord_history.md"),
        ("storage_key", "private/chats/export.txt"),
        ("storage_key", "instructor-reviews/export.txt"),
        ("storage_key", "abc123-whatsapp.txt"),
        ("storage_key", "abc123-instructor-review.txt"),
    ],
)
def test_shared_ingest_source_policy_rejects_private_labels(field, value):
    kwargs = {"document_type": "course", "source_name": "catalog.txt", "storage_key": ""}
    kwargs[field] = value
    with pytest.raises(ValueError, match="permanently disabled"):
        validate_public_ingest_source(**kwargs)


def test_text_ingest_cannot_relabel_private_source_as_course():
    with pytest.raises(ValueError, match="permanently disabled"):
        ingest_text_source(
            text="synthetic private content",
            source_name="telegram-export.txt",
            storage_key="private_chat/export.txt",
            document_type="course",
        )


def test_auto_source_indexer_uses_shared_fail_closed_path_policy(tmp_path):
    private_dir = tmp_path / "discord-export"
    private_dir.mkdir()
    (private_dir / "history.txt").write_text("synthetic private content", encoding="utf-8")
    assert list(_iter_source_files(tmp_path)) == []
