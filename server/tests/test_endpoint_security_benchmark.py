from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "server"))

from evaluation.endpoint_security_benchmark import (  # noqa: E402
    DEFAULT_MANIFEST,
    load_cases,
    write_run,
)


def test_endpoint_cases_are_loaded_from_checksummed_bounded_manifest():
    manifest, cases, dataset = load_cases(DEFAULT_MANIFEST)
    assert manifest["dataset_role"] == "synthetic_development_endpoint_subset"
    assert len(cases) == 12
    assert len({case.item_id for case in cases}) == 12
    assert hashlib.sha256(dataset.read_bytes()).hexdigest() == manifest["content_sha256"]
    assert "independent final" in str(manifest["scope_note"])


def test_endpoint_run_manifest_binds_dataset_runner_config_and_redacted_files(tmp_path):
    metadata, _cases, dataset = load_cases(DEFAULT_MANIFEST)
    result = {
        "scope": "synthetic test",
        "total": 1,
        "passed": 1,
        "attack_success_rate": 0.0,
        "safe_refusal_accuracy": 1.0,
        "false_refusal_rate": 0.0,
        "critical_bypass_count": 0,
        "language": {"tr": {"passed": 0, "total": 0}, "en": {"passed": 1, "total": 1}},
        "rows": [{
            "item_id": "synthetic-id",
            "cluster_id": "synthetic-cluster",
            "language": "en",
            "attack_category": "synthetic",
            "layer": "input",
            "expected_action": "allow",
            "passed": True,
            "http_status": 200,
            "response_digest": "0" * 64,
        }],
    }
    run_dir = write_run(
        result,
        output_root=tmp_path,
        run_id="bounded-test",
        source_manifest=DEFAULT_MANIFEST,
        source_dataset=dataset,
        source_metadata=metadata,
    )
    run_manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert run_manifest["source_dataset_sha256"] == metadata["content_sha256"]
    assert len(run_manifest["runner_sha256"]) == 64
    assert len(run_manifest["configuration_sha256"]) == 64
    checksums = json.loads((run_dir / "artifact_checksums.json").read_text())
    assert set(checksums) == {"rows.redacted.jsonl", "summary.json", "run_manifest.json"}
    serialized = "".join(path.read_text() for path in run_dir.iterdir() if path.is_file())
    assert "synthetic prompt" not in serialized
    assert "raw response" not in serialized
