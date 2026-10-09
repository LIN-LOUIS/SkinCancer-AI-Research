from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.run_protocol_p2 import P2Orchestrator, build_parser, safe_error_code


REPO = Path(__file__).resolve().parents[1]


def test_cli_help_lists_resumable_stages() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/run_protocol_p2.py", "--help"],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0
    for stage in ("prepare", "gpu-preflight", "precheck", "train", "evaluate", "publish", "all"):
        assert stage in completed.stdout
    assert "--resume" in completed.stdout


def test_cli_rejects_any_seed_other_than_42() -> None:
    parser = build_parser()
    arguments = parser.parse_args(["prepare", "--seed", "43"])
    with pytest.raises(ValueError, match="SEED_NOT_FROZEN"):
        P2Orchestrator.from_arguments(arguments)


def test_stage_order_is_frozen() -> None:
    assert P2Orchestrator.STAGES == (
        "prepare", "gpu-preflight", "precheck", "train", "evaluate", "publish"
    )


def test_evaluate_refuses_missing_two_locks_without_opening_data(tmp_path: Path) -> None:
    orchestrator = object.__new__(P2Orchestrator)
    orchestrator.private_root = tmp_path
    orchestrator.run_root = tmp_path / "runs"
    orchestrator.report_root = tmp_path / "reports"
    orchestrator.figure_root = tmp_path / "figures"
    with pytest.raises(RuntimeError, match="BOTH_SOURCE_LOCKS_REQUIRED"):
        orchestrator._load_locks()


def test_safe_error_is_aggregate_only() -> None:
    error = RuntimeError("failed for /secret/raw/patient-123/image.jpg")
    assert safe_error_code(error) == "RuntimeError"
    assert "/secret" not in safe_error_code(error)


def test_stage_marker_is_atomic_and_contains_no_private_paths(tmp_path: Path) -> None:
    orchestrator = object.__new__(P2Orchestrator)
    orchestrator.stage_root = tmp_path
    orchestrator._write_stage_marker("prepare", {"state": "PASS", "rows": 123})
    payload = json.loads((tmp_path / "prepare.json").read_text(encoding="utf-8"))
    assert payload == {"rows": 123, "stage": "prepare", "state": "PASS"}
    assert not (tmp_path / "prepare.json.tmp").exists()
