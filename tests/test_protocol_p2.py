from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

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


def test_acceptance_uses_recorded_gpu_without_live_cuda_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    orchestrator = object.__new__(P2Orchestrator)
    orchestrator.repo = REPO
    orchestrator.report_root = tmp_path / "reports"
    orchestrator.report_root.mkdir()
    orchestrator.stage_root = tmp_path / "stages"
    orchestrator.stage_root.mkdir()
    orchestrator.config = SimpleNamespace(sha256="config", seed=42, bootstrap_replicates=2000)
    (orchestrator.stage_root / "gpu-preflight.json").write_text(
        json.dumps({"device": "RECORDED RTX 3060", "peak_gpu_memory_bytes": 123}),
        encoding="utf-8",
    )
    (orchestrator.stage_root / "precheck.json").write_text(
        json.dumps({"elapsed_seconds": 1.5}), encoding="utf-8"
    )
    bundle = SimpleNamespace(predictions={name: [1, 2] for name in ("E1", "E2", "E3", "E4", "E5")})
    histories = {
        source: [{"peak_gpu_memory_bytes": 456, "elapsed_seconds": 2.0}]
        for source in ("A", "B")
    }
    locks = {
        source: {
            "selected_epoch": 0, "source_val_auroc": 0.7, "threshold": 0.5,
            "threshold_origin": "source_val_youden_j",
        }
        for source in ("A", "B")
    }
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *_: (_ for _ in ()).throw(RuntimeError("CUDA hidden")))

    orchestrator._write_acceptance(bundle, histories, locks, {})

    report = (orchestrator.report_root / "acceptance_report.md").read_text(encoding="utf-8")
    assert "RECORDED RTX 3060" in report
