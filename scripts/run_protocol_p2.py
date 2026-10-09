from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.p2.config import P2Paths, load_p2_config, validate_p2_prerequisites
from src.p2.data import load_data_seal, prepare_private_views
from src.p2.evaluation import (
    EvaluationBundle,
    aggregate_prediction_metrics,
    evaluate_locked_matrix,
    pair_e2_predictions,
)
from src.p2.reporting import publish_aggregate_reports
from src.p2.training import SourceSelectionLock, train_accumulation_group, train_source
from src.p2.torch_runtime import initialize_resnet50
from src.utils.run_registry import append_run_record


def safe_error_code(error: Exception) -> str:
    return type(error).__name__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run frozen Paper 1 P2 seed-42 baseline")
    parser.add_argument(
        "stage",
        choices=("prepare", "gpu-preflight", "precheck", "train", "evaluate", "publish", "all"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--p1-summary", type=Path)
    return parser


class P2Orchestrator:
    STAGES = ("prepare", "gpu-preflight", "precheck", "train", "evaluate", "publish")

    def __init__(self, repo: Path, *, weights: Path, p1_summary: Path, resume: bool) -> None:
        self.repo = repo
        self.config_path = repo / "experiments/paper1_concept_dg/configs/p2_seed42.yaml"
        self.config = load_p2_config(self.config_path)
        self.protocol = repo / "docs/protocols/paper1_protocol_v0.1.md"
        self.p0_acceptance = repo / "data_local/protocol_v0.1/p0_acceptance.json"
        self.p1_summary = p1_summary
        self.weights = weights
        self.private_root = repo / "data_local/protocol_v0.1/p2_seed42"
        self.run_root = repo / "runs/protocol_v0.1/p2_seed42"
        self.stage_root = self.run_root / "stages"
        self.report_root = repo / "reports/protocol_v0.1/p2_seed42"
        self.figure_root = repo / "experiments/paper1_concept_dg/figures/p2_seed42"
        self.resume = resume

    @classmethod
    def from_arguments(cls, arguments: argparse.Namespace) -> "P2Orchestrator":
        if arguments.seed != 42:
            raise ValueError("SEED_NOT_FROZEN")
        repo = Path(__file__).resolve().parents[1]
        weights = arguments.weights or repo / "runs/protocol_v0.1/p2_seed42/pretrained/resnet50-11ad3fa6.pth"
        default_p1 = repo / "runs/protocol_v0.1/smoke_seed42_f40e53e/smoke_summary.json"
        if not default_p1.is_file():
            default_p1 = repo / "runs/protocol_v0.1/smoke_seed42/smoke_summary.json"
        return cls(
            repo,
            weights=weights,
            p1_summary=arguments.p1_summary or default_p1,
            resume=arguments.resume,
        )

    def _write_stage_marker(self, stage: str, payload: Mapping[str, Any]) -> None:
        self.stage_root.mkdir(parents=True, exist_ok=True)
        path = self.stage_root / f"{stage}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps({"stage": stage, **dict(payload)}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def _stage_passed(self, stage: str) -> bool:
        path = self.stage_root / f"{stage}.json"
        if not path.is_file():
            return False
        try:
            return json.loads(path.read_text(encoding="utf-8")).get("state") == "PASS"
        except (OSError, json.JSONDecodeError):
            return False

    def _require_stage(self, stage: str) -> None:
        if not self._stage_passed(stage):
            raise RuntimeError(f"REQUIRED_STAGE_NOT_PASS:{stage}")

    def prepare(self) -> dict[str, Any]:
        prerequisites = validate_p2_prerequisites(
            self.config,
            P2Paths(
                self.protocol,
                self.p0_acceptance,
                self.p1_summary,
                self.weights,
                self.repo,
            ),
        )
        if (self.private_root / "data_seal.json").is_file():
            if not self.resume:
                raise RuntimeError("PRIVATE_OUTPUT_ALREADY_EXISTS")
            seal = load_data_seal(self.private_root)
        else:
            seal = prepare_private_views(
                self.repo / "data_local/protocol_v0.1/manifests/derm7pt_v0.1.csv",
                self.repo / "data_local/protocol_v0.1/manifests/pad_ufes20_v0.1.csv",
                self.private_root,
            )
        result = {
            "state": "PASS",
            "config_sha256": prerequisites["config_sha256"],
            "protocol_sha256": prerequisites["protocol_sha256"],
            "weights_sha256": prerequisites["weights_sha256"],
            "data_seal_sha256": seal.sha256,
            "counts": seal.counts,
            "free_bytes": prerequisites["free_bytes"],
        }
        self._write_stage_marker("prepare", result)
        return result

    def gpu_preflight(self) -> dict[str, Any]:
        self._require_stage("prepare")
        if self.resume and self._stage_passed("gpu-preflight"):
            return json.loads((self.stage_root / "gpu-preflight.json").read_text(encoding="utf-8"))
        import torch
        from torch import nn

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA_REQUIRED")
        device = torch.device("cuda")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        model = initialize_resnet50(self.weights, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss(reduction="sum")
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        batches = [
            (torch.rand(4, 3, 224, 224), torch.tensor([0.0, 1.0, 0.0, 1.0]))
            for _ in range(4)
        ]
        started = time.monotonic()
        stats = train_accumulation_group(
            model, optimizer, criterion, batches, device=device, scaler=scaler
        )
        torch.cuda.synchronize()
        result = {
            "state": "PASS",
            "device": torch.cuda.get_device_name(0),
            "total_gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
            "micro_batch_size": 4,
            "accumulation_steps": 4,
            "effective_batch_size": 16,
            "input_size": 224,
            "amp": True,
            "samples": stats["samples"],
            "elapsed_seconds": time.monotonic() - started,
        }
        self._write_stage_marker("gpu-preflight", result)
        return result

    def precheck(self) -> dict[str, Any]:
        self._require_stage("gpu-preflight")
        if self.resume and self._stage_passed("precheck"):
            return json.loads((self.stage_root / "precheck.json").read_text(encoding="utf-8"))
        seal = load_data_seal(self.private_root)
        output = self.run_root / "precheck"
        results = {}
        started = time.monotonic()
        for source in ("A", "B"):
            results[source] = train_source(
                config=self.config,
                seal=seal,
                source=source,
                weights_path=self.weights,
                output_root=output,
                resume=False,
                epoch_limit=1,
                create_lock=False,
            )
        result = {
            "state": "PASS",
            "formal_training": False,
            "sources": {
                source: {
                    "state": values["state"],
                    "epochs_completed": values["epochs_completed"],
                    "elapsed_seconds": sum(row["elapsed_seconds"] for row in values["history"]),
                    "peak_gpu_memory_bytes": max(row["peak_gpu_memory_bytes"] for row in values["history"]),
                }
                for source, values in results.items()
            },
            "elapsed_seconds": time.monotonic() - started,
        }
        self._write_stage_marker("precheck", result)
        return result

    def _lock_path(self, source: str) -> Path:
        return self.run_root / "formal" / f"resnet50_{source}_seed42" / "selection_lock.json"

    def _load_locks(self) -> dict[str, dict[str, Any]]:
        paths = {source: self._lock_path(source) for source in ("A", "B")}
        if any(not path.is_file() for path in paths.values()):
            raise RuntimeError("BOTH_SOURCE_LOCKS_REQUIRED")
        return {source: json.loads(path.read_text(encoding="utf-8")) for source, path in paths.items()}

    def train(self) -> dict[str, Any]:
        self._require_stage("precheck")
        seal = load_data_seal(self.private_root)
        output = self.run_root / "formal"
        locks: dict[str, Any] = {}
        started = time.monotonic()
        for source in ("A", "B"):
            if self._lock_path(source).is_file():
                if not self.resume:
                    raise RuntimeError("SELECTION_LOCK_ALREADY_EXISTS")
                locks[source] = json.loads(self._lock_path(source).read_text(encoding="utf-8"))
            else:
                locks[source] = train_source(
                    config=self.config,
                    seal=seal,
                    source=source,
                    weights_path=self.weights,
                    output_root=output,
                    resume=self.resume,
                    create_lock=True,
                ).as_dict()
            self._register_source(locks[source])
        result = {
            "state": "PASS",
            "elapsed_seconds": time.monotonic() - started,
            "sources": {
                source: {
                    "selected_epoch": lock["selected_epoch"],
                    "source_val_auroc": lock["source_val_auroc"],
                    "threshold": lock["threshold"],
                    "threshold_origin": lock["threshold_origin"],
                }
                for source, lock in locks.items()
            },
        }
        self._write_stage_marker("train", result)
        return result

    def _register_source(self, lock: Mapping[str, Any]) -> None:
        import torch

        history_path = Path(str(lock["checkpoint"])).parents[1] / "history.json"
        history = json.loads(history_path.read_text(encoding="utf-8"))["history"]
        record = {
            "run_id": f"resnet50-{lock['source'].lower()}-seed42-p2",
            "source_domain": lock["source"],
            "tests": "P2 formal seed42 baseline",
            "model": "resnet50",
            "weights_provenance": {"url": self.config.weights_url, "sha256": lock["weights_sha256"]},
            "data_version": "protocol-v0.1-p2-data-seal",
            "protocol_version": "0.1",
            "source_split_hash": lock["source_manifest_sha256"],
            "commit_sha": subprocess.run(["git", "rev-parse", "HEAD"], cwd=self.repo, capture_output=True, text=True, check=True).stdout.strip(),
            "config_hash": lock["config_sha256"],
            "seed": 42,
            "environment": {
                "python": sys.version.split()[0], "torch": torch.__version__,
                "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                "amp": True, "peak_gpu_memory_bytes": max(row["peak_gpu_memory_bytes"] for row in history),
            },
            "epoch": lock["selected_epoch"],
            "checkpoint": "private_best_checkpoint",
            "threshold_origin": lock["threshold_origin"],
            "threshold": lock["threshold"],
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "metrics": lock["source_val_metrics"],
            "artifact_paths": ["private_checkpoint", "private_history", "private_selection_lock"],
            "state": "PASS",
        }
        registry = self.run_root / "registry.jsonl"
        if registry.is_file() and any(json.loads(line).get("run_id") == record["run_id"] for line in registry.read_text(encoding="utf-8").splitlines() if line):
            return
        append_run_record(registry, record)

    def evaluate(self) -> dict[str, Any]:
        self._require_stage("train")
        seal = load_data_seal(self.private_root)
        locks = self._load_locks()
        output = self.run_root / "evaluation"
        if (output / "evaluation_consumed.json").is_file():
            if not self.resume:
                raise RuntimeError("EVALUATION_SEAL_ALREADY_CONSUMED")
            bundle = self._load_private_bundle(seal, locks)
        else:
            bundle = evaluate_locked_matrix(
                config=self.config, seal=seal, locks=locks, output_root=output
            )
        result = {
            "state": "PASS",
            "rows": {experiment: len(rows) for experiment, rows in bundle.predictions.items()},
            "metrics": bundle.metrics,
            "e2_paired": bundle.e2_paired,
        }
        self._write_stage_marker("evaluate", result)
        return result

    def _load_private_bundle(self, seal: Any, locks: Mapping[str, Mapping[str, Any]]) -> EvaluationBundle:
        predictions: dict[str, list[dict[str, Any]]] = {}
        paths: dict[str, str] = {}
        for experiment in ("E1", "E2", "E3", "E4", "E5"):
            path = self.run_root / "evaluation/predictions" / f"{experiment}.csv"
            with path.open(newline="", encoding="utf-8") as handle:
                rows = []
                for row in csv.DictReader(handle):
                    converted: dict[str, Any] = dict(row)
                    converted["target"] = int(row["target"])
                    converted["prediction"] = int(row["prediction"])
                    converted["score"] = float(row["score"])
                    rows.append(converted)
            predictions[experiment] = rows; paths[experiment] = str(path)
        source_for = {"E1": "A", "E2": "A", "E3": "A", "E4": "B", "E5": "B"}
        metrics = {
            experiment: aggregate_prediction_metrics(rows, float(locks[source_for[experiment]]["threshold"]))
            for experiment, rows in predictions.items()
        }
        return EvaluationBundle(
            "p2_evaluation_bundle_v1", metrics, predictions,
            pair_e2_predictions(predictions["E1"], predictions["E2"]), paths, seal.sha256,
        )

    def publish(self) -> dict[str, Any]:
        self._require_stage("evaluate")
        if self.report_root.exists() or self.figure_root.exists():
            if self.resume and (self.report_root / "metrics.json").is_file():
                return {"state": "PASS", "resumed": True}
            raise RuntimeError("PUBLIC_OUTPUT_ALREADY_EXISTS")
        seal = load_data_seal(self.private_root)
        locks = self._load_locks()
        bundle = self._load_private_bundle(seal, locks)
        histories = {
            source: json.loads((self.run_root / "formal" / f"resnet50_{source}_seed42/history.json").read_text(encoding="utf-8"))["history"]
            for source in ("A", "B")
        }
        published = publish_aggregate_reports(
            bundle=bundle,
            histories=histories,
            locks=locks,
            report_root=self.report_root,
            figure_root=self.figure_root,
            bootstrap_replicates=self.config.bootstrap_replicates,
            bootstrap_seed=self.config.bootstrap_seed,
        )
        self._write_acceptance(bundle, histories, locks, published)
        result = {
            "state": "PASS",
            "reports": len(published["reports"]) + 1,
            "figures": len(published["figures"]),
        }
        self._write_stage_marker("publish", result)
        return result

    def _write_acceptance(self, bundle: EvaluationBundle, histories: Mapping[str, Any], locks: Mapping[str, Any], published: Mapping[str, Any]) -> None:
        import torch

        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=self.repo, capture_output=True, text=True, check=True).stdout.strip()
        preflight = json.loads((self.stage_root / "gpu-preflight.json").read_text(encoding="utf-8"))
        precheck = json.loads((self.stage_root / "precheck.json").read_text(encoding="utf-8"))
        lines = [
            "# Paper 1 P2 Seed 42 验收报告", "",
            f"- Git SHA: `{commit}`", f"- config SHA256: `{self.config.sha256}`",
            f"- seed: `{self.config.seed}`", f"- Python: `{sys.version.split()[0]}`",
            f"- PyTorch: `{torch.__version__}`", f"- CUDA runtime: `{torch.version.cuda}`",
            f"- GPU: `{torch.cuda.get_device_name(0)}`",
            f"- 224×224 AMP 门禁峰值显存: `{preflight['peak_gpu_memory_bytes']}` bytes",
            f"- A/B 单 epoch 预检总耗时: `{precheck['elapsed_seconds']:.3f}` seconds", "",
            "## Source selection", "",
            "| Source | Best epoch (0-based) | source-val AUROC | threshold | origin | epochs run | peak GPU bytes | train seconds |",
            "|---|---:|---:|---:|---|---:|---:|---:|",
        ]
        for source in ("A", "B"):
            history = histories[source]
            lines.append(f"| {source} | {locks[source]['selected_epoch']} | {locks[source]['source_val_auroc']:.6f} | {locks[source]['threshold']:.8f} | {locks[source]['threshold_origin']} | {len(history)} | {max(row['peak_gpu_memory_bytes'] for row in history)} | {sum(row['elapsed_seconds'] for row in history):.3f} |")
        lines += ["", "## Evaluation", "", "完整指标见 `metrics.csv` / `metrics.json`；所有公开内容均为聚合结果。", "", "| Experiment | rows |", "|---|---:|"]
        for experiment, rows in bundle.predictions.items(): lines.append(f"| {experiment} | {len(rows)} |")
        lines += ["", "## Bootstrap", "", f"- requested replicates: {self.config.bootstrap_replicates}", "- unit: PAD patient cluster; E3/E5 paired patient draws", "", "**READY**", ""]
        (self.report_root / "acceptance_report.md").write_text("\n".join(lines), encoding="utf-8")

    def run(self, stage: str) -> dict[str, Any]:
        if stage == "all":
            result: dict[str, Any] = {}
            for name in self.STAGES:
                result[name] = getattr(self, name.replace("-", "_"))()
                print(json.dumps({"stage": name, "state": "PASS"}, sort_keys=True), flush=True)
            return result
        return getattr(self, stage.replace("-", "_"))()


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        orchestrator = P2Orchestrator.from_arguments(arguments)
        result = orchestrator.run(arguments.stage)
    except Exception as exc:
        print(json.dumps({"state": "BLOCKED", "error": safe_error_code(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps({"state": "READY", "result": result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
