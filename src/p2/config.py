from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import shutil
from typing import Any


EXPECTED_CONFIG: dict[str, Any] = {
    "schema_version": "p2_seed42_v1",
    "protocol": {"path": "docs/protocols/paper1_protocol_v0.1.md"},
    "model": {
        "name": "resnet50",
        "pretrained": True,
        "weights_url": "https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
        "weights_sha256": "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca",
    },
    "sources": ["A", "B"],
    "source_domains": {"A": "derm", "B": "clinic"},
    "seeds": [42],
    "preprocessing": {
        "input_size": 224,
        "train_crop_scale": [0.8, 1.0],
        "horizontal_flip_probability": 0.5,
        "eval_resize": 256,
        "eval_crop": 224,
        "interpolation": "bilinear",
        "rgb": True,
        "normalization_mean": [0.485, 0.456, 0.406],
        "normalization_std": [0.229, 0.224, 0.225],
    },
    "training": {
        "optimizer": "AdamW",
        "learning_rate": 0.0001,
        "weight_decay": 0.0001,
        "scheduler": None,
        "loss": "BCEWithLogitsLoss",
        "pos_weight_origin": "source_train_only",
        "effective_batch_size": 16,
        "micro_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "tail_gradient_normalization": "sample_count",
        "amp": True,
        "max_epochs": 20,
        "patience": 5,
    },
    "selection": {
        "split": "source_val",
        "metric": "auroc",
        "tie_break": "earliest_epoch",
        "threshold_method": "youden_j",
        "threshold_ties": ["higher_specificity", "higher_threshold"],
    },
    "bootstrap": {"replicates": 2000, "seed": 42, "unit": "patient_cluster"},
}


@dataclass(frozen=True)
class P2Config:
    raw: dict[str, Any]
    sha256: str

    @property
    def seeds(self) -> tuple[int, ...]: return tuple(self.raw["seeds"])
    @property
    def seed(self) -> int: return self.seeds[0]
    @property
    def sources(self) -> tuple[str, ...]: return tuple(self.raw["sources"])
    @property
    def source_domains(self) -> dict[str, str]: return dict(self.raw["source_domains"])
    @property
    def model(self) -> str: return self.raw["model"]["name"]
    @property
    def pretrained(self) -> bool: return self.raw["model"]["pretrained"]
    @property
    def weights_url(self) -> str: return self.raw["model"]["weights_url"]
    @property
    def weights_sha256(self) -> str: return self.raw["model"]["weights_sha256"]
    @property
    def input_size(self) -> int: return self.raw["preprocessing"]["input_size"]
    @property
    def train_crop_scale(self) -> tuple[float, float]: return tuple(self.raw["preprocessing"]["train_crop_scale"])
    @property
    def horizontal_flip_probability(self) -> float: return self.raw["preprocessing"]["horizontal_flip_probability"]
    @property
    def eval_resize(self) -> int: return self.raw["preprocessing"]["eval_resize"]
    @property
    def eval_crop(self) -> int: return self.raw["preprocessing"]["eval_crop"]
    @property
    def interpolation(self) -> str: return self.raw["preprocessing"]["interpolation"]
    @property
    def normalization_mean(self) -> tuple[float, float, float]: return tuple(self.raw["preprocessing"]["normalization_mean"])
    @property
    def normalization_std(self) -> tuple[float, float, float]: return tuple(self.raw["preprocessing"]["normalization_std"])
    @property
    def optimizer(self) -> str: return self.raw["training"]["optimizer"]
    @property
    def learning_rate(self) -> float: return self.raw["training"]["learning_rate"]
    @property
    def weight_decay(self) -> float: return self.raw["training"]["weight_decay"]
    @property
    def scheduler(self) -> None: return self.raw["training"]["scheduler"]
    @property
    def loss(self) -> str: return self.raw["training"]["loss"]
    @property
    def effective_batch_size(self) -> int: return self.raw["training"]["effective_batch_size"]
    @property
    def micro_batch_size(self) -> int: return self.raw["training"]["micro_batch_size"]
    @property
    def gradient_accumulation_steps(self) -> int: return self.raw["training"]["gradient_accumulation_steps"]
    @property
    def amp(self) -> bool: return self.raw["training"]["amp"]
    @property
    def max_epochs(self) -> int: return self.raw["training"]["max_epochs"]
    @property
    def patience(self) -> int: return self.raw["training"]["patience"]
    @property
    def selection_split(self) -> str: return self.raw["selection"]["split"]
    @property
    def selection_metric(self) -> str: return self.raw["selection"]["metric"]
    @property
    def selection_tie_break(self) -> str: return self.raw["selection"]["tie_break"]
    @property
    def threshold_method(self) -> str: return self.raw["selection"]["threshold_method"]
    @property
    def bootstrap_replicates(self) -> int: return self.raw["bootstrap"]["replicates"]
    @property
    def bootstrap_seed(self) -> int: return self.raw["bootstrap"]["seed"]


@dataclass(frozen=True)
class P2Paths:
    protocol: Path
    p0_acceptance: Path
    p1_summary: Path
    weights: Path
    storage_root: Path


def _digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_p2_config(path: Path) -> P2Config:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("P2_CONFIG_INVALID") from exc
    if raw != EXPECTED_CONFIG:
        raise ValueError("FROZEN_CONFIG_MISMATCH")
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
    return P2Config(raw=raw, sha256=sha256(canonical).hexdigest())


def validate_p2_prerequisites(
    config: P2Config,
    paths: P2Paths,
    *,
    minimum_free_bytes: int = 15 * 1024**3,
) -> dict[str, Any]:
    required = (paths.protocol, paths.p0_acceptance, paths.p1_summary, paths.weights)
    if any(not path.is_file() for path in required):
        raise ValueError("P2_PREREQUISITE_MISSING")
    try:
        p0 = json.loads(paths.p0_acceptance.read_text(encoding="utf-8"))
        p1 = json.loads(paths.p1_summary.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("P2_PREREQUISITE_INVALID") from exc
    if p0.get("status") != "READY":
        raise ValueError("P0_NOT_READY")
    if (
        p1.get("status") != "PASS"
        or p1.get("seed") != 42
        or p1.get("model") != "resnet50"
        or any(p1.get("sources", {}).get(source, {}).get("state") != "PASS" for source in config.sources)
    ):
        raise ValueError("P1_NOT_READY")
    weight_digest = _digest(paths.weights)
    if weight_digest != config.weights_sha256 or p1.get("weights_sha256") != weight_digest:
        raise ValueError("WEIGHTS_PROVENANCE_MISMATCH")
    free = shutil.disk_usage(paths.storage_root).free
    if free < minimum_free_bytes:
        raise ValueError("INSUFFICIENT_DISK")
    return {
        "status": "READY",
        "protocol_sha256": _digest(paths.protocol),
        "weights_sha256": weight_digest,
        "config_sha256": config.sha256,
        "free_bytes": free,
    }
