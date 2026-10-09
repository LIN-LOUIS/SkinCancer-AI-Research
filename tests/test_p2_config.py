from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.p2.config import P2Config, P2Paths, load_p2_config, validate_p2_prerequisites


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "experiments/paper1_concept_dg/configs/p2_seed42.yaml"


def test_frozen_seed42_configuration_is_exact() -> None:
    config = load_p2_config(CONFIG)

    assert config.seed == 42
    assert config.seeds == (42,)
    assert config.sources == ("A", "B")
    assert config.model == "resnet50"
    assert config.pretrained is True
    assert config.input_size == 224
    assert config.train_crop_scale == (0.8, 1.0)
    assert config.horizontal_flip_probability == 0.5
    assert config.eval_resize == 256
    assert config.eval_crop == 224
    assert config.interpolation == "bilinear"
    assert config.normalization_mean == (0.485, 0.456, 0.406)
    assert config.normalization_std == (0.229, 0.224, 0.225)
    assert config.optimizer == "AdamW"
    assert config.learning_rate == 1e-4
    assert config.weight_decay == 1e-4
    assert config.scheduler is None
    assert config.loss == "BCEWithLogitsLoss"
    assert config.effective_batch_size == 16
    assert config.micro_batch_size == 4
    assert config.gradient_accumulation_steps == 4
    assert config.amp is True
    assert config.max_epochs == 20
    assert config.patience == 5
    assert config.selection_split == "source_val"
    assert config.selection_metric == "auroc"
    assert config.selection_tie_break == "earliest_epoch"
    assert config.threshold_method == "youden_j"
    assert config.bootstrap_replicates == 2000
    assert config.bootstrap_seed == 42


def test_altered_frozen_value_is_rejected(tmp_path: Path) -> None:
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    raw["training"]["learning_rate"] = 3e-4
    altered = tmp_path / "altered.yaml"
    altered.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="FROZEN_CONFIG_MISMATCH"):
        load_p2_config(altered)


def test_prerequisites_validate_hashes_readiness_and_disk(tmp_path: Path) -> None:
    config = load_p2_config(CONFIG)
    protocol = tmp_path / "protocol.md"
    protocol.write_text("frozen protocol", encoding="utf-8")
    weight = tmp_path / "weights.pth"
    weight.write_bytes(b"official weights")
    weight_digest = hashlib.sha256(weight.read_bytes()).hexdigest()
    raw = json.loads(json.dumps(config.raw))
    raw["model"]["weights_sha256"] = weight_digest
    config = P2Config(raw=raw, sha256=config.sha256)
    p0 = tmp_path / "p0.json"
    p0.write_text(json.dumps({"status": "READY"}), encoding="utf-8")
    p1 = tmp_path / "p1.json"
    p1.write_text(
        json.dumps(
            {
                "status": "PASS",
                "seed": 42,
                "model": "resnet50",
                "weights_sha256": weight_digest,
                "sources": {"A": {"state": "PASS"}, "B": {"state": "PASS"}},
            }
        ),
        encoding="utf-8",
    )
    paths = P2Paths(
        protocol=protocol,
        p0_acceptance=p0,
        p1_summary=p1,
        weights=weight,
        storage_root=tmp_path,
    )

    result = validate_p2_prerequisites(config, paths, minimum_free_bytes=1)

    assert result["status"] == "READY"
    assert result["protocol_sha256"] == hashlib.sha256(protocol.read_bytes()).hexdigest()
    assert result["weights_sha256"] == hashlib.sha256(weight.read_bytes()).hexdigest()
    assert result["config_sha256"] == config.sha256
    assert set(result) == {
        "status",
        "protocol_sha256",
        "weights_sha256",
        "config_sha256",
        "free_bytes",
    }


@pytest.mark.parametrize(
    ("p0_status", "p1_status", "minimum_free", "error"),
    [
        ("BLOCKED", "PASS", 1, "P0_NOT_READY"),
        ("READY", "BLOCKED", 1, "P1_NOT_READY"),
        ("READY", "PASS", 10**30, "INSUFFICIENT_DISK"),
    ],
)
def test_prerequisites_fail_closed_with_aggregate_errors(
    tmp_path: Path,
    p0_status: str,
    p1_status: str,
    minimum_free: int,
    error: str,
) -> None:
    config = load_p2_config(CONFIG)
    protocol = tmp_path / "protocol.md"
    protocol.write_text("protocol", encoding="utf-8")
    weight = tmp_path / "weights.pth"
    weight.write_bytes(b"weights")
    digest = hashlib.sha256(weight.read_bytes()).hexdigest()
    raw = json.loads(json.dumps(config.raw))
    raw["model"]["weights_sha256"] = digest
    config = P2Config(raw=raw, sha256=config.sha256)
    p0 = tmp_path / "p0.json"
    p0.write_text(json.dumps({"status": p0_status}), encoding="utf-8")
    p1 = tmp_path / "p1.json"
    p1.write_text(
        json.dumps(
            {
                "status": p1_status,
                "seed": 42,
                "model": "resnet50",
                "weights_sha256": digest,
                "sources": {"A": {"state": "PASS"}, "B": {"state": "PASS"}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=error) as caught:
        validate_p2_prerequisites(
            config,
            P2Paths(protocol, p0, p1, weight, tmp_path),
            minimum_free_bytes=minimum_free,
        )
    message = str(caught.value)
    assert str(tmp_path) not in message
    assert "weights.pth" not in message
