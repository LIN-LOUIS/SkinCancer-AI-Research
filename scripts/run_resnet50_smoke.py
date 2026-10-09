from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics.binary import select_source_validation_checkpoint
from src.utils.run_registry import append_run_record


RESNET50_WEIGHTS_URL = "https://download.pytorch.org/models/resnet50-11ad3fa6.pth"


def safe_error_summary(error: Exception) -> str:
    """Return an aggregate error class without local paths or sample data."""

    return type(error).__name__


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source_records(
    manifest_path: Path, domain: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read only one approved Derm7pt source domain from a private manifest."""

    if domain not in {"derm", "clinic"}:
        raise ValueError("source domain must be derm or clinic")
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = []
        for row in csv.DictReader(handle):
            if row.get("domain") != domain or row.get("split") not in {"train", "valid"}:
                continue
            target = int(row["target"])
            if target not in {0, 1}:
                raise ValueError("source targets must be binary")
            rows.append({**row, "target": target})
    train = [row for row in rows if row["split"] == "train"]
    valid = [row for row in rows if row["split"] == "valid"]
    if not train or not valid:
        raise ValueError("source manifest requires non-empty train and valid records")
    return train, valid


def run_smoke(
    *,
    manifest_path: Path,
    weights_path: Path,
    output_root: Path,
    seed: int = 42,
    train_batches: int = 20,
    batch_size: int = 4,
    image_size: int = 64,
) -> dict[str, Any]:
    """Run the bounded A/B ResNet50 smoke without reading target-domain data."""

    if seed != 42:
        raise ValueError("Protocol v0.1 smoke seed must be 42")
    if not 20 <= train_batches <= 30:
        raise ValueError("smoke train_batches must remain between 20 and 30")
    if batch_size <= 1 or image_size < 32:
        raise ValueError("invalid smoke batch size or image size")
    if not weights_path.is_file():
        raise FileNotFoundError("ImageNet ResNet50 weights are unavailable")

    try:
        import torch
        from PIL import Image
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
        from torchvision import transforms
        from torchvision.models import resnet50
    except ImportError as exc:
        raise RuntimeError("PyTorch, torchvision, and Pillow are required") from exc

    class ManifestDataset(Dataset):
        def __init__(self, records: list[dict[str, Any]], transform: Any) -> None:
            self.records = records
            self.transform = transform

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, index: int) -> tuple[Any, Any]:
            record = self.records[index]
            with Image.open(record["image_path"]) as image:
                tensor = self.transform(image.convert("RGB"))
            return tensor, torch.tensor(float(record["target"]), dtype=torch.float32)

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_digest = _sha256(weights_path)
    manifest_digest = _sha256(manifest_path)
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize,
        ]
    )
    valid_transform = transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.ToTensor(), normalize]
    )
    pretrained_state = torch.load(weights_path, map_location="cpu", weights_only=True)
    output_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}

    for domain, source_name in (("derm", "A"), ("clinic", "B")):
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        train_records, valid_records = load_source_records(manifest_path, domain)
        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(
            ManifestDataset(train_records, train_transform),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            generator=generator,
        )
        valid_loader = DataLoader(
            ManifestDataset(valid_records, valid_transform),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        model = resnet50(weights=None)
        model.load_state_dict(pretrained_state)
        model.fc = nn.Linear(model.fc.in_features, 1)
        model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
        criterion = nn.BCEWithLogitsLoss()
        model.train()
        losses: list[float] = []
        for batch_index, (images, targets) in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            logits = model(images.to(device)).squeeze(1)
            loss = criterion(logits, targets.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if batch_index >= train_batches:
                break
        if len(losses) != train_batches:
            raise RuntimeError("source train split did not satisfy smoke batch budget")

        checkpoint_dir = output_root / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"resnet50_{source_name}_seed42_smoke.pt"
        torch.save(
            {"model_state": model.state_dict(), "seed": seed, "source_domain": source_name},
            checkpoint_path,
        )
        reloaded = resnet50(weights=None)
        reloaded.fc = nn.Linear(reloaded.fc.in_features, 1)
        saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
        reloaded.load_state_dict(saved["model_state"])
        reloaded.to(device)
        reloaded.eval()
        targets_out: list[int] = []
        scores_out: list[float] = []
        with torch.no_grad():
            for images, targets in valid_loader:
                scores = torch.sigmoid(reloaded(images.to(device)).squeeze(1)).cpu()
                targets_out.extend(int(value) for value in targets.tolist())
                scores_out.extend(float(value) for value in scores.tolist())

        relative_checkpoint = str(checkpoint_path.relative_to(output_root))
        selected = select_source_validation_checkpoint(
            [
                {
                    "epoch": 0,
                    "split_role": "source_val",
                    "targets": targets_out,
                    "scores": scores_out,
                    "checkpoint": relative_checkpoint,
                }
            ]
        )
        config = {
            "batch_size": batch_size,
            "image_size": image_size,
            "seed": seed,
            "train_batches": train_batches,
            "optimizer": "SGD",
            "learning_rate": 1e-4,
            "source_domain": source_name,
        }
        config_hash = sha256(
            json.dumps(config, sort_keys=True).encode("utf-8")
        ).hexdigest()
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=True,
        )
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        run_id = f"resnet50-{source_name.lower()}-seed42-smoke"
        record = {
            "run_id": run_id,
            "source_domain": source_name,
            "tests": "P1 bounded smoke; not research performance",
            "model": "resnet50",
            "weights_provenance": {
                "url": RESNET50_WEIGHTS_URL,
                "sha256": weights_digest,
            },
            "data_version": "derm7pt-manifest-v0.1",
            "protocol_version": "0.1",
            "source_split_hash": manifest_digest,
            "commit_sha": completed.stdout.strip(),
            "config_hash": config_hash,
            "seed": seed,
            "environment": {
                "device": str(device),
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
            },
            "epoch": 0,
            "checkpoint": relative_checkpoint,
            "threshold_origin": selected["threshold_origin"],
            "threshold": selected["threshold"],
            "timestamp": timestamp,
            "metrics": selected["validation_metrics"],
            "artifact_paths": [relative_checkpoint],
            "state": "PASS",
        }
        append_run_record(output_root / "registry.jsonl", record)
        results[source_name] = {
            "state": "PASS",
            "train_batches": len(losses),
            "train_samples_seen": len(losses) * batch_size,
            "validation_samples": len(targets_out),
            "validation_n_pos": selected["validation_metrics"]["n_pos"],
            "validation_n_neg": selected["validation_metrics"]["n_neg"],
            "checkpoint_reload": "PASS",
            "device": str(device),
        }
        print(
            json.dumps(
                {"source_domain": source_name, **results[source_name]},
                sort_keys=True,
            ),
            flush=True,
        )

    summary = {
        "status": "PASS",
        "seed": seed,
        "model": "resnet50",
        "full_training": False,
        "weights_sha256": weights_digest,
        "sources": results,
    }
    temporary = output_root / "smoke_summary.json.tmp"
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_root / "smoke_summary.json")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run bounded ResNet50 seed=42 smoke on Derm7pt A/B sources"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=64)
    arguments = parser.parse_args(argv)
    try:
        summary = run_smoke(
            manifest_path=arguments.manifest,
            weights_path=arguments.weights,
            output_root=arguments.output_root,
            seed=arguments.seed,
            train_batches=arguments.train_batches,
            batch_size=arguments.batch_size,
            image_size=arguments.image_size,
        )
    except Exception as exc:
        print(f"SMOKE BLOCKED: {safe_error_summary(exc)}", file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
