from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import random
import time
from typing import Any, Iterable, Mapping, Sequence

from src.metrics.binary import binary_metrics, youden_threshold
from src.p2.data import DataSeal, load_training_view
from src.p2.torch_runtime import (
    ManifestImageDataset,
    build_transforms,
    configure_determinism,
    initialize_resnet50,
)


class ResumeMismatchError(RuntimeError):
    pass


@dataclass(frozen=True)
class EarlyStoppingState:
    patience: int
    best_epoch: int | None = None
    best_auroc: float = float("-inf")
    epochs_without_improvement: int = 0
    should_stop: bool = False


@dataclass(frozen=True)
class SourceSelectionLock:
    schema_version: str
    source: str
    seed: int
    immutable: bool
    data_seal_sha256: str
    config_sha256: str
    weights_sha256: str
    selected_epoch: int
    checkpoint: str
    checkpoint_sha256: str
    threshold: float
    threshold_origin: str
    source_val_metrics: dict[str, Any]
    source_val_auroc: float
    source_manifest_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_pos_weight(records: Sequence[Mapping[str, Any]]) -> float:
    targets = [int(record["target"]) for record in records]
    positives = sum(targets)
    negatives = len(targets) - positives
    if not positives or not negatives:
        raise ValueError("pos_weight requires both classes")
    return negatives / positives


def update_early_stopping(
    state: EarlyStoppingState, epoch: int, val_auroc: float
) -> EarlyStoppingState:
    if val_auroc > state.best_auroc:
        return EarlyStoppingState(
            patience=state.patience,
            best_epoch=epoch,
            best_auroc=float(val_auroc),
            epochs_without_improvement=0,
            should_stop=False,
        )
    without = state.epochs_without_improvement + 1
    return EarlyStoppingState(
        patience=state.patience,
        best_epoch=state.best_epoch,
        best_auroc=state.best_auroc,
        epochs_without_improvement=without,
        should_stop=without >= state.patience,
    )


def validate_resume_payload(
    payload: Mapping[str, Any], expected_provenance: Mapping[str, Any]
) -> None:
    keys = ("source", "seed", "config_sha256", "manifest_sha256", "weights_sha256")
    if any(payload.get(key) != expected_provenance.get(key) for key in keys):
        raise ResumeMismatchError("RESUME_PROVENANCE_MISMATCH")


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def save_training_checkpoint(
    path: Path,
    *,
    model: Any,
    optimizer: Any,
    scaler: Any,
    next_epoch: int,
    provenance: Mapping[str, Any],
    early_stopping: EarlyStoppingState,
    history: Sequence[Mapping[str, Any]],
    generator: Any,
    validation: Mapping[str, Any] | None = None,
) -> None:
    import torch

    payload = {
        **dict(provenance),
        "schema_version": "p2_training_checkpoint_v1",
        "next_epoch": next_epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "early_stopping": asdict(early_stopping),
        "history": list(history),
        "python_rng_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "generator_state": generator.get_state() if generator is not None else None,
        "validation": dict(validation) if validation is not None else None,
    }
    _atomic_torch_save(payload, path)


def load_training_checkpoint(
    path: Path,
    *,
    model: Any,
    optimizer: Any,
    scaler: Any,
    expected_provenance: Mapping[str, Any],
    generator: Any,
) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    validate_resume_payload(payload, expected_provenance)
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    if scaler is not None and payload.get("scaler_state") is not None:
        scaler.load_state_dict(payload["scaler_state"])
    random.setstate(payload["python_rng_state"])
    torch.set_rng_state(payload["torch_rng_state"])
    if torch.cuda.is_available() and payload.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
    if generator is not None and payload.get("generator_state") is not None:
        generator.set_state(payload["generator_state"])
    return payload


def train_accumulation_group(
    model: Any,
    optimizer: Any,
    criterion: Any,
    batches: Sequence[tuple[Any, Any]],
    *,
    device: Any,
    scaler: Any,
) -> dict[str, float | int]:
    import torch

    optimizer.zero_grad(set_to_none=True)
    sample_count = 0
    loss_sum = 0.0
    amp_enabled = scaler is not None and device.type == "cuda"
    for images, targets in batches:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = model(images).squeeze(1)
            loss = criterion(logits, targets)
        if scaler is None:
            loss.backward()
        else:
            scaler.scale(loss).backward()
        sample_count += int(targets.numel())
        loss_sum += float(loss.detach().cpu())
    if not sample_count:
        raise ValueError("gradient accumulation group must not be empty")
    if scaler is not None:
        scaler.unscale_(optimizer)
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(sample_count)
    if scaler is None:
        optimizer.step()
    else:
        scaler.step(optimizer)
        scaler.update()
    return {
        "loss_sum": loss_sum,
        "mean_loss": loss_sum / sample_count,
        "samples": sample_count,
        "micro_batches": len(batches),
    }


def _groups(loader: Iterable[tuple[Any, Any]], size: int) -> Iterable[list[tuple[Any, Any]]]:
    group: list[tuple[Any, Any]] = []
    for batch in loader:
        group.append(batch)
        if len(group) == size:
            yield group
            group = []
    if group:
        yield group


def train_one_epoch(
    model: Any,
    loader: Any,
    optimizer: Any,
    criterion: Any,
    *,
    device: Any,
    scaler: Any,
    accumulation_steps: int,
) -> dict[str, float | int]:
    model.train()
    total_loss = 0.0
    samples = 0
    optimizer_steps = 0
    for batches in _groups(loader, accumulation_steps):
        result = train_accumulation_group(
            model, optimizer, criterion, batches, device=device, scaler=scaler
        )
        total_loss += float(result["loss_sum"])
        samples += int(result["samples"])
        optimizer_steps += 1
    return {"loss": total_loss / samples, "samples": samples, "optimizer_steps": optimizer_steps}


def infer_scores(model: Any, loader: Any, device: Any) -> tuple[list[int], list[float]]:
    import torch

    model.eval()
    targets_out: list[int] = []
    scores_out: list[float] = []
    with torch.no_grad():
        for images, targets in loader:
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images.to(device, non_blocking=True)).squeeze(1)
            scores = torch.sigmoid(logits).cpu()
            targets_out.extend(int(value) for value in targets.tolist())
            scores_out.extend(float(value) for value in scores.tolist())
    return targets_out, scores_out


def _file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def train_source(
    *,
    config: Any,
    seal: DataSeal,
    source: str,
    weights_path: Path,
    output_root: Path,
    resume: bool = False,
    epoch_limit: int | None = None,
    create_lock: bool = True,
) -> SourceSelectionLock | dict[str, Any]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    deterministic = configure_determinism(config.seed)
    device = torch.device("cuda")
    records = load_training_view(seal, source)
    train_transform, evaluation_transform = build_transforms(config)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        ManifestImageDataset(records["train"], train_transform),
        batch_size=config.micro_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        ManifestImageDataset(records["valid"], evaluation_transform),
        batch_size=config.micro_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    model = initialize_resnet50(weights_path, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    pos_weight = torch.tensor([compute_pos_weight(records["train"])], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="sum")
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp)
    manifest_sha = sha256(
        (seal.view_hashes[f"{source}_train"] + seal.view_hashes[f"{source}_valid"]).encode()
    ).hexdigest()
    provenance = {
        "source": source,
        "seed": config.seed,
        "config_sha256": config.sha256,
        "manifest_sha256": manifest_sha,
        "weights_sha256": _file_digest(weights_path),
    }
    run_root = output_root / f"resnet50_{source}_seed42"
    last_path = run_root / "checkpoints/last.pt"
    best_path = run_root / "checkpoints/best.pt"
    state = EarlyStoppingState(config.patience)
    history: list[dict[str, Any]] = []
    start_epoch = 0
    if resume and last_path.is_file():
        payload = load_training_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            expected_provenance=provenance,
            generator=generator,
        )
        start_epoch = int(payload["next_epoch"])
        state = EarlyStoppingState(**payload["early_stopping"])
        history = list(payload["history"])
    elif resume and run_root.exists():
        raise ResumeMismatchError("RESUME_CHECKPOINT_MISSING")
    elif run_root.exists():
        raise RuntimeError("RUN_OUTPUT_ALREADY_EXISTS")

    maximum = min(config.max_epochs, epoch_limit or config.max_epochs)
    for epoch in range(start_epoch, maximum):
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats(device)
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device=device,
            scaler=scaler,
            accumulation_steps=config.gradient_accumulation_steps,
        )
        targets, scores = infer_scores(model, valid_loader, device)
        validation = binary_metrics(targets, scores, threshold=0.5)
        updated = update_early_stopping(state, epoch, validation["auroc"])
        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "source_val_auroc": validation["auroc"],
            "source_val_auprc": validation["average_precision"],
            "elapsed_seconds": time.monotonic() - started,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "checkpoint": "checkpoints/last.pt",
        }
        history.append(row)
        if updated.best_epoch == epoch:
            save_training_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                next_epoch=epoch + 1,
                provenance=provenance,
                early_stopping=updated,
                history=history,
                generator=generator,
                validation={"targets": targets, "scores": scores},
            )
        state = updated
        save_training_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            next_epoch=epoch + 1,
            provenance=provenance,
            early_stopping=state,
            history=history,
            generator=generator,
        )
        _atomic_json(
            {
                "schema_version": "p2_training_history_v1",
                "source": source,
                "seed": config.seed,
                "pos_weight": float(pos_weight.item()),
                "deterministic": deterministic,
                "amp": bool(config.amp),
                "history": history,
            },
            run_root / "history.json",
        )
        if state.should_stop:
            break

    if not create_lock:
        return {
            "state": "PASS",
            "source": source,
            "epochs_completed": len(history),
            "history": history,
        }
    if state.best_epoch is None or not best_path.is_file():
        raise RuntimeError("BEST_CHECKPOINT_MISSING")
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    validation_private = best["validation"]
    threshold = youden_threshold(validation_private["targets"], validation_private["scores"])["threshold"]
    metrics = binary_metrics(validation_private["targets"], validation_private["scores"], threshold)
    lock = SourceSelectionLock(
        schema_version="p2_source_selection_lock_v1",
        source=source,
        seed=config.seed,
        immutable=True,
        data_seal_sha256=seal.sha256,
        config_sha256=config.sha256,
        weights_sha256=provenance["weights_sha256"],
        selected_epoch=int(state.best_epoch),
        checkpoint=str(best_path),
        checkpoint_sha256=_file_digest(best_path),
        threshold=float(threshold),
        threshold_origin="source_val_youden_j",
        source_val_metrics=metrics,
        source_val_auroc=float(state.best_auroc),
        source_manifest_sha256=manifest_sha,
    )
    lock_path = run_root / "selection_lock.json"
    if lock_path.exists():
        raise RuntimeError("SELECTION_LOCK_ALREADY_EXISTS")
    _atomic_json(lock.as_dict(), lock_path)
    return lock
