from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from src.p2.training import (
    EarlyStoppingState,
    ResumeMismatchError,
    compute_pos_weight,
    load_training_checkpoint,
    save_training_checkpoint,
    train_accumulation_group,
    update_early_stopping,
    validate_resume_payload,
)


def test_pos_weight_uses_only_supplied_source_train_records() -> None:
    records = [{"target": value} for value in (1, 0, 0, 1, 0, 0)]
    assert compute_pos_weight(records) == 2.0
    with pytest.raises(ValueError, match="both classes"):
        compute_pos_weight([{"target": 0}])


def test_early_stopping_strict_improvement_tie_keeps_earliest_epoch() -> None:
    state = EarlyStoppingState(patience=5)
    for epoch, auc in enumerate((0.60, 0.70, 0.70, 0.69, 0.68, 0.67, 0.66)):
        state = update_early_stopping(state, epoch, auc)

    assert state.best_epoch == 1
    assert state.best_auroc == 0.70
    assert state.epochs_without_improvement == 5
    assert state.should_stop is True


def test_resume_rejects_every_provenance_mismatch() -> None:
    expected = {
        "source": "A",
        "seed": 42,
        "config_sha256": "config",
        "manifest_sha256": "manifest",
        "weights_sha256": "weights",
    }
    validate_resume_payload({**expected, "next_epoch": 2}, expected)
    for key in expected:
        altered = {**expected, "next_epoch": 2, key: "wrong"}
        with pytest.raises(ResumeMismatchError, match="RESUME_PROVENANCE_MISMATCH"):
            validate_resume_payload(altered, expected)


def test_tail_gradient_accumulation_normalizes_by_sample_count() -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(0.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")
    batches = [
        (torch.tensor([[1.0], [2.0], [3.0], [4.0]]), torch.tensor([1.0, 0.0, 1.0, 0.0])),
        (torch.tensor([[5.0], [6.0]]), torch.tensor([1.0, 0.0])),
    ]
    reference = deepcopy(model)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    images = torch.cat([batch[0] for batch in batches])
    targets = torch.cat([batch[1] for batch in batches])
    reference_optimizer.zero_grad()
    criterion(reference(images).squeeze(1), targets).div(len(targets)).backward()
    reference_optimizer.step()

    result = train_accumulation_group(
        model, optimizer, criterion, batches, device=torch.device("cpu"), scaler=None
    )

    assert result["samples"] == 6
    assert result["micro_batches"] == 2
    assert model.weight.detach().item() == pytest.approx(reference.weight.detach().item())


def test_torch_checkpoint_resume_matches_uninterrupted_next_step(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(42)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = torch.tensor([1.0, 0.0])
    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer.zero_grad()
    criterion(model(x).squeeze(1), y).backward()
    optimizer.step()
    checkpoint = tmp_path / "last.pt"
    provenance = {
        "source": "A", "seed": 42, "config_sha256": "c",
        "manifest_sha256": "m", "weights_sha256": "w",
    }
    save_training_checkpoint(
        checkpoint, model=model, optimizer=optimizer, scaler=None,
        next_epoch=1, provenance=provenance, early_stopping=EarlyStoppingState(5),
        history=[{"epoch": 0}], generator=None,
    )

    uninterrupted_model = deepcopy(model)
    uninterrupted_optimizer = torch.optim.AdamW(uninterrupted_model.parameters(), lr=1e-3)
    uninterrupted_optimizer.load_state_dict(optimizer.state_dict())
    uninterrupted_optimizer.zero_grad()
    criterion(uninterrupted_model(x).squeeze(1), y).backward()
    uninterrupted_optimizer.step()

    resumed_model = torch.nn.Linear(2, 1)
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-3)
    payload = load_training_checkpoint(
        checkpoint, model=resumed_model, optimizer=resumed_optimizer, scaler=None,
        expected_provenance=provenance, generator=None,
    )
    resumed_optimizer.zero_grad()
    criterion(resumed_model(x).squeeze(1), y).backward()
    resumed_optimizer.step()

    assert payload["next_epoch"] == 1
    for expected, actual in zip(uninterrupted_model.parameters(), resumed_model.parameters()):
        assert torch.equal(expected, actual)
