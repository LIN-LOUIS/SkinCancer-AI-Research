from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Sequence


def configure_determinism(seed: int) -> dict[str, Any]:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    exceptions: list[str] = []
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError as exc:
        exceptions.append(type(exc).__name__)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return {
        "requested": True,
        "enabled": torch.are_deterministic_algorithms_enabled(),
        "exceptions": exceptions,
    }


def build_transforms(config: Any) -> tuple[Any, Any]:
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    interpolation = {
        "bilinear": InterpolationMode.BILINEAR,
    }[config.interpolation]
    normalize = transforms.Normalize(config.normalization_mean, config.normalization_std)
    train = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                config.input_size,
                scale=config.train_crop_scale,
                interpolation=interpolation,
            ),
            transforms.RandomHorizontalFlip(config.horizontal_flip_probability),
            transforms.ToTensor(),
            normalize,
        ]
    )
    evaluation = transforms.Compose(
        [
            transforms.Resize(config.eval_resize, interpolation=interpolation),
            transforms.CenterCrop(config.eval_crop),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train, evaluation


class ManifestImageDataset:
    def __init__(self, records: Sequence[dict[str, Any]], transform: Any) -> None:
        self.records = list(records)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        import torch
        from PIL import Image

        row = self.records[index]
        with Image.open(row["image_path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, torch.tensor(float(row["target"]), dtype=torch.float32)


def initialize_resnet50(weights_path: Path, device: Any) -> Any:
    import torch
    from torch import nn
    from torchvision.models import resnet50

    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model = resnet50(weights=None)
    model.load_state_dict(state)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model.to(device)
