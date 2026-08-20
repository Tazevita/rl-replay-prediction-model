"""Dependency-free classification metrics."""

from __future__ import annotations

import torch


def update_confusion(
    confusion: torch.Tensor, targets: torch.Tensor, predictions: torch.Tensor
) -> None:
    size = confusion.shape[0]
    encoded = targets.to("cpu") * size + predictions.to("cpu")
    confusion += torch.bincount(encoded, minlength=size * size).view(size, size)


def classification_metrics(confusion: torch.Tensor) -> dict[str, object]:
    confusion = confusion.to(torch.float64)
    true_positive = confusion.diag()
    support = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)
    precision = true_positive / predicted.clamp_min(1)
    recall = true_positive / support.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    present = support > 0
    macro_f1 = f1[present].mean().item() if present.any() else 0.0
    total = confusion.sum().item()
    accuracy = true_positive.sum().item() / total if total else 0.0
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_f1": f1.tolist(),
        "support": support.to(torch.int64).tolist(),
        "confusion": confusion.to(torch.int64).tolist(),
    }
