"""Train and validate the exclusive-intent GRU."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import (
    ReplayWindowDataset,
    TrackBatchSampler,
    label_counts,
    load_manifest,
    split_replays,
)
from .metrics import classification_metrics, update_confusion
from .model import IntentGRU, ModelConfig


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def class_weights(counts: list[int]) -> torch.Tensor:
    """Use bounded inverse-square-root weights for imbalanced heuristic labels."""
    if any(count == 0 for count in counts):
        missing = [str(index) for index, count in enumerate(counts) if count == 0]
        raise ValueError(f"training split has no examples for label indices: {', '.join(missing)}")
    maximum = max(counts)
    weights = [min(10.0, math.sqrt(maximum / count)) for count in counts]
    mean = sum(weights) / len(weights)
    return torch.tensor([weight / mean for weight in weights], dtype=torch.float32)


def estimate_feature_stats(loader: DataLoader, feature_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    total = torch.zeros(feature_count, dtype=torch.float64)
    squared_total = torch.zeros(feature_count, dtype=torch.float64)
    state_count = 0
    for sequences, _, _, _, _ in loader:
        values = sequences.to(torch.float64)
        total += values.sum(dim=(0, 1))
        squared_total += values.square().sum(dim=(0, 1))
        state_count += values.shape[0] * values.shape[1]
    if state_count == 0:
        raise ValueError("cannot estimate feature statistics from an empty dataset")
    mean = total / state_count
    variance = (squared_total / state_count - mean.square()).clamp_min(0)
    scale = variance.sqrt()
    scale[scale < 1e-6] = 1.0
    return mean.to(torch.float32), scale.to(torch.float32)


def run_epoch(
    model: IntentGRU,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    intent_count: int,
    position_scale: torch.Tensor,
    position_loss_weight: float,
    facing_loss_weight: float,
) -> tuple[float, dict[str, object]]:
    training = optimizer is not None
    model.train(training)
    confusion = torch.zeros((intent_count, intent_count), dtype=torch.int64)
    loss_sum = 0.0
    example_count = 0
    position_error_sum = 0.0
    facing_angle_sum = 0.0

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for sequences, targets, confidence, target_position, target_forward in loader:
            sequences = sequences.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            confidence = confidence.to(device, dtype=torch.float32, non_blocking=True).clamp_min(0.05)
            target_position = target_position.to(device, non_blocking=True)
            target_forward = target_forward.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)
            logits, predicted_position, predicted_forward = model.predict(sequences)
            intent_loss = criterion(logits, targets)
            position_loss = F.smooth_l1_loss(
                predicted_position, target_position, reduction="none"
            ).mean(dim=1)
            facing_cosine = F.cosine_similarity(predicted_forward, target_forward, dim=1)
            facing_loss = 1.0 - facing_cosine
            loss = (
                (intent_loss * confidence).sum() / confidence.sum()
                + position_loss_weight * position_loss.mean()
                + facing_loss_weight * facing_loss.mean()
            )
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            batch_size = targets.shape[0]
            loss_sum += loss.item() * batch_size
            example_count += batch_size
            update_confusion(confusion, targets, logits.argmax(dim=1))
            position_error = ((predicted_position - target_position) * position_scale).norm(dim=1)
            facing_angle = facing_cosine.clamp(-1.0, 1.0).acos() * (180.0 / math.pi)
            position_error_sum += position_error.sum().item()
            facing_angle_sum += facing_angle.sum().item()

    metrics = classification_metrics(confusion)
    metrics["endpoint_position_mean_distance_uu"] = position_error_sum / max(example_count, 1)
    metrics["endpoint_facing_mae_degrees"] = facing_angle_sum / max(example_count, 1)
    return loss_sum / max(example_count, 1), metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="../training_output/manifest.json")
    parser.add_argument("--output", default="artifacts/intent_gru.pt")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--head-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--position-loss-weight", type=float, default=1.0)
    parser.add_argument("--facing-loss-weight", type=float, default=0.25)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--cache-tracks",
        type=int,
        default=4,
        help="maximum decoded tracks retained per dataset worker",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.epochs < 1 or args.patience < 1:
        raise ValueError("batch-size, epochs, and patience must be positive")
    if args.position_loss_weight < 0 or args.facing_loss_weight < 0:
        raise ValueError("endpoint loss weights cannot be negative")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    manifest_path = Path(args.manifest).resolve()
    manifest, shards = load_manifest(manifest_path)
    train_shards, validation_shards = split_replays(
        shards, args.validation_fraction, args.seed
    )
    labels = [str(label) for label in manifest["labels"]]
    sequence_length = int(manifest["sequence_length"])
    feature_names = [str(name) for name in manifest["feature_names"]]
    feature_count = len(feature_names)
    position_scale_values = manifest.get("normalization", {}).get("position_xyz")
    if not isinstance(position_scale_values, list) or len(position_scale_values) != 3:
        raise ValueError("manifest does not specify the three-axis position scale")

    train_dataset = ReplayWindowDataset(
        train_shards, sequence_length, feature_count, args.cache_tracks
    )
    validation_dataset = ReplayWindowDataset(
        validation_shards, sequence_length, feature_count, args.cache_tracks
    )
    if not train_dataset or not validation_dataset:
        raise ValueError("the replay split produced an empty dataset")

    counts = label_counts(train_dataset, len(labels))
    device = choose_device(args.device)
    pin_memory = device.type == "cuda"
    loader_options = {
        "num_workers": args.workers,
        "pin_memory": pin_memory,
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=TrackBatchSampler(
            train_dataset, args.batch_size, shuffle=True, seed=args.seed
        ),
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_sampler=TrackBatchSampler(validation_dataset, args.batch_size),
        **loader_options,
    )

    stats_options = dict(loader_options)
    stats_options["persistent_workers"] = False
    stats_loader = DataLoader(
        train_dataset,
        batch_sampler=TrackBatchSampler(train_dataset, args.batch_size),
        **stats_options,
    )
    print(f"device={device} train_windows={len(train_dataset)} validation_windows={len(validation_dataset)}")
    print(f"train_replays={[shard.replay_id for shard in train_shards]}")
    print(f"validation_replays={[shard.replay_id for shard in validation_shards]}")
    print(f"label_counts={dict(zip(labels, counts))}")
    print("estimating_feature_stats")
    feature_mean, feature_scale = estimate_feature_stats(stats_loader, feature_count)
    print("feature_stats_ready")

    config = ModelConfig(
        feature_count=feature_count,
        intent_count=len(labels),
        hidden_size=args.hidden_size,
        head_size=args.head_size,
        dropout=args.dropout,
    )
    model = IntentGRU(config).to(device)
    model.set_feature_stats(feature_mean.to(device), feature_scale.to(device))
    position_scale = torch.tensor(position_scale_values, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(counts).to(device), reduction="none")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_validation_loss = math.inf
    stale_epochs = 0

    try:
        for epoch in range(1, args.epochs + 1):
            train_loss, train_metrics = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer,
                len(labels),
                position_scale,
                args.position_loss_weight,
                args.facing_loss_weight,
            )
            validation_loss, validation_metrics = run_epoch(
                model,
                validation_loader,
                criterion,
                device,
                None,
                len(labels),
                position_scale,
                args.position_loss_weight,
                args.facing_loss_weight,
            )
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "train": train_metrics,
                "validation": validation_metrics,
            }
            history.append(record)
            validation_f1 = float(validation_metrics["macro_f1"])
            print(
                f"epoch={epoch:03d} train_loss={train_loss:.4f} "
                f"val_loss={validation_loss:.4f} "
                f"val_accuracy={float(validation_metrics['accuracy']):.4f} "
                f"val_macro_f1={validation_f1:.4f} "
                f"val_position_distance={float(validation_metrics['endpoint_position_mean_distance_uu']):.1f}uu "
                f"val_facing_mae={float(validation_metrics['endpoint_facing_mae_degrees']):.1f}deg"
            )

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                stale_epochs = 0
                checkpoint = {
                    "checkpoint_version": 2,
                    "model_state": model.state_dict(),
                    "model_config": config.to_dict(),
                    "labels": labels,
                    "feature_names": feature_names,
                    "sequence_length": sequence_length,
                    "sample_rate_hz": float(manifest.get("sample_rate_hz", 0.0)),
                    "target_window": manifest.get(
                        "target_window",
                        {
                            "start_seconds": 0.0,
                            "end_seconds": float(
                                manifest.get("label_config", {}).get("horizon_seconds", 0.0)
                            ),
                        },
                    ),
                    "label_config": manifest.get("label_config", {}),
                    "normalization": manifest.get("normalization", {}),
                    "prediction_targets": manifest.get("prediction_targets", {}),
                    "loss_weights": {
                        "position": args.position_loss_weight,
                        "facing": args.facing_loss_weight,
                    },
                    "train_replay_ids": [shard.replay_id for shard in train_shards],
                    "validation_replay_ids": [shard.replay_id for shard in validation_shards],
                    "epoch": epoch,
                    "validation_metrics": validation_metrics,
                }
                temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
                torch.save(checkpoint, temporary_path)
                temporary_path.replace(output_path)
            else:
                stale_epochs += 1
                if stale_epochs >= args.patience:
                    print(
                        f"early_stopping epoch={epoch} "
                        f"best_validation_loss={best_validation_loss:.4f}"
                    )
                    break
    finally:
        train_dataset.close()
        validation_dataset.close()

    history_path = output_path.with_suffix(".history.json")
    history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    print(f"checkpoint={output_path}")
    print(f"history={history_path}")


if __name__ == "__main__":
    main()
