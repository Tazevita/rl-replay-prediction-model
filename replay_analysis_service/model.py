"""GRU architecture required to load replay-analysis checkpoints."""

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    feature_count: int
    intent_count: int
    hidden_size: int = 96
    head_size: int = 64
    dropout: float = 0.1

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class IntentGRU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("feature_mean", torch.zeros(config.feature_count))
        self.register_buffer("feature_scale", torch.ones(config.feature_count))
        self.gru = nn.GRU(
            input_size=config.feature_count,
            hidden_size=config.hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.representation = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.head_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.intent_head = nn.Linear(config.head_size, config.intent_count)
        self.position_head = nn.Linear(config.head_size, 3)
        self.forward_head = nn.Linear(config.head_size, 3)

    def set_feature_stats(self, mean: torch.Tensor, scale: torch.Tensor) -> None:
        if mean.shape != self.feature_mean.shape or scale.shape != self.feature_scale.shape:
            raise ValueError("feature statistics do not match the model input size")
        self.feature_mean.copy_(mean)
        self.feature_scale.copy_(scale)

    def predict(
        self, sequence: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if sequence.ndim != 3:
            raise ValueError("sequence must have shape [batch, time, features]")
        sequence = (sequence - self.feature_mean) / self.feature_scale
        _, hidden = self.gru(sequence)
        representation = self.representation(hidden[-1])
        return (
            self.intent_head(representation),
            self.position_head(representation),
            self.forward_head(representation),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.predict(sequence)[0]
