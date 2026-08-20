"""Run intent inference from normalized feature states."""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from .model import IntentGRU, ModelConfig
from .train import choose_device


@dataclass(frozen=True)
class EndpointPrediction:
    probabilities: dict[str, float]
    position: tuple[float, float, float]
    forward: tuple[float, float, float]
    seconds: float


class RollingIntentPredictor:
    """Retain exactly the checkpoint's configured one-second feature window."""

    def __init__(self, checkpoint: str | Path, device: str = "auto") -> None:
        self.device = choose_device(device)
        saved = torch.load(Path(checkpoint), map_location=self.device, weights_only=True)
        if int(saved.get("checkpoint_version", 0)) != 2:
            raise ValueError("unsupported checkpoint version")
        self.labels = [str(label) for label in saved["labels"]]
        self.feature_names = [str(name) for name in saved["feature_names"]]
        self.sequence_length = int(saved["sequence_length"])
        self.sample_rate_hz = float(saved.get("sample_rate_hz", 0.0))
        self.target_window = saved.get("target_window", {"start_seconds": 0.0, "end_seconds": 1.5})
        self.label_config = saved.get("label_config", {})
        position_scale = saved.get("normalization", {}).get("position_xyz")
        if not isinstance(position_scale, (list, tuple)) or len(position_scale) != 3:
            raise ValueError("checkpoint does not specify the three-axis position scale")
        self.position_scale = torch.tensor(position_scale, dtype=torch.float32)
        self.model = IntentGRU(ModelConfig(**saved["model_config"])).to(self.device)
        self.model.load_state_dict(saved["model_state"])
        self.model.eval()
        self.history: deque[torch.Tensor] = deque(maxlen=self.sequence_length)

    def reset(self) -> None:
        self.history.clear()

    def add_state(self, features: Sequence[float]) -> dict[str, float] | None:
        prediction = self.add_state_prediction(features)
        return prediction.probabilities if prediction is not None else None

    def add_state_prediction(self, features: Sequence[float]) -> EndpointPrediction | None:
        if len(features) != len(self.feature_names):
            raise ValueError(
                f"expected {len(self.feature_names)} features, received {len(features)}"
            )
        self.history.append(torch.tensor(features, dtype=torch.float32))
        if len(self.history) < self.sequence_length:
            return None
        sequence = torch.stack(tuple(self.history)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits, normalized_position, forward = self.model.predict(sequence)
            probabilities = logits.softmax(dim=-1)[0].to("cpu")
            position = normalized_position[0].to("cpu") * self.position_scale
            forward = torch.nn.functional.normalize(forward[0], dim=0).to("cpu")
        return EndpointPrediction(
            probabilities=dict(zip(self.labels, probabilities.tolist())),
            position=tuple(position.tolist()),
            forward=tuple(forward.tolist()),
            seconds=float(self.target_window["end_seconds"]),
        )

    def predict_window(self, states: Sequence[Sequence[float]]) -> dict[str, float]:
        return self.predict_endpoint_window(states).probabilities

    def predict_windows(
        self, windows: Sequence[Sequence[Sequence[float]]]
    ) -> list[dict[str, float]]:
        return [prediction.probabilities for prediction in self.predict_endpoint_windows(windows)]

    def predict_endpoint_window(
        self, states: Sequence[Sequence[float]]
    ) -> EndpointPrediction:
        return self.predict_endpoint_windows([states])[0]

    def predict_endpoint_windows(
        self, windows: Sequence[Sequence[Sequence[float]]]
    ) -> list[EndpointPrediction]:
        if not windows:
            return []
        for states in windows:
            if len(states) != self.sequence_length:
                raise ValueError(
                    f"expected {self.sequence_length} states, received {len(states)}"
                )
            if any(len(state) != len(self.feature_names) for state in states):
                raise ValueError(f"expected {len(self.feature_names)} features per state")

        sequences = torch.tensor(windows, dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            logits, normalized_positions, forwards = self.model.predict(sequences)
            probabilities = logits.softmax(dim=-1).to("cpu")
            positions = normalized_positions.to("cpu") * self.position_scale
            forwards = torch.nn.functional.normalize(forwards, dim=1).to("cpu")
        return [
            EndpointPrediction(
                probabilities=dict(zip(self.labels, sample_probabilities.tolist())),
                position=tuple(position.tolist()),
                forward=tuple(forward.tolist()),
                seconds=float(self.target_window["end_seconds"]),
            )
            for sample_probabilities, position, forward in zip(
                probabilities, positions, forwards
            )
        ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("input", help="JSON array of normalized feature states")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    states = payload.get("features") if isinstance(payload, dict) else payload
    if not isinstance(states, list):
        raise ValueError("input must be an array or an object containing a features array")
    predictor = RollingIntentPredictor(args.checkpoint, args.device)
    prediction = predictor.predict_endpoint_window(states)
    ranked = dict(
        sorted(prediction.probabilities.items(), key=lambda item: item[1], reverse=True)
    )
    print(
        json.dumps(
            {
                "intents": ranked,
                "endpoint_seconds": prediction.seconds,
                "position": dict(zip(("x", "y", "z"), prediction.position)),
                "forward": dict(zip(("x", "y", "z"), prediction.forward)),
                "coordinates": "canonical team-relative Rocket League units",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
