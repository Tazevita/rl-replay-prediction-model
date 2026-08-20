import pytest

torch = pytest.importorskip("torch")

from prediction_model.model import IntentGRU, ModelConfig
from prediction_model.predict import RollingIntentPredictor


def test_predictor_returns_scaled_endpoint_and_normalized_facing(tmp_path):
    config = ModelConfig(feature_count=2, intent_count=2, hidden_size=4, head_size=3)
    model = IntentGRU(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.position_head.bias.copy_(torch.tensor([0.5, -0.25, 0.1]))
        model.forward_head.bias.copy_(torch.tensor([0.0, 2.0, 0.0]))

    checkpoint = tmp_path / "endpoint.pt"
    torch.save(
        {
            "checkpoint_version": 2,
            "model_state": model.state_dict(),
            "model_config": config.to_dict(),
            "labels": ["ROTATE", "HOLD"],
            "feature_names": ["first", "second"],
            "sequence_length": 2,
            "sample_rate_hz": 10.0,
            "target_window": {"start_seconds": 2.0, "end_seconds": 3.5},
            "label_config": {},
            "normalization": {"position_xyz": [4096.0, 5120.0, 2044.0]},
        },
        checkpoint,
    )

    prediction = RollingIntentPredictor(checkpoint, "cpu").predict_endpoint_window(
        [[0.0, 0.0], [0.0, 0.0]]
    )

    assert prediction.seconds == 3.5
    assert prediction.position == pytest.approx((2048.0, -1280.0, 204.4))
    assert prediction.forward == pytest.approx((0.0, 1.0, 0.0))
    assert sum(prediction.probabilities.values()) == pytest.approx(1.0)


def test_predictor_batches_windows(tmp_path):
    config = ModelConfig(feature_count=2, intent_count=2, hidden_size=4, head_size=3)
    model = IntentGRU(config)
    checkpoint = tmp_path / "batch.pt"
    torch.save(
        {
            "checkpoint_version": 2,
            "model_state": model.state_dict(),
            "model_config": config.to_dict(),
            "labels": ["ROTATE", "HOLD"],
            "feature_names": ["first", "second"],
            "sequence_length": 2,
            "sample_rate_hz": 10.0,
            "target_window": {"start_seconds": 0.0, "end_seconds": 1.0},
            "label_config": {},
            "normalization": {"position_xyz": [4096.0, 5120.0, 2044.0]},
        },
        checkpoint,
    )
    predictor = RollingIntentPredictor(checkpoint, "cpu")
    windows = [
        [[0.0, 0.0], [0.0, 0.0]],
        [[1.0, 1.0], [1.0, 1.0]],
    ]

    batched = predictor.predict_endpoint_windows(windows)
    singles = [predictor.predict_endpoint_window(window) for window in windows]

    assert len(batched) == 2
    for batch_prediction, single_prediction in zip(batched, singles):
        assert batch_prediction.probabilities == pytest.approx(single_prediction.probabilities)
        assert batch_prediction.position == pytest.approx(single_prediction.position)
        assert batch_prediction.forward == pytest.approx(single_prediction.forward)
