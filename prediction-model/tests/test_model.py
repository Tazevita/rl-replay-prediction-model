import pytest

torch = pytest.importorskip("torch")

from prediction_model.model import IntentGRU, ModelConfig
from prediction_model.train import class_weights, run_epoch


def test_model_returns_one_logit_per_intent():
    model = IntentGRU(ModelConfig(feature_count=94, intent_count=12))
    model.set_feature_stats(torch.ones(94), torch.full((94,), 2.0))
    logits = model(torch.zeros(4, 11, 94))
    assert logits.shape == (4, 12)


def test_model_predicts_endpoint_position_and_facing():
    model = IntentGRU(ModelConfig(feature_count=94, intent_count=12))
    logits, position, forward = model.predict(torch.zeros(4, 11, 94))

    assert logits.shape == (4, 12)
    assert position.shape == (4, 3)
    assert forward.shape == (4, 3)


def test_class_weights_favor_rare_intents_without_extreme_values():
    weights = class_weights([1000, 100, 1])
    assert weights[2] > weights[1] > weights[0]
    assert weights.mean().item() == pytest.approx(1.0)


def test_training_epoch_optimizes_intent_and_endpoint_targets():
    model = IntentGRU(ModelConfig(feature_count=2, intent_count=2, hidden_size=4, head_size=3))
    dataset = torch.utils.data.TensorDataset(
        torch.zeros(2, 3, 2),
        torch.tensor([0, 1]),
        torch.ones(2),
        torch.tensor([[0.25, -0.5, 0.1], [-0.25, 0.5, 0.1]]),
        torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss, metrics = run_epoch(
        model,
        loader,
        criterion,
        torch.device("cpu"),
        optimizer,
        intent_count=2,
        position_scale=torch.tensor([4096.0, 5120.0, 2044.0]),
        position_loss_weight=1.0,
        facing_loss_weight=0.25,
    )

    assert loss > 0
    assert metrics["endpoint_position_mean_distance_uu"] > 0
    assert 0 <= metrics["endpoint_facing_mae_degrees"] <= 180
