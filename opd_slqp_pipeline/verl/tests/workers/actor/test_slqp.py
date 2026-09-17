import json

import torch

from verl.workers.actor.slqp import FEATURE_NAMES, SLQPCalibration


def make_calibration(boundary: float = 0.0) -> SLQPCalibration:
    return SLQPCalibration(
        feature_mean=torch.zeros(4),
        feature_std=torch.ones(4),
        direction=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        good_boundary=boundary,
        hidden_size=4,
        student_model="test",
        excluded_special_token_ids=(0, 99),
    )


def test_prompt_end_is_detached_but_response_receives_gradient():
    hidden = torch.randn(2, 7, 4, requires_grad=True)
    mask = torch.ones(2, 3)
    loss, _ = make_calibration(boundary=100.0).quality_and_loss(
        hidden, mask, response_length=3, huber_delta=1.0, min_response_tokens=2
    )
    loss.backward()
    assert torch.count_nonzero(hidden.grad[:, 3, :]) == 0
    assert torch.count_nonzero(hidden.grad[:, 4:, :]) > 0


def test_good_side_has_exactly_zero_loss():
    hidden = torch.randn(2, 7, 4, requires_grad=True)
    loss, metrics = make_calibration(boundary=-1e6).quality_and_loss(
        hidden, torch.ones(2, 3), response_length=3, huber_delta=1.0, min_response_tokens=2
    )
    assert loss.item() == 0.0
    assert metrics["active_fraction"].item() == 0.0


def test_loss_is_weighted_by_generated_trajectory_tokens():
    hidden = torch.randn(2, 8, 4, requires_grad=True)
    trajectory_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.float32)
    calibration = make_calibration(boundary=100.0)
    loss, _ = calibration.quality_and_loss(
        hidden,
        trajectory_mask,
        response_length=4,
        huber_delta=1.0,
        min_response_tokens=2,
    )
    features = calibration.trajectory_features(hidden, trajectory_mask, response_length=4)
    quality = features[:, 0]
    margin = torch.relu(torch.tensor(100.0) - quality)
    per_response = torch.where(margin <= 1.0, 0.5 * margin.square(), margin - 0.5)
    expected = (2.0 * per_response[0] + 4.0 * per_response[1]) / 6.0
    assert torch.allclose(loss, expected)


def test_load_rejects_wrong_hidden_size_schema(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            {
                "feature_names": list(FEATURE_NAMES),
                "feature_mean": [0, 0, 0, 0],
                "feature_std": [1, 1, 1, 1],
                "direction": [1, 0, 0, 0],
                "good_boundary": 0,
                "hidden_size": 1024,
                "student_model": "Qwen/Qwen3-0.6B-Base",
                "excluded_special_token_ids": [0, 99],
            }
        ),
        encoding="utf-8",
    )
    calibration = SLQPCalibration.from_json(path)
    assert calibration.excluded_special_token_ids == (0, 99)
    hidden = torch.randn(1, 5, 8)
    try:
        calibration.trajectory_features(hidden, torch.ones(1, 2), response_length=2)
    except ValueError as error:
        assert "hidden_size" in str(error)
    else:
        raise AssertionError("Expected hidden-size mismatch to fail")
