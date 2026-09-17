"""Single-response latent quality projection (SLQP).

The calibration is frozen. Gradients flow only through the current
student response trajectory, while the prompt-end state is stop-gradient.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch


FEATURE_NAMES = (
    "response_prompt_distance_mean",
    "response_prompt_distance_std",
    "response_step_distance_mean",
    "response_step_distance_std",
)


@dataclass(frozen=True)
class SLQPCalibration:
    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    direction: torch.Tensor
    good_boundary: float
    hidden_size: int
    student_model: str
    excluded_special_token_ids: tuple[int, ...]

    @classmethod
    def from_json(cls, path: str | Path) -> "SLQPCalibration":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"SLQP calibration file does not exist: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("feature_names") != list(FEATURE_NAMES):
            raise ValueError(
                "SLQP calibration feature_names do not match the four implemented trajectory features"
            )
        mean = torch.tensor(payload["feature_mean"], dtype=torch.float32)
        std = torch.tensor(payload["feature_std"], dtype=torch.float32)
        direction = torch.tensor(payload["direction"], dtype=torch.float32)
        if mean.shape != (4,) or std.shape != (4,) or direction.shape != (4,):
            raise ValueError("SLQP feature_mean, feature_std, and direction must each contain four values")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not torch.isfinite(direction).all():
            raise ValueError("SLQP calibration contains non-finite values")
        if (std <= 0).any():
            raise ValueError("SLQP feature_std values must be positive")
        direction_norm = direction.norm()
        if direction_norm <= 0:
            raise ValueError("SLQP direction must be non-zero")
        direction = direction / direction_norm
        return cls(
            feature_mean=mean,
            feature_std=std,
            direction=direction,
            good_boundary=float(payload["good_boundary"]),
            hidden_size=int(payload["hidden_size"]),
            student_model=str(payload.get("student_model", "unknown")),
            excluded_special_token_ids=tuple(
                sorted({int(token_id) for token_id in payload.get("excluded_special_token_ids", [])})
            ),
        )

    def trajectory_features(
        self,
        full_hidden: torch.Tensor,
        trajectory_mask: torch.Tensor,
        response_length: int,
    ) -> torch.Tensor:
        """Return the differentiable four-dimensional feature for each response."""
        if full_hidden.ndim != 3:
            raise ValueError(f"Expected [batch, sequence, hidden] states, got {tuple(full_hidden.shape)}")
        if full_hidden.size(-1) != self.hidden_size:
            raise ValueError(
                f"SLQP calibration hidden_size={self.hidden_size}, model hidden_size={full_hidden.size(-1)}"
            )
        if response_length < 1 or full_hidden.size(1) <= response_length:
            raise ValueError("SLQP needs at least one prompt token and one response token")

        hidden = full_hidden.float()
        mask = trajectory_mask[:, :response_length].to(dtype=hidden.dtype)
        prompt_end = hidden[:, -response_length - 1, :].detach()
        response_hidden = hidden[:, -response_length:, :]
        scale = math.sqrt(hidden.size(-1))

        prompt_distance = torch.linalg.vector_norm(
            response_hidden - prompt_end.unsqueeze(1), dim=-1
        ) / scale
        # Previous means the previous included generated token, not an excluded
        # EOS/padding/control token. This keeps calibration and training geometry
        # identical even if a special token appears inside the response span.
        positions = torch.arange(1, response_length + 1, device=hidden.device).unsqueeze(0)
        valid_positions = torch.where(mask.bool(), positions, torch.zeros_like(positions))
        previous_positions = torch.cat(
            (
                torch.zeros((hidden.size(0), 1), dtype=positions.dtype, device=hidden.device),
                torch.cummax(valid_positions[:, :-1], dim=1).values,
            ),
            dim=1,
        )
        gather_index = (previous_positions.clamp_min(1) - 1).unsqueeze(-1).expand_as(response_hidden)
        previous_response_hidden = torch.gather(response_hidden, dim=1, index=gather_index)
        previous_hidden = torch.where(
            previous_positions.unsqueeze(-1) > 0,
            previous_response_hidden,
            prompt_end.unsqueeze(1),
        )
        step_distance = torch.linalg.vector_norm(response_hidden - previous_hidden, dim=-1) / scale

        def masked_mean_std(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            count = mask.sum(dim=1).clamp_min(1.0)
            mean = (values * mask).sum(dim=1) / count
            variance = (((values - mean.unsqueeze(1)) ** 2) * mask).sum(dim=1) / count
            return mean, torch.sqrt(variance.clamp_min(0.0) + 1e-8)

        distance_mean, distance_std = masked_mean_std(prompt_distance)
        step_mean, step_std = masked_mean_std(step_distance)
        return torch.stack((distance_mean, distance_std, step_mean, step_std), dim=-1)

    def quality_and_loss(
        self,
        full_hidden: torch.Tensor,
        trajectory_mask: torch.Tensor,
        response_length: int,
        huber_delta: float,
        min_response_tokens: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if huber_delta <= 0:
            raise ValueError("SLQP huber_delta must be positive")
        features = self.trajectory_features(full_hidden, trajectory_mask, response_length)
        mean = self.feature_mean.to(device=features.device, dtype=features.dtype)
        std = self.feature_std.to(device=features.device, dtype=features.dtype)
        direction = self.direction.to(device=features.device, dtype=features.dtype)
        standardized = (features - mean) / std
        quality = standardized @ direction

        margin = torch.relu(quality.new_tensor(self.good_boundary) - quality)
        loss_per_sample = torch.where(
            margin <= huber_delta,
            0.5 * margin.square() / huber_delta,
            margin - 0.5 * huber_delta,
        )
        token_count = trajectory_mask[:, :response_length].sum(dim=1).to(features.dtype).detach()
        valid = (token_count >= min_response_tokens).to(features.dtype)
        token_weight = token_count * valid
        token_denominator = token_weight.sum().clamp_min(1.0)
        sequence_denominator = valid.sum().clamp_min(1.0)

        # Q itself is length-invariant because its four features are mean/std.
        # Only the outer loss is weighted by the number of actually generated,
        # non-special response tokens: sum_i T_i * loss_i / sum_i T_i.
        loss = (loss_per_sample * token_weight).sum() / token_denominator
        active = ((margin > 0).to(features.dtype) * token_weight).sum() / token_denominator
        metrics = {
            "loss": loss.detach(),
            "quality_mean": (quality.detach() * valid).sum() / sequence_denominator,
            "token_weighted_quality_mean": (quality.detach() * token_weight).sum() / token_denominator,
            "margin_mean": (margin.detach() * valid).sum() / sequence_denominator,
            "active_fraction": active.detach(),
            "generated_tokens": token_weight.sum().detach(),
            "response_length_mean": token_weight.sum().detach() / sequence_denominator,
            "distance_mean": features[:, 0].detach().mean(),
            "distance_std": features[:, 1].detach().mean(),
            "step_mean": features[:, 2].detach().mean(),
            "step_std": features[:, 3].detach().mean(),
        }
        return loss, metrics
