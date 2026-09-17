#!/usr/bin/env python3
"""Fit SLQP PCA direction and two-cluster boundary after local quality scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FEATURE_NAMES = (
    "response_prompt_distance_mean",
    "response_prompt_distance_std",
    "response_step_distance_mean",
    "response_step_distance_std",
)
def kmeans_two(values: np.ndarray) -> np.ndarray:
    centers = np.quantile(values, [0.25, 0.75]).astype(np.float64)
    for _ in range(100):
        labels = np.argmin(np.abs(values[:, None] - centers[None, :]), axis=1)
        updated = np.asarray(
            [values[labels == i].mean() if np.any(labels == i) else centers[i] for i in range(2)]
        )
        if np.allclose(updated, centers, atol=1e-10, rtol=0):
            break
        centers = updated
    return np.sort(centers)


def trajectory_features_from_hidden(
    response_hidden: np.ndarray,
    prompt_end_hidden: np.ndarray,
    response_token_ids: np.ndarray,
    excluded_special_token_ids: set[int],
) -> np.ndarray:
    keep = np.asarray(
        [int(token_id) not in excluded_special_token_ids for token_id in response_token_ids],
        dtype=np.bool_,
    )
    response = response_hidden[keep].astype(np.float64)
    if len(response) < 2:
        raise ValueError("A collected response has fewer than two non-special generated tokens")
    prompt_end = prompt_end_hidden.astype(np.float64)
    scale = response.shape[-1] ** 0.5
    distance = np.linalg.norm(response - prompt_end[None, :], axis=-1) / scale
    previous = np.concatenate((prompt_end[None, :], response[:-1]), axis=0)
    step = np.linalg.norm(response - previous, axis=-1) / scale
    return np.asarray(
        [distance.mean(), distance.std(), step.mean(), step.std()], dtype=np.float64
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, help="Unzipped collection directory")
    parser.add_argument("--scores", required=True, help="JSONL containing id and quality_score")
    parser.add_argument("--output", required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    bundle = Path(args.bundle)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if "excluded_special_token_ids" in manifest:
        excluded_special_token_ids = {
            int(token_id) for token_id in manifest["excluded_special_token_ids"]
        }
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            manifest["model"], trust_remote_code=args.trust_remote_code
        )
        excluded_special_token_ids = {int(token_id) for token_id in tokenizer.all_special_ids}
    scores = {}
    with Path(args.scores).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                scores[row["id"]] = float(row["quality_score"])

    ids, features, quality_scores = [], [], []
    for row in manifest["rows"]:
        sample_id = row["id"]
        if sample_id not in scores:
            continue
        with np.load(bundle / "hidden" / f"{sample_id}.npz") as saved:
            feature = trajectory_features_from_hidden(
                saved["response_hidden"],
                saved["prompt_end_hidden"],
                saved["response_token_ids"],
                excluded_special_token_ids,
            )
        ids.append(sample_id)
        features.append(feature)
        quality_scores.append(scores[sample_id])
    if len(ids) < 10:
        raise ValueError(f"Only {len(ids)} collected samples have quality scores")

    matrix = np.asarray(features, dtype=np.float64)
    targets = np.asarray(quality_scores, dtype=np.float64)
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    if np.any(std < 1e-8):
        raise ValueError(f"Degenerate feature standard deviation: {std.tolist()}")
    standardized = (matrix - mean) / std
    _, _, vh = np.linalg.svd(standardized, full_matrices=False)
    direction = vh[0]
    projection = standardized @ direction
    correlation = np.corrcoef(projection, targets)[0, 1]
    if not np.isfinite(correlation) or abs(correlation) < 1e-6:
        raise ValueError("The PCA projection is not correlated with the scored response quality")
    if correlation < 0:
        direction = -direction
        projection = -projection
    centers = kmeans_two(projection)

    payload = {
        "version": 2,
        "student_model": manifest["model"],
        "hidden_size": manifest["hidden_size"],
        "feature_names": list(FEATURE_NAMES),
        "excluded_special_token_ids": sorted(excluded_special_token_ids),
        "token_inclusion_rule": "generated response tokens excluding tokenizer special tokens",
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "direction": direction.tolist(),
        "good_boundary": float(centers.mean()),
        "cluster_centers": centers.tolist(),
        "calibration_samples": len(ids),
        "quality_score_correlation": float(np.corrcoef(projection, targets)[0, 1]),
        "scored_ids": ids,
    }
    Path(args.output).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
