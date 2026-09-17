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


def read_scores(path: Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = row["id"]
            if sample_id in scores:
                raise ValueError(f"Duplicate score ID at line {line_number}: {sample_id}")
            scores[sample_id] = float(row["quality_score"])
    return scores


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def kmeans_two(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centers = np.quantile(values, [0.25, 0.75]).astype(np.float64)
    for _ in range(100):
        labels = np.argmin(np.abs(values[:, None] - centers[None, :]), axis=1)
        updated = np.asarray(
            [values[labels == index].mean() if np.any(labels == index) else centers[index] for index in range(2)]
        )
        if np.allclose(updated, centers, atol=1e-10, rtol=0):
            break
        centers = updated
    order = np.argsort(centers)
    remap = np.empty(2, dtype=np.int64)
    remap[order] = np.arange(2)
    return centers[order], remap[labels]


def silhouette_1d(values: np.ndarray, labels: np.ndarray) -> float:
    scores = []
    for index, value in enumerate(values):
        same = values[(labels == labels[index]) & (np.arange(len(values)) != index)]
        other = values[labels != labels[index]]
        if len(same) == 0 or len(other) == 0:
            scores.append(0.0)
            continue
        a = float(np.abs(same - value).mean())
        b = float(np.abs(other - value).mean())
        scores.append((b - a) / max(a, b, 1e-12))
    return float(np.mean(scores))


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = labels == 1
    negative = labels == 0
    n_pos = int(positive.sum())
    n_neg = int(negative.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[positive].sum() - n_pos * (n_pos - 1) / 2.0) / (n_pos * n_neg))


def pca_direction(standardized: np.ndarray) -> np.ndarray:
    _, _, vh = np.linalg.svd(standardized, full_matrices=False)
    direction = vh[0].astype(np.float64)
    return direction / np.linalg.norm(direction)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--bootstrap-runs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if int(manifest.get("version", 0)) != 2:
        raise ValueError("Expected collector manifest version 2")
    if tuple(manifest.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("Manifest feature names do not match SLQP V1")
    scores = read_scores(Path(args.scores))

    ids, features, targets, lengths = [], [], [], []
    for row in manifest["rows"]:
        sample_id = row["id"]
        if sample_id not in scores:
            continue
        ids.append(sample_id)
        features.append([float(value) for value in row["trajectory_features"]])
        targets.append(scores[sample_id])
        lengths.append(int(row["trajectory_response_length"]))
    if len(ids) != int(manifest["samples"]):
        raise ValueError(
            f"Expected a score for every manifest row: matched {len(ids)} / {manifest['samples']}"
        )

    matrix = np.asarray(features, dtype=np.float64)
    quality = np.asarray(targets, dtype=np.float64)
    response_lengths = np.asarray(lengths, dtype=np.float64)
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    if np.any(std < 1e-8):
        raise ValueError(f"Degenerate feature standard deviation: {std.tolist()}")
    standardized = (matrix - mean) / std
    direction = pca_direction(standardized)
    projection = standardized @ direction
    raw_correlation = correlation(projection, quality)
    if not np.isfinite(raw_correlation) or abs(raw_correlation) < 1e-6:
        raise ValueError("PCA-1 is not correlated with scored quality")
    if raw_correlation < 0:
        direction = -direction
        projection = -projection

    centers, labels = kmeans_two(projection)
    boundary = float(centers.mean())
    cluster_quality_means = [
        float(quality[labels == cluster].mean()) for cluster in range(2)
    ]
    if cluster_quality_means[1] <= cluster_quality_means[0]:
        raise ValueError("The positive PCA cluster does not have higher mean rubric quality")

    quartile_size = len(quality) // 4
    quality_order = np.argsort(quality, kind="mergesort")
    low_indices = quality_order[:quartile_size]
    high_indices = quality_order[-quartile_size:]
    extreme_indices = np.concatenate((low_indices, high_indices))
    extreme_labels = np.concatenate(
        (np.zeros(quartile_size, dtype=np.int64), np.ones(quartile_size, dtype=np.int64))
    )
    extreme_projection = projection[extreme_indices]
    extreme_accuracy = float(
        np.mean((extreme_projection >= boundary).astype(np.int64) == extreme_labels)
    )
    extreme_auc = binary_auc(extreme_labels, extreme_projection)

    rng = np.random.default_rng(args.seed)
    bootstrap_cosines = []
    sample_size = max(10, int(round(0.8 * len(matrix))))
    for _ in range(args.bootstrap_runs):
        indices = rng.choice(len(matrix), size=sample_size, replace=False)
        boot_matrix = matrix[indices]
        boot_std = boot_matrix.std(axis=0)
        if np.any(boot_std < 1e-8):
            continue
        boot_standardized = (boot_matrix - boot_matrix.mean(axis=0)) / boot_std
        boot_direction = pca_direction(boot_standardized)
        bootstrap_cosines.append(abs(float(np.dot(direction, boot_direction))))

    quality_pearson = correlation(projection, quality)
    quality_spearman = correlation(rankdata(projection), rankdata(quality))
    length_pearson = correlation(projection, response_lengths)
    quality_length_pearson = correlation(quality, response_lengths)
    quality_length_spearman = correlation(rankdata(quality), rankdata(response_lengths))
    log_lengths = np.log1p(response_lengths)
    length_design = np.column_stack((np.ones(len(log_lengths)), log_lengths))
    projection_residual = projection - length_design @ np.linalg.lstsq(
        length_design, projection, rcond=None
    )[0]
    quality_residual = quality - length_design @ np.linalg.lstsq(
        length_design, quality, rcond=None
    )[0]
    length_controlled_quality_pearson = correlation(projection_residual, quality_residual)
    payload = {
        "version": 2,
        "student_model": manifest["model"],
        "hidden_size": int(manifest["hidden_size"]),
        "feature_names": list(FEATURE_NAMES),
        "excluded_special_token_ids": [
            int(token_id) for token_id in manifest["excluded_special_token_ids"]
        ],
        "token_inclusion_rule": manifest["token_inclusion_rule"],
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "direction": direction.tolist(),
        "good_boundary": boundary,
        "cluster_centers": centers.tolist(),
        "calibration_samples": len(ids),
        "quality_score_correlation": quality_pearson,
        "scored_ids": ids,
    }
    report = {
        "method": "student-only four-feature standardized PCA-1, quality-oriented, KMeans-2 midpoint",
        "samples": len(ids),
        "quality_score_min": float(quality.min()),
        "quality_score_mean": float(quality.mean()),
        "quality_score_median": float(np.median(quality)),
        "quality_score_max": float(quality.max()),
        "quality_pearson": quality_pearson,
        "quality_spearman": quality_spearman,
        "projection_length_pearson": length_pearson,
        "quality_length_pearson": quality_length_pearson,
        "quality_length_spearman": quality_length_spearman,
        "length_controlled_quality_pearson": length_controlled_quality_pearson,
        "feature_projection_correlations": {
            name: correlation(projection, standardized[:, index])
            for index, name in enumerate(FEATURE_NAMES)
        },
        "cluster_centers": centers.tolist(),
        "cluster_sizes": [int(np.sum(labels == cluster)) for cluster in range(2)],
        "cluster_quality_means": cluster_quality_means,
        "cluster_response_length_means": [
            float(response_lengths[labels == cluster].mean()) for cluster in range(2)
        ],
        "cluster_quality_gap": cluster_quality_means[1] - cluster_quality_means[0],
        "silhouette_1d": silhouette_1d(projection, labels),
        "extreme_quartile_count": int(len(extreme_indices)),
        "bottom_quartile_score_range": [
            float(quality[low_indices].min()),
            float(quality[low_indices].max()),
        ],
        "top_quartile_score_range": [
            float(quality[high_indices].min()),
            float(quality[high_indices].max()),
        ],
        "extreme_quartile_accuracy": extreme_accuracy,
        "extreme_quartile_auc": extreme_auc,
        "bootstrap_runs_completed": len(bootstrap_cosines),
        "bootstrap_direction_cosine_mean": float(np.mean(bootstrap_cosines)),
        "bootstrap_direction_cosine_p05": float(np.quantile(bootstrap_cosines, 0.05)),
        "boundary": boundary,
        "active_fraction_at_calibration": float(np.mean(projection < boundary)),
    }
    Path(args.output).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    Path(args.report).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
