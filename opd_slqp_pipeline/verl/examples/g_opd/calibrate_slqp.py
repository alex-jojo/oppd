#!/usr/bin/env python3
"""Fit the frozen four-feature SLQP calibration for one exact student checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

FEATURE_NAMES = (
    "response_prompt_distance_mean",
    "response_prompt_distance_std",
    "response_step_distance_mean",
    "response_step_distance_std",
)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Exact student checkpoint used for SLQP training")
    parser.add_argument("--input", required=True, help="JSONL with quality_score and token ids or prompt/response")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def load_rows(path: Path, tokenizer, max_length: int) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "quality_score" not in row:
                raise ValueError(f"Line {line_number} is missing quality_score")
            if "prompt_token_ids" in row and "response_token_ids" in row:
                prompt_ids = [int(x) for x in row["prompt_token_ids"]]
                response_ids = [int(x) for x in row["response_token_ids"]]
            elif "prompt" in row and "response" in row:
                prompt_ids = tokenizer.encode(row["prompt"], add_special_tokens=True)
                response_ids = tokenizer.encode(row["response"], add_special_tokens=False)
            else:
                raise ValueError(
                    f"Line {line_number} needs prompt_token_ids/response_token_ids or prompt/response"
                )
            if not prompt_ids or not response_ids:
                raise ValueError(f"Line {line_number} has an empty prompt or response")
            if len(prompt_ids) >= max_length:
                raise ValueError(f"Line {line_number} prompt alone reaches max_length={max_length}")
            response_ids = response_ids[: max_length - len(prompt_ids)]
            rows.append(
                {
                    "id": row.get("id", line_number),
                    "prompt_ids": prompt_ids,
                    "response_ids": response_ids,
                    "quality_score": float(row["quality_score"]),
                }
            )
    if len(rows) < 10:
        raise ValueError("SLQP calibration needs at least 10 labeled responses")
    return rows


def trajectory_feature(
    hidden: torch.Tensor,
    prompt_length: int,
    response_token_ids: list[int],
    excluded_special_token_ids: set[int],
) -> np.ndarray:
    hidden = hidden.float()
    prompt_end = hidden[prompt_length - 1]
    kept_positions = [
        index for index, token_id in enumerate(response_token_ids)
        if token_id not in excluded_special_token_ids
    ]
    if len(kept_positions) < 2:
        raise ValueError("SLQP calibration response has fewer than two non-special generated tokens")
    response_all = hidden[prompt_length : prompt_length + len(response_token_ids)]
    response = response_all[torch.tensor(kept_positions, dtype=torch.long, device=hidden.device)]
    scale = hidden.size(-1) ** 0.5
    distance = torch.linalg.vector_norm(response - prompt_end, dim=-1) / scale
    previous = torch.cat((prompt_end.unsqueeze(0), response[:-1]), dim=0)
    step = torch.linalg.vector_norm(response - previous, dim=-1) / scale
    return np.asarray(
        [
            distance.mean().item(),
            distance.std(unbiased=False).item(),
            step.mean().item(),
            step.std(unbiased=False).item(),
        ],
        dtype=np.float64,
    )


def kmeans_two(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centers = np.quantile(values, [0.25, 0.75]).astype(np.float64)
    for _ in range(100):
        labels = np.argmin(np.abs(values[:, None] - centers[None, :]), axis=1)
        updated = np.asarray(
            [values[labels == i].mean() if np.any(labels == i) else centers[i] for i in range(2)]
        )
        if np.allclose(updated, centers, atol=1e-10, rtol=0):
            break
        centers = updated
    order = np.argsort(centers)
    centers = centers[order]
    labels = np.asarray([int(np.where(order == label)[0][0]) for label in labels])
    return centers, labels


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    rows = load_rows(Path(args.input), tokenizer, args.max_length)
    excluded_special_token_ids = {int(token_id) for token_id in tokenizer.all_special_ids}

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=args.trust_remote_code,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    features = []

    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        sequences = [row["prompt_ids"] + row["response_ids"] for row in batch]
        max_len = max(map(len, sequences))
        input_ids = torch.full(
            (len(batch), max_len), tokenizer.pad_token_id, dtype=torch.long, device=device
        )
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long, device=device)
        for index, sequence in enumerate(sequences):
            input_ids[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
            attention_mask[index, : len(sequence)] = 1
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
        final_hidden = output.hidden_states[-1]
        for index, row in enumerate(batch):
            features.append(
                trajectory_feature(
                    final_hidden[index],
                    len(row["prompt_ids"]),
                    row["response_ids"],
                    excluded_special_token_ids,
                )
            )
        print(json.dumps({"calibrated": min(start + len(batch), len(rows)), "total": len(rows)}), flush=True)

    matrix = np.stack(features)
    scores = np.asarray([row["quality_score"] for row in rows], dtype=np.float64)
    feature_mean = matrix.mean(axis=0)
    feature_std = matrix.std(axis=0)
    if np.any(feature_std < 1e-8):
        raise ValueError(f"Degenerate SLQP feature standard deviation: {feature_std.tolist()}")
    standardized = (matrix - feature_mean) / feature_std
    _, _, vh = np.linalg.svd(standardized, full_matrices=False)
    direction = vh[0]
    raw_quality = standardized @ direction
    correlation = np.corrcoef(raw_quality, scores)[0, 1]
    if not np.isfinite(correlation) or abs(correlation) < 1e-6:
        raise ValueError("Cannot orient PCA direction because it is uncorrelated with quality_score")
    if correlation < 0:
        direction = -direction
    quality = standardized @ direction
    centers, labels = kmeans_two(quality)
    boundary = float(centers.mean())
    predicted_good = labels == 1
    target_good = scores >= np.median(scores)

    payload = {
        "version": 2,
        "student_model": args.model,
        "hidden_size": int(model.config.hidden_size),
        "feature_names": list(FEATURE_NAMES),
        "excluded_special_token_ids": sorted(excluded_special_token_ids),
        "token_inclusion_rule": "generated response tokens excluding tokenizer special tokens",
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "direction": direction.tolist(),
        "good_boundary": boundary,
        "cluster_centers": centers.tolist(),
        "calibration_samples": len(rows),
        "quality_score_correlation": float(np.corrcoef(quality, scores)[0, 1]),
        "median_split_agreement": float(np.mean(predicted_good == target_good)),
        "input_file": str(Path(args.input).resolve()),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
