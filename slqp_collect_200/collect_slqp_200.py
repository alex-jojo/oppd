#!/usr/bin/env python3
"""Collect 200 Qwen3 student responses and every response-token final hidden state."""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path

import numpy as np


FEATURE_NAMES = (
    "response_prompt_distance_mean",
    "response_prompt_distance_std",
    "response_step_distance_mean",
    "response_step_distance_std",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("generate", "replay", "package"), required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--data", required=True, help="Local parquet or JSON/JSONL math dataset")
    parser.add_argument("--output", default="slqp_capture_200")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--generation-batch-size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def read_existing(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    rows = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row["id"]] = row
    return rows


def rewrite_rows(path: Path, rows) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")
    os.replace(temporary, path)


def load_dataset_rows(path: str):
    from datasets import load_dataset

    local_path = Path(path)
    suffix = local_path.suffix.lower()
    if local_path.is_file() and suffix == ".parquet":
        return load_dataset("parquet", data_files=path, split="train")
    if local_path.is_file() and suffix in {".json", ".jsonl"}:
        return load_dataset("json", data_files=path, split="train")
    if not local_path.exists():
        return load_dataset(path, split="train")
    raise ValueError("--data must be a Hugging Face dataset id or local parquet/JSON file")


def extract_problem(row: dict):
    for key in ("prompt", "messages", "question", "problem", "query"):
        if key in row and row[key] is not None:
            value = row[key]
            return value.tolist() if isinstance(value, np.ndarray) else value
    raise KeyError(f"Could not find a prompt field in columns: {sorted(row)}")


def render_prompt(problem, tokenizer) -> str:
    if isinstance(problem, list):
        messages = problem
    elif isinstance(problem, dict) and "content" in problem:
        messages = [problem]
    else:
        messages = [{"role": "user", "content": str(problem)}]
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    return str(problem).rstrip() + "\n"


def selected_indices(length: int, samples: int, seed: int) -> list[int]:
    if samples > length:
        raise ValueError(f"Requested {samples} samples from a dataset containing only {length}")
    return np.random.default_rng(seed).permutation(length).tolist()


def generate(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    responses_path = output_dir / "responses.jsonl"
    existing = read_existing(responses_path)
    rejected = read_existing(output_dir / "rejected_responses.jsonl")
    if len(existing) > args.samples:
        raise RuntimeError(f"Found {len(existing)} generated rows, expected at most {args.samples}")
    if len(existing) == args.samples:
        print(
            json.dumps({"phase": "generate", "collected": len(existing), "target": args.samples, "resumed": True}),
            flush=True,
        )
        return
    dataset = load_dataset_rows(args.data)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)

    pending = []
    for dataset_index in selected_indices(len(dataset), args.samples, args.seed):
        sample_id = f"sample_{dataset_index:07d}"
        if sample_id in existing or sample_id in rejected:
            continue
        raw = dataset[int(dataset_index)]
        problem = extract_problem(raw)
        prompt = render_prompt(problem, tokenizer)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(prompt_ids) > args.max_prompt_tokens:
            continue
        pending.append((sample_id, int(dataset_index), problem, prompt))
        if len(pending) + len(existing) >= args.samples:
            break
    if len(pending) + len(existing) < args.samples:
        raise RuntimeError(
            f"Only found {len(pending) + len(existing)} usable prompts; increase the candidate pool or prompt limit"
        )

    engine = LLM(
        model=args.model,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_prompt_tokens + args.max_new_tokens,
        seed=args.seed,
    )
    sampling = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )
    mode = "a" if responses_path.exists() else "w"
    with responses_path.open(mode, encoding="utf-8") as handle:
        for start in range(0, len(pending), args.generation_batch_size):
            batch = pending[start : start + args.generation_batch_size]
            outputs = engine.generate([item[3] for item in batch], sampling, use_tqdm=True)
            for metadata, output in zip(batch, outputs, strict=True):
                sample_id, dataset_index, problem, prompt = metadata
                candidate = output.outputs[0]
                row = {
                    "id": sample_id,
                    "dataset_index": dataset_index,
                    "problem": problem,
                    "prompt": prompt,
                    "prompt_token_ids": [int(x) for x in output.prompt_token_ids],
                    "response": candidate.text,
                    "response_token_ids": [int(x) for x in candidate.token_ids],
                    "finish_reason": candidate.finish_reason,
                    "model": args.model,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                }
                handle.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")
                handle.flush()
                existing[sample_id] = row
            print(
                json.dumps({"phase": "generate", "collected": len(existing), "target": args.samples}),
                flush=True,
            )


def trajectory_features(hidden, prompt_length: int, response_ids: list[int], excluded_ids: set[int]):
    import torch

    prompt_end = hidden[prompt_length - 1].float()
    response_all = hidden[prompt_length : prompt_length + len(response_ids)].float()
    kept_positions = np.asarray(
        [index for index, token_id in enumerate(response_ids) if token_id not in excluded_ids],
        dtype=np.int32,
    )
    if len(kept_positions) < 2:
        raise RuntimeError("A collected response has fewer than two non-special generated tokens")
    kept_index = torch.tensor(kept_positions, dtype=torch.long, device=hidden.device)
    response = response_all[kept_index]
    scale = hidden.size(-1) ** 0.5
    distance = torch.linalg.vector_norm(response - prompt_end, dim=-1) / scale
    previous = torch.cat((prompt_end.unsqueeze(0), response[:-1]), dim=0)
    step = torch.linalg.vector_norm(response - previous, dim=-1) / scale
    features = np.asarray(
        [
            distance.mean().item(),
            distance.std(unbiased=False).item(),
            step.mean().item(),
            step.std(unbiased=False).item(),
        ],
        dtype=np.float32,
    )
    return features, kept_positions


def replay(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_dir = Path(args.output)
    rows = list(read_existing(output_dir / "responses.jsonl").values())
    if len(rows) != args.samples:
        raise RuntimeError(f"Expected {args.samples} generated rows, found {len(rows)}")
    hidden_dir = output_dir / "hidden"
    hidden_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    excluded_special_token_ids = {int(token_id) for token_id in tokenizer.all_special_ids}

    invalid_rows = [
        row
        for row in rows
        if sum(int(token_id) not in excluded_special_token_ids for token_id in row["response_token_ids"]) < 2
    ]
    if invalid_rows:
        rejected_path = output_dir / "rejected_responses.jsonl"
        rejected = read_existing(rejected_path)
        new_rejections = [row for row in invalid_rows if row["id"] not in rejected]
        if new_rejections:
            mode = "a" if rejected_path.exists() else "w"
            with rejected_path.open(mode, encoding="utf-8") as handle:
                for row in new_rejections:
                    row = dict(row)
                    row["rejection_reason"] = "fewer_than_two_non_special_generated_tokens"
                    handle.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")
        invalid_ids = {row["id"] for row in invalid_rows}
        rewrite_rows(output_dir / "responses.jsonl", [row for row in rows if row["id"] not in invalid_ids])
        for sample_id in invalid_ids:
            stale_hidden = hidden_dir / f"{sample_id}.npz"
            if stale_hidden.exists():
                stale_hidden.unlink()
        print(
            json.dumps(
                {
                    "phase": "reject_and_refill",
                    "rejected": len(invalid_rows),
                    "remaining_valid_responses": len(rows) - len(invalid_rows),
                    "next": "rerun generation and replay automatically",
                }
            ),
            flush=True,
        )
        raise SystemExit(75)

    active_ids = {row["id"] for row in rows}
    for hidden_path in hidden_dir.glob("*.npz"):
        if hidden_path.stem not in active_ids:
            hidden_path.unlink()

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=args.trust_remote_code,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    backbone = getattr(model, "model", None)
    if backbone is None:
        raise RuntimeError("Expected a Hugging Face causal LM with a .model backbone")

    hidden_size = int(model.config.hidden_size)

    manifest_rows = []
    for index, row in enumerate(rows, 1):
        hidden_path = hidden_dir / f"{row['id']}.npz"
        if hidden_path.is_file():
            with np.load(hidden_path) as saved:
                feature_version = int(saved["trajectory_feature_version"]) if "trajectory_feature_version" in saved else 1
                if feature_version == 2:
                    features = saved["trajectory_features"].astype(np.float32).tolist()
                    response_length = int(saved["response_length"])
                    trajectory_response_length = int(saved["trajectory_response_length"])
                    manifest_rows.append(
                        {
                            "id": row["id"],
                            "response_length": response_length,
                            "trajectory_response_length": trajectory_response_length,
                            "trajectory_features": features,
                        }
                    )
                    continue

        prompt_ids = row["prompt_token_ids"]
        response_ids = row["response_token_ids"]
        if not response_ids:
            raise RuntimeError(f"{row['id']} has an empty response")
        all_ids = prompt_ids + response_ids
        input_ids = torch.tensor(all_ids, dtype=torch.long, device=device).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            output = backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        hidden = output.last_hidden_state[0]
        prompt_length = len(prompt_ids)
        response_length = len(response_ids)
        response_hidden = hidden[prompt_length : prompt_length + response_length]
        prompt_end = hidden[prompt_length - 1]
        features, kept_positions = trajectory_features(
            hidden, prompt_length, response_ids, excluded_special_token_ids
        )
        kept_index = torch.tensor(kept_positions, dtype=torch.long, device=device)
        trajectory_hidden = response_hidden[kept_index]
        trajectory_mask = np.zeros(response_length, dtype=np.bool_)
        trajectory_mask[kept_positions] = True
        np.savez(
            hidden_path,
            response_hidden=response_hidden.to(torch.float16).cpu().numpy(),
            prompt_end_hidden=prompt_end.to(torch.float16).cpu().numpy(),
            response_mean_hidden=response_hidden.float().mean(dim=0).to(torch.float16).cpu().numpy(),
            response_last_hidden=response_hidden[-1].to(torch.float16).cpu().numpy(),
            trajectory_mean_hidden=trajectory_hidden.float().mean(dim=0).to(torch.float16).cpu().numpy(),
            trajectory_last_hidden=trajectory_hidden[-1].to(torch.float16).cpu().numpy(),
            response_token_ids=np.asarray(response_ids, dtype=np.int32),
            response_positions=np.arange(response_length, dtype=np.int32),
            trajectory_mask=trajectory_mask,
            trajectory_features=features,
            trajectory_feature_version=np.asarray(2, dtype=np.int32),
            prompt_length=np.asarray(prompt_length, dtype=np.int32),
            response_length=np.asarray(response_length, dtype=np.int32),
            trajectory_response_length=np.asarray(len(kept_positions), dtype=np.int32),
        )
        manifest_rows.append(
            {
                "id": row["id"],
                "response_length": response_length,
                "trajectory_response_length": len(kept_positions),
                "trajectory_features": features.tolist(),
            }
        )
        del output, hidden, response_hidden, trajectory_hidden, input_ids, attention_mask
        if index % 10 == 0 or index == len(rows):
            print(
                json.dumps({"phase": "replay", "completed": index, "target": len(rows)}),
                flush=True,
            )

    manifest = {
        "version": 2,
        "model": args.model,
        "hidden_size": hidden_size,
        "samples": len(rows),
        "max_new_tokens": args.max_new_tokens,
        "feature_names": list(FEATURE_NAMES),
        "excluded_special_token_ids": sorted(excluded_special_token_ids),
        "token_inclusion_rule": "generated response tokens excluding tokenizer special tokens",
        "hidden_dtype": "float16",
        "hidden_coverage": "every response token",
        "contains": [
            "response text and token ids",
            "prompt token ids",
            "every response-token final hidden",
            "prompt-end hidden",
            "response mean hidden",
            "response last-token hidden",
            "special-token-excluded trajectory mean/last hidden and mask",
            "four SLQP trajectory features",
        ],
        "rows": manifest_rows,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def package(args: argparse.Namespace) -> None:
    output_dir = Path(args.output)
    responses_path = output_dir / "responses.jsonl"
    rejected_path = output_dir / "rejected_responses.jsonl"
    manifest_path = output_dir / "manifest.json"
    hidden_dir = output_dir / "hidden"
    if not responses_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("Generation or replay is incomplete")
    hidden_files = sorted(hidden_dir.glob("*.npz"))
    if len(hidden_files) != args.samples:
        raise RuntimeError(f"Expected {args.samples} hidden files, found {len(hidden_files)}")

    archive = output_dir / "download.zip"
    temporary = output_dir / "download.tmp.zip"
    if temporary.exists():
        temporary.unlink()
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as bundle:
        bundle.write(responses_path, "responses.jsonl")
        bundle.write(manifest_path, "manifest.json")
        if rejected_path.is_file():
            bundle.write(rejected_path, "rejected_responses.jsonl")
        for path in hidden_files:
            bundle.write(path, f"hidden/{path.name}")
    archive_bytes = temporary.stat().st_size
    os.replace(temporary, archive)
    complete = {
        "samples": args.samples,
        "archive": str(archive.resolve()),
        "archive_bytes": archive_bytes,
    }
    (output_dir / "COMPLETE.json").write_text(
        json.dumps(complete, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(complete, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    if args.samples != 200:
        print(json.dumps({"warning": "This experiment was designed for 200 samples"}), flush=True)
    if args.stage == "generate":
        generate(args)
    elif args.stage == "replay":
        replay(args)
    else:
        package(args)


if __name__ == "__main__":
    main()
