#!/usr/bin/env python3
"""Prepare the 272-example Minerva-Math test set for VERL validation."""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Dataset, load_dataset


SOURCE_DATASET = "math-ai/minervamath"
SOURCE_SPLIT = "test"
EXPECTED_ROWS = 272
REQUIRED_COLUMNS = {"data_source", "prompt", "ability", "reward_model", "extra_info"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def validate_existing(path: Path) -> int:
    pq.read_metadata(path)
    dataset = load_dataset("parquet", data_files=str(path), split="train")
    missing = REQUIRED_COLUMNS.difference(dataset.column_names)
    if missing:
        raise ValueError(f"Prepared Minerva-Math parquet is missing columns: {sorted(missing)}")
    if len(dataset) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} Minerva-Math rows, found {len(dataset)}")
    return len(dataset)


def main():
    args = parse_args()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.is_file():
        rows = validate_existing(output)
        print(json.dumps({"phase": "reuse_minerva_math", "path": str(output), "rows": rows}))
        return

    source = load_dataset(SOURCE_DATASET, split=SOURCE_SPLIT)
    missing_source = {"question", "answer"}.difference(source.column_names)
    if missing_source:
        raise ValueError(
            f"{SOURCE_DATASET} is missing required columns {sorted(missing_source)}; "
            f"found {source.column_names}"
        )
    if len(source) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} source rows, found {len(source)}")

    rows = []
    for index, item in enumerate(source):
        question = str(item["question"]).strip()
        answer = str(item["answer"]).strip()
        prompt = question + "\nPlease reason step by step, and put your final answer within \\boxed{}."
        rows.append(
            {
                # Capitalized "Minerva" selects VERL's math_verify reward route.
                "data_source": "math-ai/Minerva-Math",
                "prompt": [{"role": "user", "content": prompt}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "index": index,
                    "source_dataset": SOURCE_DATASET,
                    "source_split": SOURCE_SPLIT,
                },
            }
        )

    prepared = Dataset.from_list(rows)
    temporary = output.with_suffix(output.suffix + ".tmp")
    prepared.to_parquet(str(temporary))
    temporary.replace(output)
    count = validate_existing(output)
    print(
        json.dumps(
            {
                "phase": "prepared_minerva_math",
                "source": SOURCE_DATASET,
                "path": str(output),
                "rows": count,
                "columns": prepared.column_names,
            }
        )
    )


if __name__ == "__main__":
    main()
