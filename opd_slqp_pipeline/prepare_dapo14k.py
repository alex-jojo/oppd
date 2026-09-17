#!/usr/bin/env python3
"""Download guanning-ai/dapo14k and write the exact VERL RL parquet schema."""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Dataset, load_dataset


REQUIRED_COLUMNS = {"data_source", "prompt", "ability", "reward_model", "extra_info"}
SOURCE_DATASET = "guanning-ai/dapo14k"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def validate_existing(path: Path) -> int:
    metadata = pq.read_metadata(path)
    columns = set(metadata.schema.names)
    # Nested Arrow leaves do not necessarily expose the top-level names, so
    # perform the authoritative schema check through datasets below.
    dataset = load_dataset("parquet", data_files=str(path), split="train")
    missing = REQUIRED_COLUMNS.difference(dataset.column_names)
    if missing:
        raise ValueError(f"Prepared parquet is missing columns: {sorted(missing)}; Arrow leaves={sorted(columns)}")
    if len(dataset) <= 0:
        raise ValueError("Prepared dapo14k parquet is empty")
    return len(dataset)


def main():
    args = parse_args()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.is_file():
        rows = validate_existing(output)
        print(json.dumps({"phase": "reuse_dapo14k", "path": str(output), "rows": rows}))
        return

    source = load_dataset(SOURCE_DATASET, split="train")
    required_source = {"problem", "answer"}
    missing_source = required_source.difference(source.column_names)
    if missing_source:
        raise ValueError(
            f"{SOURCE_DATASET} is missing required columns {sorted(missing_source)}; "
            f"found {source.column_names}"
        )

    rows = []
    for index, item in enumerate(source):
        problem = str(item["problem"])
        answer = str(item["answer"])
        rows.append(
            {
                "data_source": "math_dapo",
                "prompt": [{"role": "user", "content": problem}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "index": index,
                    "source_dataset": SOURCE_DATASET,
                    "source_datasource": str(item.get("datasource", "dapo14k")),
                },
            }
        )

    prepared = Dataset.from_list(rows)
    temporary = output.with_suffix(output.suffix + ".tmp")
    prepared.to_parquet(str(temporary))
    temporary.replace(output)
    count = validate_existing(output)
    if count != len(source):
        raise RuntimeError(f"Row-count mismatch after conversion: source={len(source)}, output={count}")
    print(
        json.dumps(
            {
                "phase": "prepared_dapo14k",
                "source": SOURCE_DATASET,
                "path": str(output),
                "rows": count,
                "columns": prepared.column_names,
            }
        )
    )


if __name__ == "__main__":
    main()
