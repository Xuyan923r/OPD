#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def recover_teacher_log_probs(record: dict) -> list[float | None]:
    reward = record.get("reward")
    if not isinstance(reward, dict):
        return []

    teacher_output = reward.get("teacher_output")
    if not isinstance(teacher_output, dict):
        return []

    meta_info = teacher_output.get("meta_info")
    if not isinstance(meta_info, dict):
        return []

    input_token_logprobs = meta_info.get("input_token_logprobs")
    if not isinstance(input_token_logprobs, list):
        return []

    recovered: list[float | None] = []
    for item in input_token_logprobs[1:]:
        logprob = None
        if isinstance(item, (list, tuple)) and len(item) > 0:
            logprob = item[0]
        elif isinstance(item, dict):
            logprob = item.get("logprob")
        recovered.append(None if logprob is None else float(logprob))

    response_length = int(record.get("response_length") or 0)
    if response_length > 0:
        recovered = recovered[-response_length:]
    return recovered


def rebuild_student_minus_teacher(
    student_log_probs: list[float | None], teacher_log_probs: list[float | None], length: int
) -> list[float | None]:
    result: list[float | None] = []
    for i in range(length):
        if (
            i >= len(student_log_probs)
            or i >= len(teacher_log_probs)
            or student_log_probs[i] is None
            or teacher_log_probs[i] is None
        ):
            result.append(None)
        else:
            result.append(float(student_log_probs[i]) - float(teacher_log_probs[i]))
    return result


def process_file(input_path: Path, output_path: Path) -> tuple[int, int]:
    updated = 0
    total = 0

    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            total += 1
            record = json.loads(line)

            teacher_log_probs = record.get("teacher_log_probs") or []
            if not teacher_log_probs:
                teacher_log_probs = recover_teacher_log_probs(record)
                if teacher_log_probs:
                    record["teacher_log_probs"] = teacher_log_probs
                    student_log_probs = record.get("student_log_probs") or []
                    response_token_ids = record.get("response_token_ids") or []
                    diff = rebuild_student_minus_teacher(
                        student_log_probs=student_log_probs,
                        teacher_log_probs=teacher_log_probs,
                        length=len(response_token_ids),
                    )
                    record["student_minus_teacher_log_probs"] = diff
                    record["student_teacher_token_score_diff"] = diff
                    updated += 1

            dst.write(json.dumps(record, ensure_ascii=False) + "\n")

    return updated, total


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill teacher_log_probs into saved rollout trajectories.")
    parser.add_argument("--input", required=True, help="Input rollout jsonl file or directory.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for backfilled files. Defaults to a sibling '*_backfilled' directory.",
    )
    parser.add_argument(
        "--glob",
        default="rollout_*.jsonl",
        help="Glob pattern when --input is a directory. Default: rollout_*.jsonl",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_dir():
        files = sorted(input_path.glob(args.glob))
        if not files:
            raise FileNotFoundError(f"No files matched {args.glob} under {input_path}")
        output_dir = Path(args.output_dir) if args.output_dir else input_path.with_name(f"{input_path.name}_backfilled")
        output_dir.mkdir(parents=True, exist_ok=True)
        grand_updated = 0
        grand_total = 0
        for file_path in files:
            updated, total = process_file(file_path, output_dir / file_path.name)
            grand_updated += updated
            grand_total += total
            print(f"{file_path.name}: updated {updated}/{total}")
        print(f"Done. Updated {grand_updated}/{grand_total} records into {output_dir}")
        return

    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent.with_name(f"{input_path.parent.name}_backfilled")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / input_path.name
    updated, total = process_file(input_path, output_path)
    print(f"Done. Updated {updated}/{total} records into {output_path}")


if __name__ == "__main__":
    main()
