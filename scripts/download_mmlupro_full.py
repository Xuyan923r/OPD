#!/usr/bin/env python3
import argparse
import json
import string
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset


def _build_prompt(question: str, options: list[str]) -> str:
    letters = string.ascii_uppercase
    option_lines = [f"{letters[i]}. {opt}" for i, opt in enumerate(options)]
    return (
        f"{question}\n\nOptions:\n"
        + "\n".join(option_lines)
        + "\n\nPlease answer with the option letter only."
    )


def _normalize_answer_letter(item: dict, options: list[str]) -> tuple[str, int]:
    letters = list(string.ascii_uppercase[: len(options)])
    answer = item.get("answer")
    answer_index = item.get("answer_index")

    if isinstance(answer, str):
        ans = answer.strip().upper()
        if len(ans) == 1 and ans in letters:
            return ans, letters.index(ans)
        if ans.isdigit():
            idx = int(ans)
            if 0 <= idx < len(options):
                return letters[idx], idx
        for idx, opt in enumerate(options):
            if ans == str(opt).strip().upper():
                return letters[idx], idx

    if isinstance(answer, (int, float)):
        idx = int(answer)
        if 0 <= idx < len(options):
            return letters[idx], idx

    if isinstance(answer_index, (int, float)):
        idx = int(answer_index)
        if 0 <= idx < len(options):
            return letters[idx], idx

    raise ValueError(f"Cannot normalize answer: answer={answer!r}, answer_index={answer_index!r}")


def _iter_records(ds: Dataset, split_name: str):
    for idx, item in enumerate(ds):
        question = str(item.get("question", "")).strip()
        options = item.get("options")
        if not question or not isinstance(options, list) or len(options) == 0:
            continue

        answer_letter, answer_index = _normalize_answer_letter(item, options)
        prompt = _build_prompt(question, [str(x) for x in options])

        yield {
            "prompt": prompt,
            "answer": answer_letter,
            "answer_index": answer_index,
            "question_id": item.get("question_id", idx),
            "category": item.get("category"),
            "source": item.get("src", item.get("source")),
            "question": question,
            "options": [str(x) for x in options],
            "split": split_name,
            "teacher_prompt": f"{prompt}\n\nAnswer: {answer_letter}",
        }


def main():
    parser = argparse.ArgumentParser(description="Download and convert full MMLU-Pro dataset to local JSONL.")
    parser.add_argument("--dataset", type=str, default="TIGER-Lab/MMLU-Pro")
    parser.add_argument(
        "--splits",
        type=str,
        default="test",
        help="Comma-separated split names, e.g. test or validation,test",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/idfsdata/yexuyan/OPD/MMLU-Pro-full-opsd-answer-only.jsonl",
    )
    args = parser.parse_args()

    split_names = [s.strip() for s in args.splits.split(",") if s.strip()]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ds_obj = load_dataset(args.dataset)
    if isinstance(ds_obj, Dataset):
        ds_dict = DatasetDict({"default": ds_obj})
    else:
        ds_dict = ds_obj

    missing = [s for s in split_names if s not in ds_dict]
    if missing:
        raise ValueError(f"Missing splits in dataset: {missing}. Available: {list(ds_dict.keys())}")

    total = 0
    with output_path.open("w", encoding="utf-8") as f:
        for split_name in split_names:
            for rec in _iter_records(ds_dict[split_name], split_name):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total += 1

    print(f"Saved {total} records to {output_path}")


if __name__ == "__main__":
    main()
