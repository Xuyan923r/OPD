#!/usr/bin/env python3
import argparse
import json
import random
import re
import string
from datetime import datetime
from pathlib import Path

import datasets
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


ALL_LETTERS = list(string.ascii_uppercase[:10])


def _model_suffix(model_path: str, n: int = 20) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(model_path))
    return safe[-n:] if len(safe) > n else safe


def extract_last_boxed(text: str):
    pattern = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
    matches = list(re.finditer(pattern, text))
    if matches:
        return matches[-1].group(1)
    return None


def extract_last_final_answer(text: str):
    pattern1 = r"Final\s*Answer\s*:\s*((?:[^<]|<[^<])*?)\n"
    pattern2 = r"The\s*answer\s*is\s*:\s*((?:[^<]|<[^<])*?)\n"
    matches1 = list(re.finditer(pattern1, text, flags=re.IGNORECASE))
    matches2 = list(re.finditer(pattern2, text, flags=re.IGNORECASE))
    if matches1:
        return matches1[-1].group(1)
    if matches2:
        return matches2[-1].group(1)
    return None


def extract_solution(solution_str: str):
    if "<|im_start|>user" in solution_str:
        model_output = re.sub(
            r"^.*?<\|im_start\|>assistant",
            "<|im_start|>assistant",
            solution_str,
            flags=re.DOTALL,
            count=1,
        )
    elif "Assistant:" in solution_str:
        model_output = solution_str.split("Assistant:")[-1].strip()
    else:
        model_output = solution_str

    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
    for stop_word in stop_words:
        if stop_word in model_output:
            model_output = model_output.split(stop_word)[0].strip()

    boxed_answer = extract_last_boxed(model_output)
    if boxed_answer:
        return boxed_answer
    return extract_last_final_answer(model_output)


def form_options(options: list[str]):
    option_str = "Options are:\n"
    for text, letter in zip(options, ALL_LETTERS, strict=False):
        option_str += f"({letter}): {text}\n"
    return option_str


def get_prediction(output: str):
    solution = extract_solution(output or "")
    if solution is None:
        return random.choice(ALL_LETTERS)
    for option in ALL_LETTERS:
        if option in solution:
            return option
    return random.choice(ALL_LETTERS)


def load_records(dataset_path: str | None):
    if dataset_path:
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    ds = datasets.load_dataset("TIGER-Lab/MMLU-Pro")
    return list(ds["test"])


def normalize_record(entry: dict):
    question = entry.get("question")
    options = entry.get("options")
    answer = entry.get("answer")
    category = entry.get("category", "other")

    if question is None or options is None:
        # Fallback to prebuilt prompt-only jsonl format.
        prompt = entry.get("prompt")
        if not isinstance(prompt, str):
            raise ValueError("Record must contain either (question, options) or prompt.")
        return {
            "question": prompt,
            "options": [],
            "answer": str(answer).strip().upper() if answer is not None else "",
            "category": category,
            "raw": entry,
        }

    return {
        "question": str(question),
        "options": [str(x) for x in options],
        "answer": str(answer).strip().upper(),
        "category": category,
        "raw": entry,
    }


def build_prompt(tokenizer, question: str, options: list[str]):
    if options:
        query = question + "\n" + form_options(options) + "\n"
    else:
        query = question + "\n"
    instruction = (
        "Please reason step by step, and put your final answer option within \\boxed{}. "
        "Only put the option letter in the box, e.g. \\boxed{A}. "
        "There is only one correct answer."
    )
    messages = [{"role": "user", "content": query + "\n" + instruction}]
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "user: " + query + "\n" + instruction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="/idfsdata/yexuyan/OPD/MMLU-Pro-full-opsd-answer-only.jsonl")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--summary_file", type=str, default=None)
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--max_num_seqs", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    out_dir = Path("/idfsdata/yexuyan/OPD/evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    if not args.output_file:
        args.output_file = str(out_dir / f"outputs_mmlupro_{_model_suffix(args.model_path)}_{ts}.json")
    if not args.summary_file:
        args.summary_file = str(out_dir / f"mmlupro_summary_{_model_suffix(args.model_path)}_{ts}.json")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        trust_remote_code=True,
    )

    raw_records = load_records(args.dataset_path)
    records = [normalize_record(x) for x in raw_records]
    categories = sorted({r["category"] for r in records})
    per_category = {c: [0, 0] for c in categories}  # [correct, total]

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    outputs_all = []
    success = 0
    total = 0

    print("----- Start MMLU-Pro evaluation -----")
    print(f"records={len(records)} categories={len(categories)}")

    for category in categories:
        category_entries = [entry for entry in records if entry["category"] == category]
        cat_outputs = []
        for i in range(0, len(category_entries), args.batch_size):
            batch = category_entries[i : i + args.batch_size]
            prompts = [build_prompt(tokenizer, b["question"], b["options"]) for b in batch]
            batch_outputs = llm.generate(prompts, sampling_params)

            for entry, out in zip(batch, batch_outputs, strict=False):
                text = out.outputs[0].text if out.outputs else ""
                pred = get_prediction(text)
                gt = entry["answer"]
                is_correct = pred == gt
                success += int(is_correct)
                total += 1
                per_category[category][0] += int(is_correct)
                per_category[category][1] += 1

                item = dict(entry["raw"])
                item["solution"] = text
                item["prediction"] = pred
                item["correct"] = bool(is_correct)
                cat_outputs.append(item)
                outputs_all.append(item)

        correct, cnt = per_category[category]
        acc = correct / cnt if cnt else 0.0
        print(f"{category}: {acc:.4f} ({correct}/{cnt})")

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(outputs_all, f, indent=2, ensure_ascii=False)

    report = {}
    valid_accs = []
    print("\n----- Accuracy Report -----")
    for category in categories:
        correct, cnt = per_category[category]
        acc = correct / cnt if cnt else 0.0
        report[category] = {
            "correct": correct,
            "total": cnt,
            "accuracy": acc,
        }
        if cnt:
            valid_accs.append(acc)
        print(f"{category}: {correct}/{cnt} -> {acc*100:.2f}%")

    micro = success / total if total else 0.0
    macro = sum(valid_accs) / len(valid_accs) if valid_accs else 0.0
    print(f"\nMicro Average Accuracy: {micro*100:.2f}%")
    print(f"Macro Average Accuracy: {macro*100:.2f}%")

    summary = {
        "dataset": "mmlupro",
        "model": args.model_path,
        "records": total,
        "micro_accuracy": micro,
        "macro_accuracy": macro,
        "per_category": report,
        "output_file": args.output_file,
    }
    with open(args.summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open("/idfsdata/yexuyan/OPD/evaluation/final_results.jsonl", "a", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": "mmlupro",
                "model": args.model_path,
                "micro_accuracy_pct": round(micro * 100, 2),
                "macro_accuracy_pct": round(macro * 100, 2),
                "timestamp_utc": ts,
                "summary_file": args.summary_file,
            },
            f,
            ensure_ascii=False,
        )
        f.write("\n")


if __name__ == "__main__":
    main()
