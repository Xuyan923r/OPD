import asyncio
import os
import aiohttp
import torch

from slime.rollout.rm_hub import grade_answer_verl
from slime.rollout.rm_hub.gpqa import (
    _extract_choices_from_prompt,
    _extract_letter_from_response,
    _prompt_to_text,
    _resolve_valid_letters,
)
from slime.rollout.rm_hub.math_utils import extract_answer
from slime.utils.types import Sample


_BOXED_PREFIX = r"\boxed"
_FINAL_ANSWER_PREFIX = "Final Answer:"

_TRAILING_SPECIAL_MARKERS = ("<|im_end|>", "</s>", "<|endoftext|>")

_TEACHER_REQUEST_SEMAPHORE: asyncio.Semaphore | None = None
_TEACHER_REQUEST_CONCURRENCY = max(
    1,
    int(
        os.getenv(
            "OPD_TEACHER_RM_CONCURRENCY",
            os.getenv("OPD_PI_TEACHER_RM_CONCURRENCY", "1"),
        )
    ),
)


def _get_teacher_request_semaphore() -> asyncio.Semaphore:
    global _TEACHER_REQUEST_SEMAPHORE
    if _TEACHER_REQUEST_SEMAPHORE is None:
        _TEACHER_REQUEST_SEMAPHORE = asyncio.Semaphore(_TEACHER_REQUEST_CONCURRENCY)
    return _TEACHER_REQUEST_SEMAPHORE



def _strip_trailing_special_markers(text: str) -> str:
    cleaned = text.rstrip()
    changed = True
    while changed:
        changed = False
        for marker in _TRAILING_SPECIAL_MARKERS:
            if cleaned.endswith(marker):
                cleaned = cleaned[: -len(marker)].rstrip()
                changed = True
    return cleaned


def _extract_braced_content(text: str, open_brace_index: int) -> tuple[str | None, int | None]:
    if open_brace_index >= len(text) or text[open_brace_index] != "{":
        return None, None

    depth = 0
    chars: list[str] = []
    for index in range(open_brace_index, len(text)):
        char = text[index]
        if char == "{":
            if depth > 0:
                chars.append(char)
            depth += 1
        elif char == "}":
            depth -= 1
            if depth < 0:
                return None, None
            if depth == 0:
                return "".join(chars).strip(), index + 1
            chars.append(char)
        elif depth > 0:
            chars.append(char)

    return None, None


def _extract_last_boxed_content(text: str) -> str | None:
    last_content = None
    search_start = 0
    needle = f"{_BOXED_PREFIX}{{"
    while True:
        boxed_index = text.find(needle, search_start)
        if boxed_index < 0:
            break

        content, end_index = _extract_braced_content(text, boxed_index + len(_BOXED_PREFIX))
        if content is not None and end_index is not None:
            last_content = content
            search_start = end_index
        else:
            search_start = boxed_index + len(needle)

    return last_content


def _extract_strict_final_line_boxed_content(text: str) -> str | None:
    text = _strip_trailing_special_markers(text)
    non_empty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not non_empty_lines:
        return None

    last_line = non_empty_lines[-1]
    if not last_line.startswith(_FINAL_ANSWER_PREFIX):
        return None

    remainder = last_line[len(_FINAL_ANSWER_PREFIX) :].strip()
    if not remainder.startswith(f"{_BOXED_PREFIX}{{"):
        return None

    content, end_index = _extract_braced_content(remainder, len(_BOXED_PREFIX))
    if content is None or end_index is None:
        return None

    if remainder[end_index:].strip():
        return None

    return content.strip()


def _extract_student_answer_info(sample: Sample) -> dict[str, str | bool | None]:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    label_text = str(sample.label).strip() if sample.label is not None else None
    response_text = sample.response or ""

    strict_boxed_answer = _extract_strict_final_line_boxed_content(response_text)
    last_boxed_answer = _extract_last_boxed_content(response_text)
    boxed_answer = strict_boxed_answer if strict_boxed_answer is not None else last_boxed_answer

    student_answer = None
    target_answer = None
    boxed_format_valid = False

    # Multiple-choice datasets such as the science set store the answer as a single option letter.
    if label_text and len(label_text) == 1 and label_text.isalpha():
        prompt_text = _prompt_to_text(sample.prompt)
        choices = metadata.get("choices")
        if isinstance(choices, dict):
            choices = list(choices.values())
        elif choices is not None:
            choices = list(choices)
        elif prompt_text:
            choices = _extract_choices_from_prompt(prompt_text) or None

        valid_letters = _resolve_valid_letters(
            metadata=metadata,
            choices=choices,
            label_text=label_text,
            prompt_text=prompt_text,
        )
        target_answer = label_text.upper()

        if boxed_answer is not None:
            normalized_boxed_answer = boxed_answer.strip().upper()
            if len(normalized_boxed_answer) == 1 and normalized_boxed_answer in valid_letters:
                student_answer = normalized_boxed_answer
            if strict_boxed_answer is not None and len(normalized_boxed_answer) == 1 and normalized_boxed_answer in valid_letters:
                boxed_format_valid = True

        if student_answer is None:
            extracted_letter = _extract_letter_from_response(response_text, valid_letters)
            if extracted_letter is not None:
                student_answer = extracted_letter.strip().upper()
    else:
        target_answer = label_text
        if target_answer and "\\boxed" in target_answer:
            extracted_target = extract_answer(target_answer, mode="auto")
            if extracted_target is not None:
                target_answer = extracted_target

        if boxed_answer is not None:
            student_answer = boxed_answer.strip() or None
            boxed_format_valid = strict_boxed_answer is not None and student_answer is not None

        if student_answer is None:
            extracted_answer = extract_answer(response_text, mode="auto")
            if extracted_answer is not None:
                student_answer = extracted_answer.strip()

        if target_answer is not None:
            target_answer = target_answer.strip()

    if isinstance(student_answer, str):
        student_answer = student_answer.strip() or None
    if isinstance(target_answer, str):
        target_answer = target_answer.strip() or None

    return {
        "student_extracted_answer": student_answer,
        "target_answer": target_answer,
        "student_answer_parseable": student_answer is not None,
        "student_boxed_answer": boxed_answer.strip() if isinstance(boxed_answer, str) and boxed_answer.strip() else None,
        "student_boxed_format_valid": bool(boxed_format_valid),
        "student_answer_correct": (
            student_answer is not None and target_answer is not None and student_answer == target_answer
        ),
    }


def _build_student_only_eval_reward(sample: Sample) -> dict[str, object]:
    answer_info = _extract_student_answer_info(sample)
    strict_correct = bool(answer_info["student_answer_correct"])
    return {
        "accuracy": 1.0 if strict_correct else 0.0,
        **answer_info,
        "student_answer_reward": 1.0 if strict_correct else 0.0,
    }


async def reward_func_student_only_eval(args, sample, **kwargs):
    return _build_student_only_eval_reward(sample)


def _extract_aligned_teacher_log_probs(teacher_output: dict | None, response_length: int) -> torch.Tensor:
    meta_info = teacher_output.get("meta_info") if isinstance(teacher_output, dict) else None
    input_token_logprobs = meta_info.get("input_token_logprobs") if isinstance(meta_info, dict) else None

    values: list[float | None] = []
    if isinstance(input_token_logprobs, list):
        for item in input_token_logprobs:
            logprob = None
            if isinstance(item, (list, tuple)) and len(item) > 0:
                logprob = item[0]
            elif isinstance(item, dict):
                logprob = item.get("logprob")
            values.append(None if logprob is None else float(logprob))

    if len(values) == response_length + 1 and values and values[0] is None:
        values = values[1:]
    elif len(values) > response_length:
        values = values[-response_length:]
    elif len(values) < response_length:
        values = [None] * (response_length - len(values)) + values

    return torch.tensor([
        0.0 if value is None else value for value in values
    ], dtype=torch.float32)


def _build_teacher_payload(sample: Sample) -> dict[str, object]:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    teacher_prompt_text = metadata.get("teacher_prompt_text")
    teacher_logprob_start_len = metadata.get("teacher_logprob_start_len")

    payload: dict[str, object] = {
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }

    if isinstance(teacher_prompt_text, str) and teacher_prompt_text.strip():
        # PI / privileged-prompt OPD: teacher sees its own prompt plus the exact student response.
        payload["text"] = teacher_prompt_text + (sample.response or "")
        # We only need log-probs aligned to the student response. Requesting log-probs
        # for the entire privileged prompt wastes memory and can OOM the teacher server.
        if isinstance(teacher_logprob_start_len, int) and teacher_logprob_start_len >= 0:
            payload["logprob_start_len"] = teacher_logprob_start_len
    else:
        payload["input_ids"] = sample.tokens

    return payload


async def reward_func(args, sample, **kwargs):
    payload = _build_teacher_payload(sample)
    session_kwargs = {}
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    teacher_prompt_text = metadata.get("teacher_prompt_text")

    async with aiohttp.ClientSession(**session_kwargs) as session:
        # Throttle all teacher RM requests, including eval requests that do not carry
        # privileged-prompt metadata, so step-0 validation cannot overwhelm the teacher server.
        async with _get_teacher_request_semaphore():
            async with session.post(args.rm_url, json=payload) as resp:
                resp.raise_for_status()
                teacher_output = await resp.json()

    answer_info = _extract_student_answer_info(sample)

    # Accuracy is used for logging/eval metrics; training reward remains zero in post_process_rewards.
    accuracy = 1.0 if grade_answer_verl(sample.response or "", sample.label or "") else 0.0
    return {
        "teacher_output": teacher_output,
        "accuracy": accuracy,
        **answer_info,
        "student_answer_reward": 1.0 if answer_info["student_answer_correct"] else 0.0,
    }


def post_process_rewards(args, samples: list[Sample], **kwargs):
    """Process rewards from teacher model and extract teacher log probabilities.

    This function:
    1. Extracts teacher log-probs from the reward response (which contains sglang's logprob output)
    2. Trims them to match the response length
    3. Stores them in sample.teacher_log_probs for OPD KL penalty computation
    4. Returns scalar rewards (0.0 for pure distillation) compatible with GRPO/PPO

    Note: The reward_func calls the teacher server which returns token-level log-probs.
    For pure on-policy distillation without task rewards, we return 0.0 for each sample.
    The actual learning signal comes from the OPD KL penalty applied in compute_advantages_and_returns.
    """
    raw_rewards = []
    response_lengths = [sample.response_length for sample in samples]

    teacher_outputs = []
    for sample in samples:
        reward = sample.reward
        if isinstance(reward, dict) and "teacher_output" in reward:
            teacher_output = reward["teacher_output"]
            raw_rewards.append(float(reward.get("accuracy", 0.0)))
        else:
            # Backward-compatible path for historical checkpoints/scripts.
            teacher_output = reward
            raw_rewards.append(0.0)
        teacher_outputs.append(teacher_output)

    # Extract teacher log-probs from the sglang response and robustly align them to
    # the response segment. Some sglang responses include an initial null log-prob slot,
    # while others already return exactly response_length values.
    teacher_log_probs = [
        _extract_aligned_teacher_log_probs(teacher_output, response_length)
        for teacher_output, response_length in zip(teacher_outputs, response_lengths, strict=False)
    ]

    for sample, t_log_probs in zip(samples, teacher_log_probs, strict=False):
        sample.teacher_log_probs = t_log_probs

    # Return scalar rewards for GRPO/PPO advantage estimator
    # For pure on-policy distillation, we use 0.0 as the task reward.
    # The learning signal comes entirely from the OPD KL penalty.
    # If you have task rewards, you can add them here.
    scalar_rewards = [0.0] * len(samples)

    return raw_rewards, scalar_rewards
