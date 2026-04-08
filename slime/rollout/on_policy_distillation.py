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


async def reward_func(args, sample, **kwargs):
    payload = {
        # "text": sample.prompt + sample.response,
        "input_ids": sample.tokens,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    session_kwargs = {}
    async with aiohttp.ClientSession(**session_kwargs) as session:
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

    # Extract teacher log-probs from the sglang response
    teacher_log_probs = [
        torch.tensor([item[0] for item in reward["meta_info"]["input_token_logprobs"][1:]], dtype=torch.float32)
        for reward in teacher_outputs
    ]
    teacher_log_probs = [
        t_log_prob[-response_length:]
        for t_log_prob, response_length in zip(teacher_log_probs, response_lengths, strict=False)
    ]

    for sample, t_log_probs in zip(samples, teacher_log_probs, strict=False):
        sample.teacher_log_probs = t_log_probs

    # Return scalar rewards for GRPO/PPO advantage estimator
    # For pure on-policy distillation, we use 0.0 as the task reward.
    # The learning signal comes entirely from the OPD KL penalty.
    # If you have task rewards, you can add them here.
    scalar_rewards = [0.0] * len(samples)

    return raw_rewards, scalar_rewards
