from __future__ import annotations

from argparse import Namespace
from copy import deepcopy
from typing import Any

from slime.rollout.sglang_rollout import generate as _generate_base
from slime.utils.types import Sample

MMLU_PRO_FINAL_ANSWER_INSTRUCTION = (
    "Reason step by step, then end with exactly one final line in the format:\n"
    "Final Answer: \\boxed{A}\n"
    "Replace A with the single correct option letter only. Do not put any extra text inside \\boxed{}."
)


def _append_instruction(prompt):
    if isinstance(prompt, str):
        if "Final Answer: \\boxed{A}" in prompt:
            return prompt
        return prompt.rstrip() + "\n\n" + MMLU_PRO_FINAL_ANSWER_INSTRUCTION

    if isinstance(prompt, list):
        prompt = deepcopy(prompt)
        if prompt and isinstance(prompt[-1], dict) and prompt[-1].get("role") == "user":
            content = prompt[-1].get("content")
            if isinstance(content, str) and "Final Answer: \\boxed{A}" not in content:
                prompt[-1]["content"] = content.rstrip() + "\n\n" + MMLU_PRO_FINAL_ANSWER_INSTRUCTION
                return prompt

        prompt.append({"role": "user", "content": MMLU_PRO_FINAL_ANSWER_INSTRUCTION})
        return prompt

    return prompt


async def generate_mmlu_pro_eval(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample:
    if evaluation and sample.response_length == 0:
        sample.prompt = _append_instruction(sample.prompt)
        sample.tokens = []

    return await _generate_base(args, sample, deepcopy(sampling_params))
