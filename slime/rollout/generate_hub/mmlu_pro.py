from __future__ import annotations

from argparse import Namespace
from copy import deepcopy
import re
from typing import Any

from slime.rollout.sglang_rollout import generate as _generate_base
from slime.utils.types import Sample

MMLU_PRO_FINAL_ANSWER_INSTRUCTION = (
    "Think through the problem carefully and write out your detailed reasoning. "
    "Then end with exactly one final non-empty line in the format:\n"
    "Final Answer: \\boxed{A}\n"
    "Replace A with the single correct option letter only. Do not put any extra text inside \\boxed{}."
)


def _append_instruction(prompt):
    if isinstance(prompt, str):
        if "Final Answer: \\boxed{A}" in prompt:
            return prompt

        # Chat-template string prompts (e.g. Qwen) usually look like:
        # <|im_start|>user ... <|im_end|>\n<|im_start|>assistant ...
        # Injecting instruction at the very end can accidentally place it into
        # assistant prefill context. Instead, patch the last user message body.
        user_spans = list(
            re.finditer(r"(?s)<\|im_start\|>user\n(.*?)<\|im_end\|>", prompt)
        )
        if user_spans:
            last = user_spans[-1]
            body_start, body_end = last.span(1)
            user_body = prompt[body_start:body_end].rstrip()
            user_body = f"{user_body}\n\n{MMLU_PRO_FINAL_ANSWER_INSTRUCTION}"
            return prompt[:body_start] + user_body + prompt[body_end:]

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
