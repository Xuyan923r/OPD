import re
import string
from collections.abc import Iterable

DEFAULT_VALID_LETTERS = list(string.ascii_uppercase[:8])
STOP_WORDS = ("</s>", "<|im_end|>", "<|endoftext|>")


def _strip_chain_of_thought(text: str) -> str:
    if not text:
        return ""

    if "</think>" in text:
        return text.rsplit("</think>", 1)[-1]

    return text


def _strip_model_wrappers(text: str) -> str:
    if not text:
        return ""

    if "<|im_start|>user" in text:
        text = re.sub(
            r"^.*?<\|im_start\|>assistant",
            "",
            text,
            flags=re.DOTALL,
            count=1,
        )
    elif "Assistant:" in text:
        text = text.split("Assistant:")[-1].strip()

    for stop_word in STOP_WORDS:
        if stop_word in text:
            text = text.split(stop_word, 1)[0].strip()

    return text.strip()


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _prompt_to_text(prompt) -> str:
    if isinstance(prompt, str):
        return prompt

    if isinstance(prompt, list):
        parts = []
        for message in prompt:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and isinstance(item.get("text"), str):
                        parts.append(item["text"])
        return "\n".join(part for part in parts if part)

    return ""


def _extract_last_boxed(text: str) -> str | None:
    pattern = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
    matches = list(re.finditer(pattern, text))
    if matches:
        return matches[-1].group(1).strip()
    return None


def _extract_last_final_answer(text: str) -> str | None:
    patterns = [
        r"Final\s*Answer\s*:\s*(.*?)(?:\n|$)",
        r"The\s*answer\s*is\s*:\s*(.*?)(?:\n|$)",
        r"Answer\s*:\s*(.*?)(?:\n|$)",
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL))
        if matches:
            return matches[-1].group(1).strip()
    return None


def _extract_letters_from_prompt(prompt_text: str) -> list[str]:
    if not prompt_text:
        return []

    letters = []
    seen = set()
    pattern = re.compile(r"(?m)^\s*(?:\(([A-Z])\)|([A-Z]))\s*[:.)]\s+")
    for match in pattern.finditer(prompt_text):
        letter = next((group for group in match.groups() if group), None)
        if letter is None:
            continue
        letter = letter.upper()
        if letter not in seen:
            seen.add(letter)
            letters.append(letter)
    return letters


def _extract_choices_from_prompt(prompt_text: str) -> list[str]:
    if not prompt_text:
        return []

    pattern = re.compile(r"(?m)^\s*(?:\(([A-Z])\)|([A-Z]))\s*[:.)]\s+(.*)$")
    choices = []
    for match in pattern.finditer(prompt_text):
        choice_text = match.group(3).strip()
        if choice_text:
            choices.append(choice_text)
    return choices


def _extract_solution_text(response: str) -> str:
    text = _strip_model_wrappers(_strip_chain_of_thought(response))
    boxed = _extract_last_boxed(text)
    if boxed:
        return boxed

    final_answer = _extract_last_final_answer(text)
    if final_answer:
        return final_answer

    return text


def _resolve_valid_letters(
    *,
    metadata: dict,
    choices: list[str] | None,
    label_text: str | None,
    prompt_text: str,
) -> list[str]:
    valid_letters = metadata.get("valid_letters")
    if valid_letters:
        return [str(letter).upper() for letter in valid_letters]

    prompt_letters = _extract_letters_from_prompt(prompt_text)
    if prompt_letters:
        return prompt_letters

    if choices:
        return list(string.ascii_uppercase[: len(choices)])

    if isinstance(label_text, str) and len(label_text.strip()) == 1 and label_text.strip().isalpha():
        upper = label_text.strip().upper()
        return list(string.ascii_uppercase[: max(DEFAULT_VALID_LETTERS.index("H") + 1, ord(upper) - ord("A") + 1)])

    return DEFAULT_VALID_LETTERS


def _extract_letter_from_response(response: str, valid_letters: Iterable[str]) -> str | None:
    """
    Best-effort extraction of the selected option letter from the model response.
    """
    if not response:
        return None

    text = _strip_model_wrappers(_strip_chain_of_thought(response))
    solution_text = _extract_solution_text(response)
    patterns = [
        r"\\boxed\{\s*([A-Z])\s*\}",
        r"final\s*(?:answer|option)\s*(?:is|:)?\s*(?:\\boxed\{\s*)?([A-Z])(?:\s*\})?",
        r"the\s*answer\s*is\s*:\s*(?:\\boxed\{\s*)?([A-Z])(?:\s*\})?",
        r"(?:answer|option|choice)\s*(?:is|:)?\s*([A-Z])",
        r"([A-Z])\s*(?:is\s*(?:the)?\s*correct)",
    ]

    valid_letters = {letter.upper() for letter in valid_letters}
    for candidate_text in (solution_text, text):
        if not candidate_text:
            continue

        for pattern in patterns:
            match = re.search(pattern, candidate_text, flags=re.IGNORECASE)
            if match:
                letter = match.group(1).upper()
                if letter in valid_letters:
                    return letter

        candidates = re.findall(r"\b([A-Z])\b", candidate_text)
        for letter in reversed(candidates):
            letter = letter.upper()
            if letter in valid_letters:
                return letter

    return None


def compute_gpqa_reward(response: str, label, metadata: dict | None = None, prompt=None) -> float:
    """Rule-based scorer for multiple-choice evaluation such as GPQA and MMLU-Pro."""
    if response is None:
        return 0.0

    metadata = metadata or {}
    prompt_text = _prompt_to_text(prompt)

    choices = metadata.get("choices")
    if isinstance(choices, dict):
        choices = list(choices.values())
    elif choices is not None:
        choices = list(choices)
    elif prompt_text:
        choices = _extract_choices_from_prompt(prompt_text) or None

    correct_letter = metadata.get("correct_letter")
    if isinstance(correct_letter, str):
        correct_letter = correct_letter.strip().upper()
    else:
        correct_letter = None

    label_text = None
    if isinstance(label, str):
        label_text = label.strip()
    elif isinstance(label, (int, float)):
        idx = int(label)

    valid_letters = _resolve_valid_letters(
        metadata=metadata,
        choices=choices,
        label_text=label_text,
        prompt_text=prompt_text,
    )

    if isinstance(label, str) and len(label_text) == 1 and label_text.upper() in valid_letters and not correct_letter:
        correct_letter = label_text.upper()
    elif isinstance(label, (int, float)):
        idx = int(label)
        if 0 <= idx < len(valid_letters):
            correct_letter = valid_letters[idx]

    answer_index = metadata.get("answer_index")
    if correct_letter is None and isinstance(answer_index, int) and 0 <= answer_index < len(valid_letters):
        correct_letter = valid_letters[answer_index]

    if not correct_letter and choices and label_text:
        normalized_label = _normalize_text(label_text)
        for idx, choice in enumerate(choices):
            if _normalize_text(str(choice)) == normalized_label:
                correct_letter = valid_letters[idx]
                metadata.setdefault("correct_answer", choice)
                break

    extracted_letter = _extract_letter_from_response(response, valid_letters)
    if extracted_letter and correct_letter:
        return 1.0 if extracted_letter == correct_letter else 0.0

    candidate_answers = []
    if correct_letter and choices:
        try:
            idx = valid_letters.index(correct_letter)
        except ValueError:
            idx = None
        if idx is not None and idx < len(choices):
            candidate_answers.append(str(choices[idx]))

    for key in ("correct_answer", "answer_text"):
        value = metadata.get(key)
        if value:
            candidate_answers.append(str(value))

    if label_text:
        candidate_answers.append(label_text)

    normalized_targets = {_normalize_text(text) for text in candidate_answers if text}
    normalized_response = _normalize_text(_strip_chain_of_thought(response))
    for target in normalized_targets:
        if target and target in normalized_response:
            return 1.0

    if extracted_letter and not correct_letter and label_text:
        return 1.0 if extracted_letter == label_text.strip().upper() else 0.0

    return 0.0
