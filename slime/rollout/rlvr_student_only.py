from __future__ import annotations

from slime.rollout.on_policy_distillation import _build_student_only_eval_reward
from slime.utils.types import Sample


async def reward_func(args, sample_or_samples, **kwargs):
    if isinstance(sample_or_samples, list):
        return [_build_student_only_eval_reward(sample) for sample in sample_or_samples]
    return _build_student_only_eval_reward(sample_or_samples)


def post_process_rewards(args, samples: list[Sample], **kwargs):
    raw_rewards: list[float] = []
    scalar_rewards: list[float] = []

    for sample in samples:
        reward = sample.reward
        if isinstance(reward, dict):
            score = float(reward.get("accuracy", reward.get("student_answer_reward", 0.0)) or 0.0)
        else:
            score = float(reward or 0.0)
            sample.reward = {"accuracy": score, "student_answer_reward": score}
        raw_rewards.append(score)
        scalar_rewards.append(score)

    return raw_rewards, scalar_rewards
