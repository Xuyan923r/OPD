from __future__ import annotations

from typing import Any

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import AutoMapping
from megatron.bridge.models.qwen.qwen3_bridge import Qwen3Bridge


def _patch_qwen3_bridge_local_spec_mappings() -> None:
    original_mapping_registry = Qwen3Bridge.mapping_registry

    if getattr(original_mapping_registry, "_slime_local_spec_patched", False):
        return

    def patched_mapping_registry(self: Qwen3Bridge) -> MegatronMappingRegistry:
        registry = original_mapping_registry(self)
        extra_mappings = [
            # Local-spec fallback creates standalone norm parameters instead of the
            # TE fused `*.layer_norm_weight` parameters used by upstream Qwen3Bridge.
            AutoMapping(
                megatron_param="decoder.layers.*.input_layernorm.weight",
                hf_param="model.layers.*.input_layernorm.weight",
            ),
            AutoMapping(
                megatron_param="decoder.layers.*.pre_mlp_layernorm.weight",
                hf_param="model.layers.*.post_attention_layernorm.weight",
            ),
        ]
        return MegatronMappingRegistry(*registry.get_all_mappings(), *extra_mappings)

    patched_mapping_registry._slime_local_spec_patched = True  # type: ignore[attr-defined]
    Qwen3Bridge.mapping_registry = patched_mapping_registry


def _patch_bridge_build_conversion_tasks() -> None:
    original_build_conversion_tasks = MegatronModelBridge.build_conversion_tasks

    if getattr(original_build_conversion_tasks, "_slime_filter_none_tasks_patched", False):
        return

    def patched_build_conversion_tasks(self: MegatronModelBridge, hf_pretrained: Any, megatron_model: Any):
        tasks = original_build_conversion_tasks(self, hf_pretrained, megatron_model)
        return [task for task in tasks if task is not None]

    patched_build_conversion_tasks._slime_filter_none_tasks_patched = True  # type: ignore[attr-defined]
    MegatronModelBridge.build_conversion_tasks = patched_build_conversion_tasks


_patch_qwen3_bridge_local_spec_mappings()
_patch_bridge_build_conversion_tasks()
