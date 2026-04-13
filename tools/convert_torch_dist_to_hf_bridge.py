import argparse
import importlib.util
import os

import megatron.bridge.training.model_load_save as _model_load_save_module
import torch
from megatron.bridge import AutoBridge

import slime_plugins.megatron_bridge  # noqa: F401


# Here we need to patch Megatron Bridge's `load_model_config`, since the checkpoint is saved
# by Megatron and lack of provider information.
_provider_override = {}
_original_load_model_config = _model_load_save_module.load_model_config


def _has_transformer_engine() -> bool:
    return importlib.util.find_spec("transformer_engine") is not None


def _patched_load_model_config(checkpoint_path):
    provider = _provider_override.get("provider")
    if provider is not None:
        # Bypass Megatron Bridge's TransformerConfig reconstruction from args,
        # which can fail if optional TE/Apex fusion flags were enabled during
        # training but are unavailable in the current conversion env.
        common_pt = os.path.join(checkpoint_path, "common.pt")
        if not os.path.exists(common_pt):
            raise FileNotFoundError(f"Missing common.pt under checkpoint: {checkpoint_path}")
        mlm_args = torch.load(common_pt, weights_only=False)["args"]
        for flag_name in (
            "apply_rope_fusion",
            "bias_swiglu_fusion",
            "bias_gelu_fusion",
            "masked_softmax_fusion",
            "persist_layer_norm",
            "memory_efficient_layer_norm",
            "gradient_accumulation_fusion",
        ):
            if hasattr(mlm_args, flag_name):
                setattr(mlm_args, flag_name, False)
        print(f"[convert] Overriding MLM TransformerConfig with Bridge provider: {type(provider).__name__}")
        return provider, mlm_args
    return _original_load_model_config(checkpoint_path)


_model_load_save_module.load_model_config = _patched_load_model_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert torch distributed checkpoint to HuggingFace format using Megatron Bridge"
    )
    parser.add_argument(
        "--input-dir", type=str, required=True, help="Path to the torch distributed checkpoint directory"
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Path to save the HuggingFace checkpoint")
    parser.add_argument(
        "--origin-hf-dir",
        type=str,
        required=True,
        help="Path to the original HuggingFace model directory (for config)",
    )
    parser.add_argument(
        "-f", "--force", action="store_true", help="Force overwrite the output directory if it exists."
    )
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and not args.force:
        raise ValueError(f"Output directory {args.output_dir} already exists. Use --force to overwrite it.")

    print(f"Loading config from {args.origin_hf_dir}")
    bridge = AutoBridge.from_hf_pretrained(args.origin_hf_dir, trust_remote_code=True)

    # Use Bridge's provider so the correct model class is created (e.g., Qwen3VLModel
    # instead of GPTModel). This is needed because MLM checkpoints lack run_config.yaml.
    provider = bridge.to_megatron_provider(load_weights=False)
    if not _has_transformer_engine():
        from megatron.bridge.models.gpt_provider import local_layer_spec as bridge_local_layer_spec

        provider.sequence_parallel = False
        provider.transformer_layer_spec = bridge_local_layer_spec
        if hasattr(provider, "use_transformer_engine_full_layer_spec"):
            provider.use_transformer_engine_full_layer_spec = False
        if hasattr(provider, "use_te_rng_tracker"):
            provider.use_te_rng_tracker = False
    provider.finalize()
    _provider_override["provider"] = provider
    print(f"[convert] Using Bridge provider: {type(provider).__name__}")

    print(f"Exporting checkpoint from {args.input_dir} to {args.output_dir}")
    bridge.export_ckpt(args.input_dir, args.output_dir)

    print("Done!")
