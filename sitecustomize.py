"""Runtime compatibility patches for this OPD workspace.

This module is auto-imported by Python at startup (if ROOT_DIR is on PYTHONPATH).
"""

import os


def _patch_transformers_autoconfig_register() -> None:
    # vLLM 0.9 may call AutoConfig.register without exist_ok=True for model types
    # that already exist in newer transformers versions (e.g. "aimv2").
    if os.environ.get("OPD_AUTOCONFIG_REGISTER_EXIST_OK", "0") != "1":
        return

    try:
        from transformers import AutoConfig
    except Exception:
        return

    original_register = AutoConfig.register

    # Avoid double patching across nested imports.
    if getattr(original_register, "_opd_patched_exist_ok", False):
        return

    def safe_register(model_type, config, exist_ok=False):
        return original_register(model_type, config, exist_ok=True)

    safe_register._opd_patched_exist_ok = True  # type: ignore[attr-defined]
    AutoConfig.register = safe_register


_patch_transformers_autoconfig_register()

