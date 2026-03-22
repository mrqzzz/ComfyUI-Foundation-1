"""Foundation-1 model cache, attention management, and ComfyUI memory integration.

This module is the single source of truth for the loaded model state.
It hooks into ComfyUI's model_management memory functions at import time so
all 'Free Memory' / 'Unload' / 'Cleanup' UI actions also clear our cache.
"""

import gc
import logging
from typing import Any, Dict, Optional, Tuple

import torch

logger = logging.getLogger("Foundation1")

# ---------------------------------------------------------------------------
# Cache state
# ---------------------------------------------------------------------------

_cached_model_data: Optional[Dict[str, Any]] = None
_cached_key: Tuple = ()

# When True, ComfyUI memory hooks will NOT evict the model.
# Set to False when unload_after_generate=True so ComfyUI's 'Free Memory'
# also clears the CPU-resident copy.
_keep_in_vram: bool = True

# True once the model has been moved to CPU via offload_to_cpu().
_offloaded_to_cpu: bool = False

# Track whether SageAttention is currently monkey-patched onto
# torch.nn.functional.scaled_dot_product_attention.
_sage_patched: bool = False


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------

def get_cache_key(checkpoint_path: str, device: str, attention: str) -> Tuple:
    return (checkpoint_path, device, attention)


# ---------------------------------------------------------------------------
# Cache accessors
# ---------------------------------------------------------------------------

def get_cached_model() -> Tuple[Optional[Dict[str, Any]], Tuple]:
    return _cached_model_data, _cached_key


def set_cached_model(
    model_data: Dict[str, Any],
    key: Tuple,
    keep_in_vram: bool = True,
) -> None:
    global _cached_model_data, _cached_key, _keep_in_vram, _offloaded_to_cpu
    _cached_model_data = model_data
    _cached_key = key
    _keep_in_vram = keep_in_vram
    _offloaded_to_cpu = False


def set_keep_in_vram(value: bool) -> None:
    """Update the keep_in_vram flag.

    Called from the generate node when unload_after_generate changes so
    that ComfyUI's memory management UI buttons behave correctly.
    """
    global _keep_in_vram
    _keep_in_vram = value


def is_offloaded() -> bool:
    return _offloaded_to_cpu


# ---------------------------------------------------------------------------
# Attention management
# ---------------------------------------------------------------------------

def apply_attention(attention_type: str, device: str) -> str:
    """Configure the requested attention backend.

    Returns the attention type that was actually applied, which may differ
    from the requested type if a fallback was needed.

    Restores the original F.scaled_dot_product_attention before any new
    attention setting is applied, so switching types always starts clean.
    """
    global _sage_patched

    # Always restore the original SDPA first to avoid double-patching
    # or leaving a stale SageAttention patch when switching to sdpa/flash.
    if _sage_patched:
        import torch.nn.functional as F
        if hasattr(F, "_f1_original_sdpa"):
            F.scaled_dot_product_attention = F._f1_original_sdpa
            del F._f1_original_sdpa
        _sage_patched = False
        logger.debug("Restored original F.scaled_dot_product_attention.")

    # Non-CUDA devices only support SDPA — warn and fall back.
    if device != "cuda":
        if attention_type in ("sageattention", "flash_attention_2"):
            logger.warning(
                f"'{attention_type}' requires CUDA but device is '{device}'. "
                "Using sdpa."
            )
        _set_sdpa_all_backends(True)
        return "sdpa"

    # ── CUDA path ──────────────────────────────────────────────────────────

    resolved = attention_type
    if attention_type == "auto":
        try:
            import sageattention  # noqa: F401
            resolved = "sageattention"
        except ImportError:
            resolved = "sdpa"

    if resolved == "sageattention":
        try:
            from sageattention import sageattn
            import torch.nn.functional as F
            F._f1_original_sdpa = F.scaled_dot_product_attention
            F.scaled_dot_product_attention = sageattn
            _sage_patched = True
            logger.info("Attention: SageAttention active (monkey-patched F.sdpa).")
        except ImportError:
            logger.warning(
                "SageAttention not installed — install with: pip install sageattention. "
                "Falling back to sdpa."
            )
            resolved = "sdpa"

    if resolved == "flash_attention_2":
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(False)
            logger.info("Attention: Flash Attention (SDPA flash-only backend) active.")
        except Exception as e:
            logger.warning(f"Failed to enable Flash Attention: {e}. Using sdpa.")
            _set_sdpa_all_backends(True)
            resolved = "sdpa"

    if resolved == "sdpa":
        _set_sdpa_all_backends(True)
        logger.info("Attention: SDPA (all backends enabled — uses Flash if available).")

    return resolved


def _set_sdpa_all_backends(enabled: bool) -> None:
    """Enable or disable all PyTorch SDPA backends uniformly."""
    try:
        torch.backends.cuda.enable_flash_sdp(enabled)
        torch.backends.cuda.enable_mem_efficient_sdp(enabled)
        torch.backends.cuda.enable_math_sdp(enabled)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Aggressive memory cleanup
# ---------------------------------------------------------------------------

def _force_gc() -> None:
    """Run multiple garbage collection passes and free CUDA cache.

    Two passes are needed because the first pass can free large objects
    whose destructors then reference other large objects that only become
    collectible on the second pass.
    """
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# CPU offload / resume
# ---------------------------------------------------------------------------

def offload_to_cpu() -> None:
    """Move the model from VRAM to system RAM.

    The model stays in memory so the next resume() is much faster than
    a cold reload from disk. VRAM is freed immediately after the move.

    Also sets _keep_in_vram=False so ComfyUI's memory management UI
    buttons can fully unload the model from RAM if the user clicks them.
    """
    global _offloaded_to_cpu, _keep_in_vram

    if _cached_model_data is None:
        return
    if _offloaded_to_cpu:
        logger.debug("Model already on CPU — skipping offload.")
        return

    try:
        _cached_model_data["model"].to("cpu")
        _offloaded_to_cpu = True
        _keep_in_vram = False
        _cached_model_data["_compiled"] = False

        _force_gc()

        logger.info("Foundation-1 offloaded to CPU. VRAM freed.")
    except Exception as e:
        logger.warning(f"CPU offload failed: {e}")


def resume_to_device(device: str) -> None:
    """Move the model from CPU back to the target device before generation."""
    global _offloaded_to_cpu

    if _cached_model_data is None or not _offloaded_to_cpu:
        return

    try:
        _cached_model_data["model"].to(device)
        _cached_model_data["device"] = device
        _offloaded_to_cpu = False
        logger.info(f"Foundation-1 resumed to {device}.")
    except Exception as e:
        logger.warning(f"Resume to {device} failed: {e}")


# ---------------------------------------------------------------------------
# Full unload
# ---------------------------------------------------------------------------

def unload_model() -> None:
    """Remove the model from memory entirely (VRAM, CPU RAM, everything).

    Nulls out all references and runs aggressive garbage collection to
    ensure Python and CUDA memory are both fully freed.
    """
    global _cached_model_data, _cached_key, _keep_in_vram, _offloaded_to_cpu

    if _cached_model_data is None:
        return

    logger.info("Unloading Foundation-1 from memory...")

    model_ref = _cached_model_data.get("model")

    _cached_model_data.clear()
    _cached_model_data = None
    _cached_key = ()
    _keep_in_vram = True
    _offloaded_to_cpu = False

    if model_ref is not None:
        try:
            if hasattr(model_ref, "pretransform"):
                del model_ref.pretransform
        except Exception:
            pass
        try:
            if hasattr(model_ref, "conditioner"):
                del model_ref.conditioner
        except Exception:
            pass
        try:
            if hasattr(model_ref, "model"):
                del model_ref.model
        except Exception:
            pass
        del model_ref

    model_ref = None
    _force_gc()
    logger.info("Foundation-1 unloaded.")


# ---------------------------------------------------------------------------
# ComfyUI memory management hooks
# ---------------------------------------------------------------------------

def _hook_comfy_model_management() -> None:
    """Patch ComfyUI memory management functions so that any memory-freeing
    action also clears our cache when _keep_in_vram is False.

    We hook multiple entry points because different ComfyUI UI elements
    call different functions:
      - 'Free VRAM' button         → soft_empty_cache()
      - 'Free Model Cache' button  → cleanup_models()
      - ComfyUI Manager actions    → free_memory() / unload_all_models()
    """
    try:
        import comfy.model_management as mm
    except ImportError:
        return

    def _make_wrapper(original_fn, name):
        def _patched(*args, **kwargs):
            if not _keep_in_vram:
                unload_model()
            return original_fn(*args, **kwargs)
        return _patched

    hooked = []

    for name in ("soft_empty_cache", "cleanup_models", "free_memory", "unload_all_models"):
        if hasattr(mm, name):
            original = getattr(mm, name)
            setattr(mm, name, _make_wrapper(original, name))
            hooked.append(name)

    if hooked:
        logger.debug(f"Hooked comfy.model_management: {', '.join(hooked)}")


# Install the hook the moment this module is first imported.
_hook_comfy_model_management()
