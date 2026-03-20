"""ComfyUI custom nodes for Foundation-1.

Foundation-1 is a structured text-to-sample diffusion model for music
production. It understands instrument identity, timbre, FX, musical
notation, BPM, bar count, and key as separate composable controls.

Nodes registered:
  - Foundation1ModelLoader  — loads a local .safetensors checkpoint
  - Foundation1Generate     — generates a tempo-synced musical loop

Model files go in:  ComfyUI/models/stable_audio/<any-subfolder>/
Required files:     Foundation_1.safetensors  +  model_config.json
"""

__version__ = "0.1.5"

import warnings

warnings.filterwarnings("ignore", message="Please use the new API settings to control TF32")
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*is deprecated.*")
warnings.filterwarnings("ignore", message="Should have tb<=t1 but got tb=")

import importlib
import importlib.util
import logging
import sys
from pathlib import Path

_NODE_DIR = Path(__file__).parent.resolve()
_KDIFF_TARGET = _NODE_DIR / "k_diffusion_files"

logger = logging.getLogger("Foundation1")
logger.propagate = False

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[Foundation1] %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


def _register_comfy_k_diffusion() -> None:
    """Register ComfyUI's k_diffusion as the top-level 'k_diffusion' module."""
    if "k_diffusion" in sys.modules:
        return
    try:
        import comfy.k_diffusion as _comfy_kd
        sys.modules["k_diffusion"] = _comfy_kd
    except ImportError:
        logger.warning("Could not import comfy.k_diffusion")


def _load_file_as_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_external_injected = False

def _inject_k_external() -> None:
    """Load external.py from our private k-diffusion install."""
    global _external_injected
    if _external_injected:
        return

    import k_diffusion as _kd

    ext_path = _KDIFF_TARGET / "k_diffusion" / "external.py"
    mod = _load_file_as_module("k_diffusion.external", ext_path)
    _kd.external = mod
    _external_injected = True


_register_comfy_k_diffusion()

try:
    _inject_k_external()
except Exception as e:
    logger.warning(f"Could not inject k_diffusion.external: {e}")

from .nodes.loader_node import Foundation1ModelLoader
from .nodes.generate_node import Foundation1Generate

NODE_CLASS_MAPPINGS = {
    "Foundation1ModelLoader": Foundation1ModelLoader,
    "Foundation1Generate": Foundation1Generate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Foundation1ModelLoader": "Foundation-1 Model Loader",
    "Foundation1Generate": "Foundation-1 Generate",
}

logger.info(
    f"Registered {len(NODE_CLASS_MAPPINGS)} nodes (v{__version__}): "
    + ", ".join(NODE_DISPLAY_NAME_MAPPINGS.values())
)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "__version__"]
