"""ComfyUI Manager install hook.

This file is automatically executed by ComfyUI Manager after git clone.
All pip installs happen here so Manager can see NODE_CLASS_MAPPINGS
before dependencies are installed.
"""

import subprocess
import sys
from pathlib import Path


def install():
    """Install required packages."""
    packages = [
        "einops>=0.7.0",
        "stable-audio-tools --no-deps",
        "rotary-embedding-torch",
        "alias-free-torch",
        "ema-pytorch",
        "einops-exts",
    ]

    node_dir = Path(__file__).parent.resolve()

    for pkg in packages:
        cmd = [sys.executable, "-m", "pip", "install", "--prefer-binary"] + pkg.split()
        print(f"[Foundation-1] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[Foundation-1] Failed to install {pkg}:\n{result.stderr}")
        else:
            print(f"[Foundation-1] Installed: {pkg}")

    kdiff_target = node_dir / "k_diffusion_files"
    kdiff_target.mkdir(parents=True, exist_ok=True)
    kdiff_cmd = [
        sys.executable, "-m", "pip", "install",
        "k-diffusion==0.1.1", "--no-deps", "--target", str(kdiff_target)
    ]
    print(f"[Foundation-1] Running: {' '.join(kdiff_cmd)}")
    result = subprocess.run(kdiff_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[Foundation-1] Failed to install k-diffusion:\n{result.stderr}")
    else:
        print(f"[Foundation-1] Installed k-diffusion to {kdiff_target}")


if __name__ == "__main__":
    install()
