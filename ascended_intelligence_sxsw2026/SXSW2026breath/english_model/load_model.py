"""
load_model: Load emotion and noise models from english_model/model/{model_name}/.

Cross-platform. Reads from paths defined in download_model.
"""
from pathlib import Path
from typing import Optional

from .download_model import (
    get_model_dir,
    get_emotion_model_dir,
    get_noise_model_dir,
)


def get_emotion_model_path() -> Optional[Path]:
    """Return path to emotion model dir if it exists and has model files."""
    p = get_emotion_model_dir()
    if not p.exists():
        return None
    # Check for typical model files
    if (p / "config.json").exists() or (p / "pytorch_model.bin").exists():
        return p
    return None


def get_noise_model_path() -> Optional[Path]:
    """Return path to noise model dir if it exists and has model files."""
    p = get_noise_model_dir()
    if not p.exists():
        return None
    sub = p / "DeepFilterNet2"  # Space has model in subfolder
    if sub.exists() and (any(sub.glob("*.yaml")) or any(sub.glob("*.json"))):
        return sub
    if any(p.glob("*.yaml")) or any(p.glob("*.json")):
        return p
    return None
