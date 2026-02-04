"""
download_model: Download emotion and noise models to english_model/model/{model_name}/.

Cross-platform (Windows, Mac, Linux). Models are stored relative to this script.
"""
from pathlib import Path

# Base model directory (same folder as model_loader, cross-platform)
_SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = _SCRIPT_DIR / "model"

# Model names (subfolders under model/)
EMOTION_MODEL_NAME = "emotion2vec_plus_base"
NOISE_MODEL_NAME = "DeepFilterNet2"

# Hugging Face identifiers
HF_EMOTION_ID = "emotion2vec/emotion2vec_plus_base"
HF_NOISE_REPO = "hshr/DeepFilterNet2"


def get_model_dir() -> Path:
    """Return base model directory (english_model/model/)."""
    return MODEL_DIR


def get_emotion_model_dir() -> Path:
    """Return emotion model directory (model/emotion2vec_plus_base/)."""
    return MODEL_DIR / EMOTION_MODEL_NAME


def get_noise_model_dir() -> Path:
    """Return noise model directory (model/DeepFilterNet2/)."""
    return MODEL_DIR / NOISE_MODEL_NAME


def download_emotion_model() -> Path:
    """
    Download emotion model to model/emotion2vec_plus_base/.
    Uses huggingface_hub. Returns path to model dir.
    Cross-platform.
    """
    out_dir = get_emotion_model_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=HF_EMOTION_ID,
            local_dir=str(out_dir),
        )
    except Exception as e:
        raise RuntimeError(f"download_model: Failed to download emotion model: {e}") from e

    return out_dir


def download_noise_model() -> Path:
    """
    Download DeepFilterNet2 to model/DeepFilterNet2/.
    Returns path to model dir.
    """
    out_dir = get_noise_model_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
        local_dir = snapshot_download(
            repo_id=HF_NOISE_REPO,
            repo_type="space",
            local_dir=str(out_dir),
        )
        # Model may be in DeepFilterNet2 subfolder
        subfolder = Path(local_dir) / NOISE_MODEL_NAME
        if subfolder.exists():
            return subfolder
        return Path(local_dir)
    except Exception as e:
        raise RuntimeError(f"download_model: Failed to download noise model: {e}") from e
