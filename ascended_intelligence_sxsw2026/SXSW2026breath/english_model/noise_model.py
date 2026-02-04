"""
noise_model: Audio noise cleaning + enhancement.

- clean(): DeepFilterNet2 or noisereduce fallback (denoise)
- enhance(): Gentle enhancement that preserves quality (no harsh processing)
"""
from pathlib import Path
from typing import Optional

import numpy as np

from .download_model import get_noise_model_dir, NOISE_MODEL_NAME, HF_NOISE_REPO
from .load_model import get_noise_model_path


def enhance_audio(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Enhance live audio for better clarity.
    - Rumble filter 50Hz (1st order)
    - Light noise reduction (prop_decrease=0.4) - runs after clean() for extra polish
    - Peak normalization to 0.95
    """
    audio = waveform.astype(np.float32)

    # 1. Rumble filter
    try:
        from scipy import signal
        nyq = sample_rate / 2
        hp = 50 / nyq
        if hp < 1.0:
            b, a = signal.butter(1, hp, btype="high")
            audio = signal.filtfilt(b, a, audio).astype(np.float32)
    except Exception:
        pass

    # 2. Light noise reduction (audible enhancement for live mic)
    try:
        import noisereduce as nr
        audio = nr.reduce_noise(
            y=audio, sr=sample_rate,
            stationary=True, prop_decrease=0.35,
        ).astype(np.float32)
    except Exception:
        pass

    # 3. Peak normalization
    peak = np.abs(audio).max()
    if peak > 0.01:
        audio = audio * (0.95 / peak)
    return audio


class NoiseCleaner:
    """Audio noise cleaning using DeepFilterNet2."""

    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = model_dir or get_noise_model_dir()
        self._model = None
        self._df_state = None
        self._enhance_fn = None
        self._resample_fn = None
        self._available = False
        self._fallback_nr = None

    def _ensure_loaded(self) -> bool:
        if self._available:
            return True
        if self._available is False and self._fallback_nr is not None:
            return False

        try:
            from huggingface_hub import snapshot_download
            import torch
        except ImportError:
            self._init_fallback()
            return False

        try:
            from df.enhance import init_df, enhance
            from df.io import resample

            model_path = get_noise_model_path()
            if model_path and model_path.exists():
                init_path = str(model_path)
            else:
                try:
                    self._model, self._df_state, _ = init_df(config_allow_defaults=True)
                    self._finish_init(torch, enhance, resample)
                    return True
                except Exception:
                    self.model_dir.mkdir(parents=True, exist_ok=True)
                    local_dir = snapshot_download(
                        repo_id=HF_NOISE_REPO,
                        repo_type="space",
                        local_dir=str(self.model_dir),
                    )
                    init_path = str(Path(local_dir) / NOISE_MODEL_NAME) if (Path(local_dir) / NOISE_MODEL_NAME).exists() else local_dir

            self._model, self._df_state, _ = init_df(init_path, config_allow_defaults=True)
            self._finish_init(torch, enhance, resample)
            return True

        except Exception:
            self._init_fallback()
            return False

    def _finish_init(self, torch, enhance, resample):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._model.to(device=device).eval()
        self._enhance_fn = enhance
        self._resample_fn = resample
        self._available = True

    def _init_fallback(self):
        try:
            import noisereduce as nr
            self._fallback_nr = nr
        except ImportError:
            self._fallback_nr = None  # noisereduce fallback; see noise_model

    def clean(self, waveform: np.ndarray, sample_rate: int, target_sr: int = 48000) -> np.ndarray:
        if not self._ensure_loaded():
            return self._clean_fallback(waveform, sample_rate)

        try:
            import torch
            audio = waveform.astype(np.float32)
            audio_t = torch.from_numpy(audio).float().unsqueeze(0)
            if sample_rate != target_sr:
                audio_t = self._resample_fn(audio_t, sample_rate, target_sr)
            enhanced = self._enhance_fn(self._model, self._df_state, audio_t)
            if sample_rate != target_sr:
                enhanced = self._resample_fn(enhanced, target_sr, sample_rate)
            return enhanced.squeeze().cpu().numpy().astype(np.float32)
        except Exception:
            return self._clean_fallback(waveform, sample_rate)

    def _clean_fallback(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        if self._fallback_nr is None:
            self._init_fallback()
        if self._fallback_nr is None:
            return waveform
        try:
            return self._fallback_nr.reduce_noise(
                y=waveform.astype(np.float32), sr=sample_rate,
                stationary=True, prop_decrease=0.2,  # Gentle to preserve quality
            ).astype(np.float32)
        except Exception:
            return waveform

    @property
    def is_deepfilternet(self) -> bool:
        return self._available


_noise_cleaner_instance: Optional[NoiseCleaner] = None


def get_noise_cleaner(model_dir: Optional[Path] = None) -> NoiseCleaner:
    global _noise_cleaner_instance
    if _noise_cleaner_instance is None:
        _noise_cleaner_instance = NoiseCleaner(model_dir=model_dir)
    return _noise_cleaner_instance
