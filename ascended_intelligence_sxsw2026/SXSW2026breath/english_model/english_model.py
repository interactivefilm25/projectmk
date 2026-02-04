"""
english_model: Emotion recognition (emotion2vec_plus_base).

Loads from english_model/model/emotion2vec_plus_base/ when available.
Cross-platform.
"""
from __future__ import annotations

import contextlib
import io
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from .download_model import get_emotion_model_dir
from .load_model import get_emotion_model_path

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

MODEL_ID = "emotion2vec_plus_base"
TARGET_SR = 16000

EMOTION2VEC_LABELS = [
    "angry", "disgusted", "fearful", "happy", "neutral",
    "other", "sad", "surprised", "unknown",
]
CANONICAL_EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad"]

EMOTION2VEC_TO_CANONICAL = {
    "angry": "angry",
    "disgusted": "disgust",
    "fearful": "fear",
    "happy": "happy",
    "neutral": "neutral",
    "other": "neutral",
    "sad": "sad",
    "surprised": "happy",
    "unknown": "neutral",
}


class Audio2EmotionModel:
    """emotion2vec_plus_base emotion model."""

    def __init__(self):
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        try:
            from funasr import AutoModel
            from tqdm import tqdm
        except ImportError as e:
            raise ImportError("english_model: Install pip install funasr modelscope tqdm") from e

        for name in ("", "root", "funasr", "modelscope", "httpx"):
            logging.getLogger(name).setLevel(logging.ERROR)

        # Try local model first (english_model/model/emotion2vec_plus_base/)
        model_path = get_emotion_model_path()
        if model_path:
            model_arg = str(model_path)
            hub_arg = None  # local path
        else:
            model_arg = "emotion2vec/emotion2vec_plus_base"
            hub_arg = "hf"

        with tqdm(total=100, desc="Loading emotion model", unit="%",
                  bar_format="{desc}: {bar}| {n}% {elapsed}", file=sys.stdout) as pbar:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                if hub_arg:
                    self._model = AutoModel(model=model_arg, hub=hub_arg)
                else:
                    self._model = AutoModel(model=model_arg)
            pbar.update(100)

    @property
    def emotions(self) -> list[str]:
        return list(CANONICAL_EMOTIONS)

    def infer(self, waveform: np.ndarray) -> tuple[np.ndarray, int]:
        import soundfile as sf

        self._ensure_loaded()

        if waveform.ndim == 2:
            waveform = waveform.squeeze(0)
        if waveform.dtype != np.float32:
            waveform = waveform.astype(np.float32)

        min_samples = 16000
        if len(waveform) < min_samples:
            waveform = np.pad(
                waveform, (0, min_samples - len(waveform)),
                mode="constant", constant_values=0,
            )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        try:
            sf.write(tmp_path, waveform, TARGET_SR)
            with tempfile.TemporaryDirectory() as out_dir:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    res = self._model.generate(
                        tmp_path,
                        output_dir=out_dir,
                        granularity="utterance",
                        extract_embedding=False,
                    )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        if not res:
            probs_arr = np.ones(len(CANONICAL_EMOTIONS)) / len(CANONICAL_EMOTIONS)
            return probs_arr.astype(np.float32), 0

        r = res[0]
        labels = r.get("labels", EMOTION2VEC_LABELS)
        scores = r.get("scores", [])

        probs_raw = np.zeros(9, dtype=np.float32)
        for i, lab in enumerate(labels):
            if i < len(scores):
                probs_raw[i] = float(scores[i])
        s = probs_raw.sum()
        if s > 0:
            probs_raw = probs_raw / s

        canonical_probs = {e: 0.0 for e in CANONICAL_EMOTIONS}
        for i, lab in enumerate(EMOTION2VEC_LABELS):
            if i < len(probs_raw):
                mapped = EMOTION2VEC_TO_CANONICAL.get(lab, "neutral")
                canonical_probs[mapped] += float(probs_raw[i])
        probs_arr = np.array([canonical_probs[e] for e in CANONICAL_EMOTIONS], dtype=np.float32)
        probs_arr = probs_arr / (probs_arr.sum() or 1e-9)

        HAPPY_PENALTY = 0.4
        happy_idx = CANONICAL_EMOTIONS.index("happy")
        probs_arr[happy_idx] *= HAPPY_PENALTY
        probs_arr = probs_arr / (probs_arr.sum() or 1e-9)

        pred_idx = int(np.argmax(probs_arr))
        return probs_arr, pred_idx


_model_instance: Audio2EmotionModel | None = None


def get_model() -> Audio2EmotionModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = Audio2EmotionModel()
    return _model_instance
