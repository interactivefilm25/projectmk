"""
english_model: Emotion recognition (Wav2Vec2 – same as TouchDesigner / realtime_emotion_td.py).

Uses Wav2Vec2ForSequenceClassification from transformers.
Loads from english_model/model/multilingual_emotion_model/ when available, else from HuggingFace.
"""
from __future__ import annotations

import contextlib
import io
import logging
import os
import sys
from pathlib import Path

import numpy as np

from .load_model import get_emotion_model_path

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Same 5 labels as realtime_emotion_td.py (TouchDesigner)
TARGET_SR = 16000
CANONICAL_EMOTIONS = [
    "anger_fear",
    "joy_excited",
    "sadness",
    "curious_reflective",
    "calm_content",
]

# Fallback: public HF model (8 classes) when no local TouchDesigner model is found.
HF_FALLBACK_REPO = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
# Map 8-class *label name* to our 5-class index (0=anger_fear, 1=joy_excited, 2=sadness, 3=curious_reflective, 4=calm_content)
# Used when model id2label is read at runtime so we're robust to any label order
FALLBACK_LABEL_TO_5_IDX = {
    "angry": 0,      # -> anger_fear
    "fearful": 0,    # -> anger_fear
    "happy": 1,      # -> joy_excited
    "surprised": 1,  # -> joy_excited
    "sad": 2,        # -> sadness
    "disgust": 3,    # -> curious_reflective
    "calm": 4,       # -> calm_content
    "neutral": 4,    # -> calm_content
}


class Audio2EmotionModel:
    """Wav2Vec2 emotion model (multilingual_emotion_model – TouchDesigner format)."""

    def __init__(self, model_path: str | Path | None = None):
        self._model_path = model_path
        self._model = None
        self._feature_extractor = None
        self._device = None
        self._use_fallback_8_to_5 = False  # True when using HF fallback (8 classes -> map to 5)

    def _ensure_loaded(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
        except ImportError as e:
            raise ImportError(
                "english_model: Install pip install transformers torch"
            ) from e

        for name in ("", "transformers", "torch"):
            logging.getLogger(name).setLevel(logging.ERROR)

        # Prefer local weights: english_model/model/multilingual_emotion_model (or nested subdir)
        path = self._model_path
        if path is None:
            path = get_emotion_model_path()
        if path is not None:
            path = str(Path(path).resolve())
            # Use local weights only; no fallback
        else:
            # Try project root then cwd for realtime_emotion_td.py-style folder
            project_root = Path(__file__).resolve().parent.parent
            for candidate in [project_root / "multilingual_emotion_model", Path("multilingual_emotion_model")]:
                if candidate.exists() and (candidate / "config.json").exists():
                    path = str(candidate.resolve())
                    break
            if path is None:
                path = HF_FALLBACK_REPO
                self._use_fallback_8_to_5 = True

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self._feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(path)
            self._model = Wav2Vec2ForSequenceClassification.from_pretrained(path)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._model.eval()

    @property
    def emotions(self) -> list[str]:
        return list(CANONICAL_EMOTIONS)

    def infer(self, waveform: np.ndarray) -> tuple[np.ndarray, int]:
        import torch

        self._ensure_loaded()

        if waveform.ndim == 2:
            waveform = waveform.squeeze(0)
        if waveform.dtype != np.float32:
            waveform = waveform.astype(np.float32)

        min_samples = 16000  # 1 s at 16 kHz
        if len(waveform) < min_samples:
            waveform = np.pad(
                waveform,
                (0, min_samples - len(waveform)),
                mode="constant",
                constant_values=0,
            )

        inputs = self._feature_extractor(
            waveform,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._model(**inputs).logits
            logits = logits.cpu().numpy().squeeze()

        if logits.ndim == 0:
            logits = np.expand_dims(logits, 0)
        logits = np.asarray(logits).flatten()

        if self._use_fallback_8_to_5 and len(logits) >= 8:
            id2label = getattr(self._model.config, "id2label", None) or {}

            def get_label(i):
                return (id2label.get(str(i)) or id2label.get(i) or "").lower()

            # Aggregate 8-class probs into our 5 classes (by label name)
            probs_5 = np.zeros(len(CANONICAL_EMOTIONS), dtype=np.float32)
            exp = np.exp(logits - np.max(logits))
            probs_8 = (exp / exp.sum())[:8]
            for i in range(len(probs_8)):
                j = FALLBACK_LABEL_TO_5_IDX.get(get_label(i), 4)
                probs_5[j] += probs_8[i]
            probs_5 = probs_5 / (probs_5.sum() or 1e-9)
            probs_arr = probs_5.astype(np.float32)

            # Top 8-class prediction -> our 5-class index
            order_8 = np.argsort(logits[:8])[::-1]
            pred_8 = int(order_8[0])
            label_8 = get_label(pred_8)
            pred_idx = FALLBACK_LABEL_TO_5_IDX.get(label_8, 4)
            # When model is uncertain (top = neutral/calm), use runner-up if it's angry or happy
            if label_8 in ("neutral", "calm") and len(order_8) > 1:
                runner_label = get_label(int(order_8[1]))
                if runner_label in ("angry", "fearful"):
                    pred_idx = 0
                    probs_arr = np.zeros(len(CANONICAL_EMOTIONS), dtype=np.float32)
                    probs_arr[0] = 1.0
                elif runner_label in ("happy", "surprised"):
                    pred_idx = 1
                    probs_arr = np.zeros(len(CANONICAL_EMOTIONS), dtype=np.float32)
                    probs_arr[1] = 1.0
            else:
                pred_idx = int(np.argmax(probs_arr))
        else:
            logits = logits[: len(CANONICAL_EMOTIONS)]
            if len(logits) < len(CANONICAL_EMOTIONS):
                logits = np.pad(logits, (0, len(CANONICAL_EMOTIONS) - len(logits)))
            exp = np.exp(logits - np.max(logits))
            probs_arr = (exp / exp.sum()).astype(np.float32)
            pred_idx = int(np.argmax(probs_arr))

        return probs_arr, pred_idx


_model_instance: Audio2EmotionModel | None = None


def get_model() -> Audio2EmotionModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = Audio2EmotionModel()
    return _model_instance
