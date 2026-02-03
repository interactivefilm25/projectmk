"""
emotion2vec/emotion2vec_plus_base emotion recognition model.
Uses FunASR; 9 emotions mapped to canonical 6.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Use short name; FunASR maps to HuggingFace emotion2vec/emotion2vec_plus_base when hub="hf"
MODEL_ID = "emotion2vec_plus_base"
TARGET_SR = 16000

# emotion2vec 9 classes: angry, disgusted, fearful, happy, neutral, other, sad, surprised, unknown
EMOTION2VEC_LABELS = [
    "angry", "disgusted", "fearful", "happy", "neutral",
    "other", "sad", "surprised", "unknown",
]

# Canonical 6 emotions (bridge expects these)
CANONICAL_EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad"]

# Map emotion2vec 9 -> canonical 6
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
    """
    emotion2vec_plus_base emotion model.
    infer(waveform) returns (probs, pred_idx) for canonical 6 emotions.
    """

    def __init__(self):
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        try:
            import logging
            import contextlib
            import io
            import sys
            from funasr import AutoModel
            from tqdm import tqdm
        except ImportError as e:
            raise ImportError(
                "Install: pip install funasr modelscope tqdm"
            ) from e
        # Suppress verbose init param logs
        for name in ("", "root", "funasr", "modelscope", "httpx"):
            logging.getLogger(name).setLevel(logging.ERROR)
        # Show progress bar during load (writes to stderr, not redirected)
        with tqdm(total=1, desc="Loading emotion model", bar_format="{desc}: {bar}|", file=sys.stderr) as pbar:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self._model = AutoModel(model=MODEL_ID, hub="hf")
            pbar.update(1)

    @property
    def emotions(self) -> list[str]:
        return list(CANONICAL_EMOTIONS)

    def infer(self, waveform: np.ndarray) -> tuple[np.ndarray, int]:
        self._ensure_loaded()
        import soundfile as sf

        if waveform.ndim == 2:
            waveform = waveform.squeeze(0)
        if waveform.dtype != np.float32:
            waveform = waveform.astype(np.float32)

        # emotion2vec expects 16 kHz; pad if too short
        min_samples = 16000  # 1 s
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
                import contextlib
                import io
                import sys
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    res = self._model.generate(
                        tmp_path,
                        output_dir=out_dir,
                        granularity="utterance",
                        extract_embedding=False,
                    )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        # res is list of dicts: [{"key": ..., "labels": [...], "scores": [...]}]
        if not res:
            probs_arr = np.ones(len(CANONICAL_EMOTIONS)) / len(CANONICAL_EMOTIONS)
            pred_idx = 0
            return probs_arr.astype(np.float32), pred_idx

        r = res[0]
        labels = r.get("labels", EMOTION2VEC_LABELS)
        scores = r.get("scores", [])

        # Build probs for emotion2vec 9
        probs_raw = np.zeros(9, dtype=np.float32)
        for i, lab in enumerate(labels):
            if i < len(scores):
                probs_raw[i] = float(scores[i])

        # Normalize
        s = probs_raw.sum()
        if s > 0:
            probs_raw = probs_raw / s

        # Map to canonical 6
        canonical_probs = {e: 0.0 for e in CANONICAL_EMOTIONS}
        for i, lab in enumerate(EMOTION2VEC_LABELS):
            if i < len(probs_raw):
                mapped = EMOTION2VEC_TO_CANONICAL.get(lab, "neutral")
                canonical_probs[mapped] += float(probs_raw[i])
        probs_arr = np.array([canonical_probs[e] for e in CANONICAL_EMOTIONS], dtype=np.float32)
        probs_arr = probs_arr / (probs_arr.sum() or 1e-9)

        # Neutral bias fix: reduce neutral dominance on ambiguous/mixed audio
        # 0.4 allows other emotions to win when model is uncertain
        NEUTRAL_PENALTY = 0.4
        neutral_idx = CANONICAL_EMOTIONS.index("neutral")
        probs_arr[neutral_idx] *= NEUTRAL_PENALTY
        probs_arr = probs_arr / (probs_arr.sum() or 1e-9)
        pred_idx = int(np.argmax(probs_arr))
        return probs_arr, pred_idx


_model_instance: Audio2EmotionModel | None = None


def get_model(model_dir: Path | None = None) -> Audio2EmotionModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = Audio2EmotionModel()
    return _model_instance
