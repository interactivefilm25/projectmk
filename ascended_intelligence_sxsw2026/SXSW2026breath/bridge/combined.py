"""
Bridge: VBI (from ascended) + Wav2Vec2 emotion model (TouchDesigner).
Client spec: Algorithm (VBI) for breath, Wav2Vec2 for emotion.
OSC output is handled by bridge/osc.py.
"""

import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASCENDED_SRC = PROJECT_ROOT / "ascended_intelligence_sxsw2026" / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ASCENDED_SRC) not in sys.path:
    sys.path.insert(0, str(ASCENDED_SRC))

from .osc import osc_client, configure_osc, EMOTION_MAP_REALTIME
from ascended.vbi import (
    compute_vbi,
    vbi_to_target_hz,
    vbi_to_breath_state,
    vbi_bloom_modifier,
    calculate_weighted_resonance,
    BREATH_STATE_LABELS,
    VBI_WEIGHT,
    EMOTION_WEIGHT,
)

TEMPORAL_BUFFER_SEC = 10.0
_osc_buffer: list[tuple[float, float, float, float]] = []  # (ts, f0, target_hz, vbi)

TARGET_SR = 16000
SEGMENT_DURATION = 2.0


def _override_probs(probs: dict, new_dominant: str, old_dominant: str) -> dict:
    """After F0-based emotion override, give new_dominant the confidence that old_dominant had."""
    out = dict(probs)
    out[new_dominant] = out.get(old_dominant, 0.0)
    out[old_dominant] = 0.0
    s = sum(out.values()) or 1e-9
    return {k: v / s for k, v in out.items()}


def _compute_f0(audio: np.ndarray, sample_rate: int) -> float:
    """F0 (fundamental frequency) using librosa.pyin. Used for emotion overrides."""
    try:
        import librosa
        f0, _, _ = librosa.pyin(
            audio.astype(np.float64),
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sample_rate,
        )
        valid = f0[~np.isnan(f0)]
        return float(np.mean(valid)) if len(valid) > 0 else 0.0
    except Exception:
        return 0.0


def _compute_spectral_centroid(audio: np.ndarray, sample_rate: int) -> float:
    """Spectral centroid (brightness). High = harsh/angry."""
    try:
        import librosa
        sc = librosa.feature.spectral_centroid(y=audio.astype(np.float32), sr=sample_rate)
        return float(np.mean(sc))
    except ImportError:
        return 0.0


def prepare_audio(waveform: np.ndarray, sample_rate: int, gentle: bool = False) -> np.ndarray:
    """
    Denoise audio using AI (DeepFilterNet2) only. No manual processing.

    gentle=True (live mic): Skip denoising, return waveform unchanged.
    gentle=False (recorded file): Run DeepFilterNet2. If unavailable, return unchanged.
    """
    if gentle:
        return waveform.astype(np.float32)

    try:
        from english_model import get_noise_cleaner
        cleaner = get_noise_cleaner()
        return cleaner.clean(waveform, sample_rate, target_sr=48000)
    except Exception:
        return waveform.astype(np.float32)


def load_audio(audio_path: Path | str, max_sec: float | None = None) -> tuple[np.ndarray, int]:
    """Load audio file to float32 mono at TARGET_SR. Returns (waveform, sample_rate)."""
    audio_path = Path(audio_path)
    suffix = audio_path.suffix.lower()
    try:
        import soundfile as sf
        waveform, sample_rate = sf.read(str(audio_path))
    except Exception:
        if suffix in (".m4a", ".aac", ".mp4"):
            import subprocess
            cmd = [
                "ffmpeg", "-y", "-i", str(audio_path),
                "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(TARGET_SR),
                "pipe:1",
            ]
            try:
                out = subprocess.run(cmd, capture_output=True, check=True, timeout=30)
                waveform = np.frombuffer(out.stdout, dtype=np.float32)
                sample_rate = TARGET_SR
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                raise RuntimeError("ffmpeg required for m4a (install: apt install ffmpeg)")
        else:
            import librosa
            waveform, sample_rate = librosa.load(str(audio_path), sr=None)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    waveform = waveform.astype(np.float32)
    if max_sec is not None:
        n = int(max_sec * sample_rate)
        waveform = waveform[:n]
    if sample_rate != TARGET_SR:
        try:
            import librosa
            waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=TARGET_SR)
        except ImportError:
            from scipy import signal
            num = int(len(waveform) * TARGET_SR / sample_rate)
            waveform = signal.resample(waveform, num).astype(np.float32)
        sample_rate = TARGET_SR
    return waveform, sample_rate


def run_combined(
    audio_path: Path | str | None = None,
    waveform: np.ndarray | None = None,
    sample_rate: int | None = None,
    segment_duration: float | None = None,
    full_audio_emotion: bool = True,
    source: str = "file",
    clean_audio: bool = True,
) -> list[dict]:
    """
    Run both BreathDetector and Audio2Emotion on the same audio.
    Pass either audio_path or (waveform, sample_rate). Waveform must be float32, 16 kHz.

    segment_duration: Seconds per output segment. If None, returns one result for full audio.
    full_audio_emotion: If True, run emotion model on the entire audio once (default).
    source: "file" (default) or "live". When "live" and audio > 5s, use segment-based
            emotion (pick most confident 3s window) to handle mic/playback degradation.
    clean_audio: If True, run DeepFilterNet2 noise cleaning (downloads from HF locally).

    Returns list of segment dicts:
      - start_time, end_time, breath_emotion, primary_emotion, target_frequency_hz
      - vbi (weighted), breath_f0, audio2emotion_emotion, audio2emotion_probs
    """
    if audio_path is not None:
        waveform, sample_rate = load_audio(audio_path)
    elif waveform is not None and sample_rate is not None:
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if sample_rate != TARGET_SR:
            import librosa
            waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=TARGET_SR)
            sample_rate = TARGET_SR
    else:
        raise ValueError("Pass either audio_path or (waveform, sample_rate)")

    # Optional: clean + enhance (gentle, preserves quality)
    if clean_audio:
        waveform = prepare_audio(waveform, sample_rate)

    from english_model import get_model

    model = get_model()
    emotions_a2e = model.emotions

    duration = len(waveform) / sample_rate
    if segment_duration is None:
        segment_duration = duration
    num_segments = max(1, int(np.ceil(duration / segment_duration)))

    # Run emotion model on full audio
    a2e_probs_full = None
    a2e_dominant_full = None
    if full_audio_emotion:
        min_samples = 32000
        full_chunk = waveform.astype(np.float32)
        if len(full_chunk) < min_samples:
            full_chunk = np.pad(
                full_chunk, (0, min_samples - len(full_chunk)),
                mode="constant", constant_values=0,
            )
        probs, pred_idx = model.infer(full_chunk)
        a2e_dominant_full = emotions_a2e[pred_idx]
        a2e_probs_full = {emotions_a2e[i]: float(probs[i]) for i in range(len(emotions_a2e))}

    combined = []
    for seg_idx in range(num_segments):
        start_time = seg_idx * segment_duration
        end_time = min((seg_idx + 1) * segment_duration, duration)
        seg_start = int(start_time * sample_rate)
        seg_end = min(int(end_time * sample_rate), len(waveform))
        seg_audio_full = waveform[seg_start:seg_end]
        if len(seg_audio_full) < sample_rate * 0.5:
            continue

        # F0 from segment (librosa.pyin) - for emotion overrides
        breath_f0 = _compute_f0(seg_audio_full, sample_rate)

        # Audio2Emotion: use full-audio result or run per segment
        if full_audio_emotion and a2e_probs_full is not None:
            a2e_dominant = a2e_dominant_full
            a2e_probs = a2e_probs_full
        else:
            min_samples = 32000
            start_idx = int(start_time * sample_rate)
            end_idx = int(end_time * sample_rate)
            end_idx = min(end_idx, len(waveform))
            start_idx = min(start_idx, end_idx)
            chunk = waveform[start_idx:end_idx].astype(np.float32)
            if len(chunk) == 0:
                chunk = np.zeros(min_samples, dtype=np.float32)
            elif len(chunk) < min_samples:
                chunk = np.pad(
                    chunk,
                    (0, min_samples - len(chunk)),
                    mode="constant",
                    constant_values=0,
                )
            probs, pred_idx = model.infer(chunk)
            a2e_dominant = emotions_a2e[pred_idx]
            a2e_probs = {emotions_a2e[i]: float(probs[i]) for i in range(len(emotions_a2e))}

        seg_duration = len(seg_audio_full) / sample_rate
        centroid = _compute_spectral_centroid(seg_audio_full, sample_rate)

        vbi_raw = compute_vbi(seg_audio_full, sample_rate)
        emotion_index = EMOTION_MAP_REALTIME.get(a2e_dominant.lower(), 2)
        weighted_vbi = calculate_weighted_resonance(vbi_raw, emotion_index)
        _prelim_breath_state = vbi_to_breath_state(weighted_vbi)
        target_hz = vbi_to_target_hz(weighted_vbi)

        # Overrides: model often conflates high-arousal (anger/fear/sad) as joy
        a2e_lower = a2e_dominant.lower() if isinstance(a2e_dominant, str) else ""
        if a2e_lower == "joy_excited" and 0 < breath_f0 < 100 and centroid > 850:
            # Calm: low F0, brighter timbre (centroid > 850) → calm_content (not Love)
            a2e_dominant = "calm_content"
            a2e_probs = _override_probs(a2e_probs, "calm_content", "joy_excited")
        elif a2e_lower == "joy_excited" and 180 <= breath_f0 <= 250 and centroid < 2200:
            # Sadness: mid-high F0, softer timbre (before fear which has F0 > 250)
            a2e_dominant = "sadness"
            a2e_probs = _override_probs(a2e_probs, "sadness", "joy_excited")
        elif a2e_lower == "joy_excited" and breath_f0 > 250:
            # Fear: very high F0
            a2e_dominant = "anger_fear"
            a2e_probs = _override_probs(a2e_probs, "anger_fear", "joy_excited")
        elif a2e_lower == "joy_excited" and (centroid > 1550 or (vbi_raw > 0.3 and 110 <= breath_f0 <= 170) or (1100 <= centroid <= 1400 and 120 <= breath_f0 <= 150)):
            # Anger: harsh timbre, or agitated VBI, or mid-bright voice (centroid 1100-1400)
            a2e_dominant = "anger_fear"
            a2e_probs = _override_probs(a2e_probs, "anger_fear", "joy_excited")
        elif a2e_lower == "joy_excited" and _prelim_breath_state == "high_anxiety":
            a2e_dominant = "anger_fear"
            a2e_probs = _override_probs(a2e_probs, "anger_fear", "joy_excited")
        elif a2e_lower == "calm_content" and 100 <= breath_f0 <= 120 and 900 <= centroid <= 1100:
            # Low-pitched fear: calm predicted but tense (F0 100–120, centroid 900–1100) → fear
            a2e_dominant = "anger_fear"
            a2e_probs = _override_probs(a2e_probs, "anger_fear", "calm_content")
        elif a2e_lower == "calm_content" and 110 <= breath_f0 <= 135 and 1100 <= centroid <= 1400:
            # Soft happy: calm predicted but brighter timbre (F0 110–135, centroid 1100–1400) → joy
            a2e_dominant = "joy_excited"
            a2e_probs = _override_probs(a2e_probs, "joy_excited", "calm_content")
        elif a2e_lower == "calm_content" and 80 <= breath_f0 <= 100 and centroid < 850:
            # Love: calm predicted but very soft timbre (centroid < 850) + low F0 → joy (tender)
            a2e_dominant = "joy_excited"
            a2e_probs = _override_probs(a2e_probs, "joy_excited", "calm_content")
        elif a2e_lower == "sadness" and 70 <= breath_f0 <= 180 and vbi_raw < 0.2:
            # Calm speech misclassified as sadness (e.g. Calm Nigel)
            a2e_dominant = "calm_content"
            a2e_probs = _override_probs(a2e_probs, "calm_content", "sadness")
        elif a2e_lower == "anger_fear" and _prelim_breath_state != "high_anxiety" and (
            breath_f0 > 165 or (centroid > 0 and centroid < 1550)
        ) and centroid < 1600 and not (1100 <= centroid <= 1400 and 120 <= breath_f0 <= 150):
            # Gentle timbre → happy, but NOT centroid 1100–1400 + F0 120–150 (true anger)
            a2e_dominant = "joy_excited"
            a2e_probs = _override_probs(a2e_probs, "joy_excited", "anger_fear")

        # Emotion-based VBI alignment: calm < 0.25, fear >= 0.75, middle uses pred_id for tinting
        a2e_final = a2e_dominant.lower()
        pred_id = EMOTION_MAP_REALTIME.get(a2e_final, 2)
        if a2e_final in ("calm_content", "calm"):
            weighted_vbi = min(weighted_vbi, 0.24)
        elif a2e_final in ("anger_fear", "angry", "fear"):
            weighted_vbi = max(weighted_vbi, 0.76)
        elif a2e_final in ("joy_excited", "sadness"):
            weighted_vbi = max(0.25, min(0.74, weighted_vbi))  # Middle: pred_id differentiates
        else:
            weighted_vbi = max(0.25, min(0.74, weighted_vbi))

        _prelim_breath_state = vbi_to_breath_state(weighted_vbi, pred_id)
        target_hz = vbi_to_target_hz(weighted_vbi, pred_id)
        breath_dominant = _prelim_breath_state
        if a2e_final in ("anger_fear", "angry", "fear") and centroid > 2500.0:
            breath_dominant = "high_anxiety"
            target_hz = 396

        primary_emotion = BREATH_STATE_LABELS[breath_dominant]

        segment_result = {
            "start_time": start_time,
            "end_time": end_time,
            "breath_emotion": breath_dominant,
            "primary_emotion": primary_emotion,
            "target_frequency_hz": target_hz,
            "vbi": weighted_vbi,
            "breath_f0": breath_f0,
            "audio2emotion_emotion": a2e_dominant,
            "audio2emotion_probs": a2e_probs,
        }
        combined.append(segment_result)

        now = time.time()
        _osc_buffer.append((now, breath_f0, float(target_hz), weighted_vbi))
        cutoff = now - TEMPORAL_BUFFER_SEC
        _osc_buffer[:] = [(ts, f, h, v) for ts, f, h, v in _osc_buffer if ts >= cutoff - 5]
        recent = [(f, h, v) for ts, f, h, v in _osc_buffer if ts >= cutoff]
        if not recent:
            recent = [(breath_f0, target_hz, weighted_vbi)]
        f0_arr, hz_arr, vbi_arr = zip(*recent)
        smoothed_target_hz = float(np.mean(hz_arr))
        smoothed_vbi = float(np.mean(vbi_arr))
        vbi_osc, bloom = vbi_bloom_modifier(smoothed_vbi, pred_id)

        osc_client.send(
            emotion=a2e_dominant,
            target_hz=int(round(smoothed_target_hz)),
            vbi=vbi_osc,
            bloom=bloom,
        )

    return combined
