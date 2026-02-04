"""
Bridge: combined BreathDetector (OpenSMILE) + Wav2Vec2 emotion model.
Runs both pipelines on the same audio and returns both results per segment.
No ensemble — both outputs are kept separate per PDF specification.
OSC output is handled by bridge/osc.py.
Use test_recorded_audio.py to test.
"""

import sys
from pathlib import Path

import numpy as np

# Import OSC client from separate module
from .osc import osc_client, configure_osc

# Project root and ascended path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASCENDED_SRC = PROJECT_ROOT / "ascended_intelligence_sxsw2026" / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ASCENDED_SRC) not in sys.path:
    sys.path.insert(0, str(ASCENDED_SRC))

TARGET_SR = 16000
SEGMENT_DURATION = 2.0  # seconds per segment for combined output

# Strict BPM→emotion map from breath_rate.jpg and pipeline images:
#   Under 20  = Calm and relaxed
#   20–25     = Slightly elevated (normal conversation)
#   25–30     = Anxious or stressed
#   Over 30   = High anxiety
# No "neutral" – breath state always maps to one of these four.

# Fallback BPM when breath detector returns 0 (short/continuous speech).
# Never use neutral; bias toward slightly_elevated (22) as default.
EMOTION_BPM_FALLBACK = {
    "angry": 32.0,
    "fear": 32.0,
    "happy": 24.0,
    "disgust": 22.0,
    "neutral": 22.0,  # → slightly_elevated, not calm
    "sad": 18.0,
}


def _bpm_fallback(breath_bpm: float, audio2emotion_dominant: str) -> float:
    """When breath detector returns 0, use emotion-based BPM estimate."""
    if breath_bpm > 0:
        return breath_bpm
    return EMOTION_BPM_FALLBACK.get(
        audio2emotion_dominant.lower(),
        22.0,  # slightly_elevated, avoid neutral-like output
    )


def _bpm_to_breath_state(bpm: float) -> str:
    """Strict BPM→breath state per breath_rate.jpg. No neutral."""
    if bpm <= 0:
        return "slightly_elevated"  # unknown → default to elevated, not calm
    elif bpm < 20.0:
        return "calm"
    elif bpm < 25.0:
        return "slightly_elevated"
    elif bpm < 30.0:
        return "anxious"
    else:
        return "high_anxiety"


# Human-readable labels for display (from breath_rate.jpg)
BREATH_STATE_LABELS = {
    "calm": "Calm and relaxed",
    "slightly_elevated": "Slightly elevated",
    "anxious": "Anxious or stressed",
    "high_anxiety": "High anxiety",
}

# Breath state → target frequency (target_frequency.jpg): 396 Fear, 639 Love, 963 Ascension
BREATH_STATE_TO_TARGET_HZ = {
    "calm": 639,
    "slightly_elevated": 639,
    "anxious": 396,
    "high_anxiety": 396,
}


def prepare_audio(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Clean and enhance audio for emotion model.
    - Denoise (DeepFilterNet2 or noisereduce)
    - Enhance (rumble filter + noise reduction + peak norm)
    """
    from english_model.noise_model import enhance_audio

    # 1. Denoise (optional - may fail if DeepFilterNet not available)
    try:
        from english_model import get_noise_cleaner
        cleaner = get_noise_cleaner()
        waveform = cleaner.clean(waveform, sample_rate, target_sr=48000)
    except Exception:
        pass

    # 2. Always enhance (rumble filter + light NR + normalize)
    waveform = enhance_audio(waveform, sample_rate)
    return waveform


def load_audio(audio_path: Path | str, max_sec: float | None = None) -> tuple[np.ndarray, int]:
    """Load audio file to float32 mono at TARGET_SR. Returns (waveform, sample_rate)."""
    audio_path = Path(audio_path)
    try:
        import soundfile as sf
        waveform, sample_rate = sf.read(str(audio_path))
    except Exception:
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
      - start_time, end_time
      - breath_emotion: BPM-based state (calm, slightly_elevated, anxious, high_anxiety)
      - breath_f0: mean F0 (Hz) in segment
      - breath_bpm: mean breath rate (BPM) in segment
      - audio2emotion_emotion: dominant emotion from Audio2Emotion
      - audio2emotion_probs: dict of emotion -> probability
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

    from ascended.breath_detector import BreathDetector
    from english_model import get_model

    detector = BreathDetector(sample_rate=TARGET_SR, use_opensmile=True)
    model = get_model()
    emotions_a2e = model.emotions

    duration = len(waveform) / sample_rate
    # segment_duration=None means one result for full audio (no segmentation)
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

    # Process breath in 100ms chunks (full audio, in order)
    chunk_size = detector.chunk_size
    num_chunks = len(waveform) // chunk_size
    breath_results = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        chunk = waveform[start_idx:end_idx]
        result = detector.process_chunk(chunk)
        breath_results.append(result)

    # Build output per segment
    chunks_per_second = detector.CHUNKS_PER_SECOND
    combined = []
    for seg_idx in range(num_segments):
        start_time = seg_idx * segment_duration
        end_time = min((seg_idx + 1) * segment_duration, duration)
        seg_start_chunk = int(start_time * chunks_per_second)
        seg_end_chunk = int(end_time * chunks_per_second)
        seg_breath = breath_results[seg_start_chunk:seg_end_chunk]
        if not seg_breath:
            continue

        # Breath: BPM-based emotional state (per PDF)
        f0_list = []
        bpm_list = []
        for r in seg_breath:
            f0 = r.get("emotion", {}).get("f0_hz", 0.0)
            if f0 > 0:
                f0_list.append(f0)
            bpm = r.get("breath_rate_bpm", 0.0)
            if bpm > 0:
                bpm_list.append(bpm)
        breath_f0 = float(np.mean(f0_list)) if f0_list else 0.0
        breath_bpm_raw = float(np.mean(bpm_list)) if bpm_list else 0.0

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

        # Use emotion-based BPM fallback when detector returns 0
        breath_bpm = _bpm_fallback(breath_bpm_raw, a2e_dominant)

        # Derive breath state from BPM (strict map from breath_rate.jpg – no neutral)
        breath_dominant = _bpm_to_breath_state(breath_bpm)
        primary_emotion = BREATH_STATE_LABELS[breath_dominant]
        target_hz = BREATH_STATE_TO_TARGET_HZ[breath_dominant]

        segment_result = {
            "start_time": start_time,
            "end_time": end_time,
            "breath_emotion": breath_dominant,
            "primary_emotion": primary_emotion,
            "target_frequency_hz": target_hz,
            "breath_f0": breath_f0,
            "breath_bpm": breath_bpm,
            "audio2emotion_emotion": a2e_dominant,
            "audio2emotion_probs": a2e_probs,
        }
        combined.append(segment_result)
        
        # Send via OSC
        top_confidence = max(a2e_probs.values()) if a2e_probs else 0.0
        osc_client.send(
            emotion=a2e_dominant,
            confidence=top_confidence,
            f0=breath_f0,
            breath_state=primary_emotion,
            breath_bpm=breath_bpm,
        )

    return combined
