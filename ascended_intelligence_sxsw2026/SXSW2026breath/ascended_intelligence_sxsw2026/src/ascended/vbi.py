"""
VBI (Vocal Breathiness Index) - Pause-Density model per client spec.
Algorithm for breath: pause ratio + spectral energy, no peak counting.
"""

import numpy as np

VBI_GAIN = 1.2  # Lowered from 1.4 to avoid jumping to High Anxiety too quickly
VBI_WEIGHT = 0.6
EMOTION_WEIGHT = 0.4

# Configurable noise floor for librosa.effects.split. Higher = less sensitivity to ambient noise.
_noise_floor_top_db = 30
CROWD_NOISE_RMS_THRESHOLD = 0.005  # RMS above this in silence → use top_db=35

BREATH_STATE_LABELS = {
    "calm": "Ascension",
    "love": "Love / Connection",
    "sadness_reflective": "Sadness / Reflective",
    "neutral_elevated": "Neutral / Elevated",
    "high_anxiety": "Fear / Liberation",
}


def get_noise_floor_top_db() -> float:
    """Return current top_db used for silence detection (default 30)."""
    return _noise_floor_top_db


def set_noise_floor_top_db(value: float) -> None:
    """Set top_db for silence detection. Use 35–40 for noisy rooms (e.g. AC, crowd)."""
    global _noise_floor_top_db
    _noise_floor_top_db = max(20.0, min(50.0, float(value)))


def adapt_noise_floor_for_silence(audio: np.ndarray, threshold: float = CROWD_NOISE_RMS_THRESHOLD) -> None:
    """
    When system is in silence (no one at mic), adjust top_db to ambient noise.
    Call before breath logic. Noisy crowd → top_db 35, quiet → 30.
    """
    rms = float(np.sqrt(np.mean(np.asarray(audio, dtype=np.float64).ravel() ** 2)) + 1e-12)
    set_noise_floor_top_db(35.0 if rms > threshold else 30.0)


def calibrate_noise_floor(audio: np.ndarray, sample_rate: int, duration_sec: float = 5.0) -> float:
    """
    Sample room tone to suggest top_db. Call with 5 sec of silence (no one at mic).
    Returns suggested top_db; use set_noise_floor_top_db() to apply.
    """
    try:
        import librosa
    except ImportError:
        return 30.0
    audio_flat = np.asarray(audio, dtype=np.float64).flatten()
    n = int(duration_sec * sample_rate)
    if len(audio_flat) < n:
        return 30.0
    chunk = audio_flat[-n:]
    rms = float(np.sqrt(np.mean(chunk ** 2)) + 1e-12)
    db = 20 * np.log10(rms + 1e-12)
    suggested = min(50.0, max(25.0, 35.0 - db))  # Noisier room → higher top_db
    return float(suggested)


def compute_vbi(audio: np.ndarray, sample_rate: int, gain: float = VBI_GAIN) -> float:
    """
    VBI (Vocal Breathiness Index) per new_instruction - Pause-Density model.
    - High pause_ratio + low breath_energy → low VBI → 963 Hz (Ascension/Calm)
    - Low pause_ratio + high breath_energy → high VBI → 396 Hz (Fear/Agitation)
    Uses configurable top_db (set_noise_floor_top_db) for room adaptation.
    """
    try:
        import librosa
    except ImportError:
        return 0.5
    audio_flat = np.asarray(audio, dtype=np.float64).flatten()
    if len(audio_flat) < sample_rate * 0.5:
        return 0.5
    non_silent = librosa.effects.split(audio_flat, top_db=_noise_floor_top_db)
    total_sound = sum(x[1] - x[0] for x in non_silent)
    pause_ratio = (len(audio_flat) - total_sound) / len(audio_flat)
    pause_ratio = max(0.0, min(1.0, float(pause_ratio)))
    S = np.abs(librosa.stft(audio_flat))
    n_bins = S.shape[0]
    start_bin = int(6000.0 / (sample_rate / 2.0) * n_bins)
    start_bin = min(start_bin, n_bins - 1)
    breath_energy = float(np.mean(S[start_bin:, :])) + 1e-12
    vbi = pow(breath_energy / (pause_ratio + 0.01), gain)
    return max(0.0, min(1.0, float(vbi)))


def vbi_to_target_hz(vbi: float, pred_id: int = -1) -> int:
    """
    Refined Frequency Codex mapping. Uses 40%% AI Emotion (pred_id) to break VBI ties.
    pred_id: 0=anger_fear, 1=joy_excited, 2=sadness, 3=curious_reflective, 4=calm_content.
    """
    if vbi >= 0.75:
        return 396   # Fear / Liberation (High Anxiety)
    if vbi < 0.25:
        return 963   # Ascension (Deep Calm)
    # Middle state: use pred_id for secondary emotional tinting
    if pred_id == 1:
        return 639   # Love / Connection (Joy)
    if pred_id == 2:
        return 528   # Sadness / Reflective (Grounding)
    return 639       # Neutral / Elevated (baseline)


def vbi_to_breath_state(vbi: float, pred_id: int = -1) -> str:
    """State labels for exhibition. pred_id used in middle range for tinting."""
    if vbi >= 0.75:
        return "high_anxiety"
    if vbi < 0.25:
        return "calm"
    if pred_id == 1:
        return "love"
    if pred_id == 2:
        return "sadness_reflective"
    return "neutral_elevated"


def vbi_bloom_modifier(vbi: float, pred_id: int) -> tuple[float, float]:
    """
    Apply exhibition modifiers: Joy → vbi*1.2 (boost bloom), Sadness → vbi*0.8 (mute).
    Returns (clamped_vbi, bloom) for OSC.
    """
    mod = 1.2 if pred_id == 1 else (0.8 if pred_id == 2 else 1.0)
    vbi_out = max(0.0, min(1.0, vbi * mod))
    bloom = 1.2 if pred_id == 1 else (0.8 if pred_id == 2 else 1.0)
    return (vbi_out, bloom)


def calculate_weighted_resonance(
    vbi_raw: float,
    emotion_index: int,
    vbi_weight: float = VBI_WEIGHT,
    emotion_weight: float = EMOTION_WEIGHT,
) -> float:
    """
    Client spec: 60% Breath (VBI) + 40% Emotion = Ritual Intensity.
    emotion_index: 0-4 (0=anger_fear, 4=calm_content).
    """
    normalized_emotion = 1.0 - (emotion_index / 4.0)
    weighted_score = (vbi_raw * vbi_weight) + (normalized_emotion * emotion_weight)
    return max(0.0, min(1.0, float(weighted_score)))
