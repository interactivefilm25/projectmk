import librosa
import time
import torch
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from typing import List, Dict, Tuple
import numpy as np
import opensmile
import pandas as pd

print("Model Loader: Initializing and loading models...")
start_time = time.time()

# These lines will run ONCE when this script is first compiled by TouchDesigner.
# This may cause a one-time freeze on startup, which is expected.

#model_id = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
model_id = "superb/hubert-large-superb-er"
#model_id = "superb/wav2vec2-base-superb-er"

model = AutoModelForAudioClassification.from_pretrained(model_id)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, do_normalize=True)
id2label = model.config.id2label
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

if device.type == 'cuda':
	model = model.half()
	print("Using half precision (FP16) on GPU")

print(f"Model loaded to {device} in {time.time() - start_time:.2f} seconds.")

def preprocess_numpy_audio(audio_array, sampling_rate, feature_extractor):
    """Preprocesses a NumPy audio array directly from a TouchDesigner CHOP."""
    if audio_array.ndim > 1:
        audio_array = audio_array.flatten()
        
    inputs = feature_extractor(
        audio_array,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=True,
    )
    return inputs
        
def predict_emotion_from_numpy(audio_array, sampling_rate):
    global model, feature_extractor, id2label, device
    try:
        print("Predict Func]: Starting preprocessing...")
        inputs = preprocess_numpy_audio(audio_array, sampling_rate, feature_extractor)
        inputs = {key: val.to(device) for key, val in inputs.items()}

        if model.dtype == torch.float16:
            inputs = {key: val.half() for key, val in inputs.items()}

        with torch.no_grad():
            print("[Predict Func]: >>> Entering model inference.")
            outputs = model(**inputs)
            print("[Predict Func]: <<< Exited model inference.")

        logits = outputs.logits
        predicted_id = torch.argmax(logits, dim=-1).item()
        predicted_label = id2label[predicted_id]

        audio_analysis = analyze_emotion_from_numpy(audio_array, sampling_rate)

        # Only return dict if both are present
        if audio_analysis and predicted_label:
            return {
                "predicted_id": predicted_id,
                "predicted_label": predicted_label,
                "audio_analysis": audio_analysis
            }

    except Exception as e:
        print(f"Error inside predict_emotion_from_numpy: {e}")
        import traceback
        traceback.print_exc()
        return f"ERROR: {e}"
     
# --- openSmile analysis functions ---

# 1) Emotion Frequency Mapping (Hz) — from document
EMOTION_F0 = {
    "Calm":    (85, 150),
    "Sadness": (90, 170),
    "Joy":     (180, 300),
    "Fear":    (300, 500),
    "Anger":   (120, 250),
    "Love":    (150, 250),
    "Trust":   (130, 200),
    "Disgust": (100, 200),
    "Neutral": (100, 220),
}

# Optional Valence/Arousal mapping (directional, not absolute)
# Valence in [-1, 1], Arousal in [0, 1]
VAL_AROUSAL = {
    "Joy":     {"valence":  +1.0, "arousal": 0.90},
    "Sadness": {"valence":  -1.0, "arousal": 0.20},
    "Anger":   {"valence":  -1.0, "arousal": 0.90},
    "Calm":    {"valence":  +0.8, "arousal": 0.25},
    "Fear":    {"valence":  -1.0, "arousal": 0.90},
    "Trust":   {"valence":  +0.8, "arousal": 0.50},
    # Not in the table, inferred from doc's descriptions:
    "Love":    {"valence":  +1.0, "arousal": 0.55},
    "Disgust": {"valence":  -1.0, "arousal": 0.50},
    "Neutral": {"valence":   0.0, "arousal": 0.40},
}

# 2) openSMILE setup (eGeMAPSv02 Functionals)
SMILE = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

def safe_mean(x: np.ndarray) -> float:
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else float("nan")


def hz_pitch(y: np.ndarray, sr: int, fmin: float = 50.0, fmax: float = 500.0) -> np.ndarray:
    """
    Robust F0 estimation in Hz using librosa.pyin; falls back to librosa.yin.
    """
    try:
        f0, _, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr)
    except Exception:
        f0 = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr)
    f0 = np.asarray(f0, dtype=float)
    # Replace NaNs (unvoiced) by interpolation where possible
    if np.isnan(f0).any():
        idx = np.arange(len(f0))
        good = np.isfinite(f0)
        if good.any():
            f0 = np.interp(idx, idx[good], f0[good])
        else:
            f0[:] = np.nan
    return f0


def get_feature(df: pd.DataFrame, candidates: List[str], default: float = np.nan) -> float:
    """
    Retrieve first matching column; otherwise attempt loose search;
    finally return default.
    """
    for c in candidates:
        if c in df.columns:
            return float(df[c].values[0])
    # loose search
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        for name_lower, real in lower.items():
            if c.lower() in name_lower:
                return float(df[real].values[0])
    return float(default)


def normalize_pos(x: float, denom: float) -> float:
    """Clamp (x/denom) to [0,1]."""
    if not np.isfinite(x) or denom <= 0:
        return 0.0
    return float(np.clip(x / denom, 0, 1))


def exp_decay_low_is_good(x: float, scale: float) -> float:
    """
    Convert a positive metric where 'lower is better' (e.g., jitter/shimmer)
    into [0,1] score, with ~0.02 producing a moderate score if scale=0.02.
    """
    if not np.isfinite(x) or x < 0:
        return 0.0
    return float(np.exp(-x / max(scale, 1e-8)))


def trapezoid_membership(mean_f0: float, low: float, high: float, margin_ratio: float = 0.15) -> float:
    """
    Soft membership in [0,1] to encourage smoothness near boundaries.
    Inside [low, high] -> 1.0; decays linearly outside within a margin.
    """
    if not np.isfinite(mean_f0):
        return 0.0
    width = high - low
    margin = max(10.0, width * margin_ratio)
    if low <= mean_f0 <= high:
        return 1.0
    elif mean_f0 < low:
        return max(0.0, 1.0 - (low - mean_f0) / margin)
    else:
        return max(0.0, 1.0 - (mean_f0 - high) / margin)


def vibrational_rating(f0: np.ndarray, jitter: float, shimmer: float, hnr: float, mfccs: np.ndarray) -> float:
    """
    Based on the doc: combine F0 stability ↑, low jitter/shimmer, high HNR, timbre richness ↑.
    Returns 0..100.
    """
    f0 = np.asarray(f0, dtype=float)
    f0 = f0[np.isfinite(f0)]
    if f0.size == 0:
        return 0.0

    mean_f0 = np.mean(f0)
    cv = np.std(f0) / mean_f0 if mean_f0 > 0 else np.inf
    pitch_stability = float(np.clip(1.0 - cv, 0, 1))

    jitter_score = exp_decay_low_is_good(jitter, scale=0.02)   # 0.02 ~ typical small jitter
    shimmer_score = exp_decay_low_is_good(shimmer, scale=0.02) # 0.02 ~ typical small shimmer
    hnr_score = normalize_pos(hnr, 40.0)                       # clamp ~40 dB

    # Timbre richness via MFCC magnitude
    mfcc_mag = float(np.mean(np.abs(mfccs)))
    timbre_richness = normalize_pos(mfcc_mag, 100.0)           # heuristic scaling

    score = (
        0.30 * pitch_stability +
        0.20 * jitter_score +
        0.20 * shimmer_score +
        0.20 * hnr_score +
        0.10 * timbre_richness
    )
    return round(score * 100.0, 2)

def analyze_emotion_from_numpy(y: np.ndarray, sr: int) -> dict:
    f0 = hz_pitch(y, sr, fmin=50.0, fmax=500.0)
    mean_f0 = safe_mean(f0)

    # ---- contour & energy features for heuristics
    # RMS energy (0..1-ish), spectral centroid (0..sr/2)
    rms = librosa.feature.rms(y=y).squeeze()
    energy_mean = float(np.clip(np.mean(rms), 0, 1))  
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr).squeeze()
    centroid_n = normalize_pos(np.mean(centroid), sr / 2)

    # Pitch dynamics
    # slope magnitude and sign as crude "rising" / "flat" indicator
    t = np.arange(len(f0))
    if np.isfinite(f0).all() and len(f0) > 1:
        # simple linear regression slope
        slope = float(np.polyfit(t, f0, 1)[0])
    else:
        slope = 0.0
    slope_abs = float(min(abs(slope) / 5.0, 1.0))  # heuristic scaling
    slope_pos = float(max(np.sign(slope), 0.0)) * min(abs(slope) / 5.0, 1.0)

    # ---- openSMILE features (eGeMAPS Functionals)
    df = SMILE.process_signal(signal=y, sampling_rate=sr)

    jitter = get_feature(df, [
        "jitterLocal_sma_amean",
        "jitterLocal_amean",
        "jitterDDP_sma_amean",
    ], default=np.nan)

    shimmer = get_feature(df, [
        "shimmerLocal_sma_amean",
        "shimmerLocal_amean",
    ], default=np.nan)

    hnr = get_feature(df, [
        "HNRdBACF_sma_amean",
        "HNRdBACF_amean",
        "HNRdBF_sma_amean",
    ], default=np.nan)

    # MFCC timbre richness (13 coefs)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # ---- normalize “micro” scores
    jitter_low_good = exp_decay_low_is_good(jitter, 0.02)
    shimmer_low_good = exp_decay_low_is_good(shimmer, 0.02)
    hnr_high_good = normalize_pos(hnr, 40.0)
    noisiness = 1.0 - hnr_high_good

    # stability (1 - CV of f0)
    f0_stability = 0.0
    if np.isfinite(mean_f0) and mean_f0 > 0:
        f0_stability = float(np.clip(1.0 - (np.std(f0) / mean_f0), 0, 1))

    # ---- base pitch membership from F0 ranges
    base_pitch_membership: Dict[str, float] = {}
    for emo, (lo, hi) in EMOTION_F0.items():
        base_pitch_membership[emo] = trapezoid_membership(mean_f0, lo, hi)

    # ---- micro-feature adjustments per emotion (0..1)
    adj: Dict[str, float] = {}
    adj["Calm"] = (0.30 * f0_stability + 0.30 * hnr_high_good +
                   0.20 * jitter_low_good + 0.20 * shimmer_low_good -
                   0.20 * energy_mean)
    adj["Sadness"] = (0.30 * (1 - energy_mean) + 0.30 * (1 - hnr_high_good) +
                      0.20 * f0_stability + 0.20 * (1 - slope_abs))
    adj["Joy"] = (0.30 * hnr_high_good + 0.20 * f0_stability +
                  0.20 * slope_pos + 0.20 * centroid_n + 0.10 * energy_mean)
    adj["Fear"] = (0.30 * noisiness + 0.30 * (1 - jitter_low_good) +
                   0.20 * (1 - shimmer_low_good) + 0.20 * slope_abs)
    adj["Anger"] = (0.30 * energy_mean + 0.20 * centroid_n +
                    0.20 * (1 - jitter_low_good) + 0.10 * (1 - shimmer_low_good) +
                    0.20 * slope_abs)
    adj["Love"] = (0.30 * hnr_high_good + 0.30 * f0_stability +
                   0.20 * jitter_low_good + 0.20 * shimmer_low_good)
    adj["Trust"] = (0.40 * f0_stability + 0.30 * hnr_high_good +
                    0.10 * jitter_low_good + 0.10 * shimmer_low_good +
                    0.10 * (1 - slope_abs))
    adj["Disgust"] = (0.40 * noisiness + 0.20 * (1 - jitter_low_good) +
                      0.20 * (1 - shimmer_low_good) + 0.20 * (1 - f0_stability))
    adj["Neutral"] = (0.40 * (1 - slope_abs) + 0.20 * f0_stability +
                      0.20 * hnr_high_good + 0.20 * (1 - energy_mean))
    
    # ---- final emotion scores (blend of pitch membership and micro features)
    scores: Dict[str, float] = {}
    for emo in EMOTION_F0:
        base = base_pitch_membership[emo]
        micro = float(np.clip(adj[emo], 0, 1))
        scores[emo] = float(np.clip(0.6 * base + 0.4 * micro, 0, 1))
    
    # rank
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_emotion, top_score = ranked[0]

    # vibrational rating (0..100)
    vib = vibrational_rating(f0, jitter, shimmer, hnr, mfccs)

    # valence/arousal from dominant emotion
    va = VAL_AROUSAL.get(top_emotion, {"valence": 0.0, "arousal": 0.5})

    return {
        "sample_rate": sr,
        "mean_pitch_Hz": round(mean_f0, 2) if np.isfinite(mean_f0) else 0,
        "jitter_local": 0 if not np.isfinite(jitter) else round(float(jitter), 6),
        "shimmer_local": 0 if not np.isfinite(shimmer) else round(float(shimmer), 6),
        "HNR_dB": 0 if not np.isfinite(hnr) else round(float(hnr), 3),
        "energy_mean": round(float(energy_mean), 4),
        "spectral_centroid_norm": round(float(centroid_n), 4),
        "f0_stability": round(float(f0_stability), 4),
        "vibrational_rating": vib,
        "emotion_scores": {k: round(v, 4) for k, v in scores.items()},
        "top_emotion": {"label": top_emotion, "score": round(top_score, 4)},
        "valence": va["valence"],
        "arousal": va["arousal"], 
    }

    # return {
    #     "sample_rate": sr,
    #     "mean_pitch_Hz": round(mean_f0, 2) if np.isfinite(mean_f0) else None,
    #     "jitter_local": None if not np.isfinite(jitter) else round(float(jitter), 6),
    #     "shimmer_local": None if not np.isfinite(shimmer) else round(float(shimmer), 6),
    #     "HNR_dB": None if not np.isfinite(hnr) else round(float(hnr), 3),
    #     "energy_mean": round(float(energy_mean), 4),
    #     "spectral_centroid_norm": round(float(centroid_n), 4),
    #     "f0_stability": round(float(f0_stability), 4),
    #     "vibrational_rating": vib,
    #     "emotion_scores": {k: round(v, 4) for k, v in scores.items()},
    #     "top_emotion": {"label": top_emotion, "score": round(top_score, 4)},
    #     "valence": va["valence"],
    #     "arousal": va["arousal"], 
    # }

# --- These functions are helpers to get the pre-loaded objects ---
def getModel():
	return model
