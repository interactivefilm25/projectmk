# ASCENDED Intelligence: Audio-Only Breath Detection

This package provides **audio-only breath detection** and **F0-based emotion mapping** for real-time or offline use. It processes audio in fixed **100 ms chunks**, keeps a **15-second rolling history**, detects breath-like events in gaps (silence + low-frequency "whoosh"), and maps **fundamental frequency (F0)** to emotion labels and valence/arousal. Output is formatted for **TouchDesigner** (normalized BPM, confidence, anxiety/calm flags, emotion, valence, arousal). It is used by the project **bridge** to supply the Breath side of the combined pipeline (Breath + emotion model).

---

## 1. Overview

- **Input**: Mono audio at **16 kHz**, supplied as consecutive **100 ms chunks** (1600 samples per chunk at 16 kHz).
- **Processing**: Each chunk is passed through **OpenSMILE** (eGeMAPSv02 functionals) to extract features (F0, loudness, RMS, low-frequency bands, jitter). A **15-second history** of chunk features is maintained. The detector finds **gaps** (silence where voice is absent), checks them with a **breath acoustic fingerprint** (low-frequency energy + RMS in a valid range), and computes **breath rate (BPM)** from intervals between detected breath events. **F0** from each chunk is converted to Hz and mapped to **emotion** via fixed F0 ranges (e.g. Fear 300–500 Hz, Neutral 90–130 Hz) and to **valence/arousal**.
- **Output**: For each chunk you get a single dict: normalized breath rate (0–1), confidence (0.3–0.8), anxiety_level (0 or 1), calm_level (0 or 1), timestamp, **breath_rate_bpm**, and **emotion** (f0_hz, top_emotion, emotion_scores, valence, arousal). This is the format consumed by the bridge when aggregating over 2 s segments.

The design follows a PDF specification (Breath_Sensing_Data_Pipeline) and is intended for real-time streaming (e.g. 10 chunks per second) or batch processing by feeding chunks in order.

---

## 2. Package layout

- **`src/ascended/breath_detector.py`** – Main class `BreathDetector` and all logic (chunk processing, history, gap detection, breath fingerprint, BPM, F0-to-emotion, TouchDesigner output).
- **`src/ascended/opensmile_wrapper.py`** – Thin wrapper around the OpenSMILE library: exposes a `Smile` instance configured with eGeMAPSv02 functionals; used by `BreathDetector` to extract features per chunk.
- **`src/ascended/tests/`** – Unit tests for the ascended package.
- **`example_streaming.py`** – Example that initializes the detector and simulates streaming chunks.
- **`requirements.txt`** – Python dependencies for this package (librosa, numpy, scipy, opensmile, soundfile, etc.).

OpenSMILE itself is provided by the project’s **opensmile-python-main** (or an installed `opensmile` package). The bridge and tests run from the **project root** and add the project root and `ascended_intelligence_sxsw2026/src` to `sys.path` so that `ascended.breath_detector` and `ascended.opensmile_wrapper` can be imported.

---

## 3. How it works (in depth)

### 3.1 Chunk-based processing

- **Chunk duration**: 100 ms (`CHUNK_DURATION = 0.1`).
- **Chunk size**: `chunk_size = int(0.1 * sample_rate)`; at 16 kHz this is **1600 samples**.
- Audio must be supplied as consecutive chunks (e.g. from a stream or by slicing a longer buffer). The detector does not load files; the bridge or your code is responsible for loading/resampling and slicing.

### 3.2 OpenSMILE features

- **Feature set**: eGeMAPSv02, **functionals** (one vector per processed segment).
- **Required features** (used inside the detector):
  - **Loudness_sma3** – loudness (smoothed).
  - **F0semitoneFrom27.5Hz_sma3nz** – F0 in semitones from 27.5 Hz (used for voice activity and for F0-to-emotion; converted to Hz via 27.5 * 2^(semitones/12)).
  - **pcm_RMSenergy_sma** – RMS energy (computed in code if not present).
  - **audSpec_Rfilt_sma3[0], [1], [2]** – low/mid frequency bands used for breath “whoosh” detection (proxies from eGeMAPSv02 such as slope and alpha ratio if exact names are missing).
  - **jitterLocal_sma3nz** – jitter (voice quality).

The wrapper in `opensmile_wrapper.py` uses the OpenSMILE library’s `Smile` with `FeatureSet.eGeMAPSv02` and `FeatureLevel.Functionals`. Each 100 ms chunk is passed to `smile.process_signal()`; the result is parsed into the required feature names and validated (NaN replaced by 0, missing names filled).

### 3.3 History buffer and breath detection

- **History**: A deque of length **150** (15 s at 10 chunks/s). Each entry stores timestamp, feature dict, and a copy of the audio chunk.
- **Voice activity**: For each chunk, voice is considered present if **F0 > 0** (and not NaN) and **RMS >= voice_energy_threshold** (default 0.005). Otherwise the chunk is treated as silence/gap.
- **Gaps**: Consecutive chunks with no voice form a gap. A gap is only considered if its duration is at least **silence_min_duration** (default 200 ms).
- **Breath fingerprint**: For each gap of at least **gap_min_duration** (500 ms), the detector checks:
  - Low-frequency energy (sum of audSpec_Rfilt_sma3[0–2]) above a threshold.
  - RMS in a band above the adaptive noise floor but below a max (breath is quiet).
  - Gap length and low-frequency dominance.
  If enough criteria pass, the gap is counted as a **breath event**.
- **BPM**: From the list of breath event times in the 15 s window, the detector computes intervals between consecutive events, filters to realistic intervals (e.g. 1–12 s), and sets **breath_rate = 60 / mean_interval**. If there are fewer than two events, BPM can be 0 or a rough estimate. Result is clamped to 0–60 BPM.

### 3.4 F0-to-emotion mapping

- **F0 in Hz**: From each chunk’s `F0semitoneFrom27.5Hz_sma3nz` using `Hz = 27.5 * 2^(semitones/12)`.
- **Silence / low F0**: If F0 is 0 or &lt; 50 Hz, the chunk is assigned **Neutral** (label and valence/arousal default).
- **Ranges**: The detector uses a fixed map **EMOTION_F0** (emotion name → (low_hz, high_hz)). Examples: Fear 300–500 Hz, Neutral 90–130 Hz, Sadness 90–170 Hz, Joy 160–180 Hz, Calm 85–195 Hz, Disgust 170–230 Hz, Anger 120–300 Hz, plus Love/Trust. Order and overlap determine which label wins when F0 falls in multiple ranges; scores are also computed by proximity when F0 is outside a range.
- **Valence/arousal**: Each emotion has a fixed **VAL_AROUSAL** entry (valence in [-1, 1], arousal in [0, 1]). The chunk’s top emotion is used to set valence and arousal in the output.

### 3.5 Output format (TouchDesigner-oriented)

For each chunk, the detector returns a dict suitable for TouchDesigner and for the bridge to aggregate:

- **breath_rate_normalized**: BPM normalized to [0, 1] by `(BPM - 8) / 32`, clamped (BPM 8–40 range).
- **anxiety_level**: 1.0 if BPM &gt; 30, else 0.0.
- **calm_level**: 1.0 if 0 &lt; BPM &lt; 20, else 0.0.
- **confidence**: 0.3–0.8, derived from evidence (breath events vs gaps) and BPM validity.
- **timestamp**: Unix time.
- **breath_rate_bpm**: Raw BPM (0–60).
- **emotion**: Dict with **f0_hz**, **top_emotion** (label, score), **emotion_scores** (all labels), **valence**, **arousal**.

The bridge uses **emotion.top_emotion.label** and **emotion.f0_hz** (and optionally **breath_rate_bpm**) to build per-segment Breath outputs and combines them with the emotion model output.

---

## 4. API in depth

### 4.1 Constructor: `BreathDetector(sample_rate=16000, use_opensmile=True)`

- **sample_rate**: Must match your audio; **16000** is required by the bridge and the rest of the project.
- **use_opensmile**: If `True`, initializes OpenSMILE (eGeMAPSv02, functionals) for feature extraction. If `False`, feature extraction is disabled and the detector will not work correctly; keep `True` for normal use.

Attributes you may care about for integration:

- **chunk_size**: Samples per chunk (e.g. 1600 at 16 kHz).
- **CHUNKS_PER_SECOND**: 10.
- **HISTORY_MAXLEN**: 150 (15 s).

### 4.2 Main entry: `process_chunk(audio_chunk)`

- **Input**: `audio_chunk` – 1D numpy array of length **chunk_size** (1600 at 16 kHz). Float32 preferred; if length differs, the detector pads or truncates.
- **Returns**: Single dict with keys as in Section 3.5 (breath_rate_normalized, anxiety_level, calm_level, confidence, timestamp, breath_rate_bpm, emotion).
- **Side effects**: Appends to the internal 15 s history buffer; breath rate and confidence are derived from that buffer.

Call this once per 100 ms of audio, in order, for real-time or batch. The bridge does exactly that: it slices the waveform into chunks of `detector.chunk_size` and calls `process_chunk` for each slice, then aggregates results over 2 s segments.

### 4.3 Legacy: `analyze_audio(audio_file_path)` and `analyze_signal(audio_signal, audio_file_path=None)`

- **analyze_audio**: Loads the file with librosa at `self.sample_rate`, then calls **analyze_signal**.
- **analyze_signal**: Splits the signal into chunks, calls **process_chunk** for each, and returns the **last** chunk result augmented with an overall `breath_rate` (average over chunks with positive normalized rate) and `audio_duration`. Useful for offline file-based testing; for integration with the bridge, the bridge does its own slicing and aggregation, so you typically do not need these.

### 4.4 Output keys (contract for integrators)

| Key | Type | Description |
|-----|------|-------------|
| `breath_rate_normalized` | float | 0.0–1.0, from (BPM - 8) / 32, clamped. |
| `anxiety_level` | float | 0.0 or 1.0 (1 if BPM > 30). |
| `calm_level` | float | 0.0 or 1.0 (1 if 0 < BPM < 20). |
| `confidence` | float | 0.3–0.8. |
| `timestamp` | float | Unix time. |
| `breath_rate_bpm` | float | Breath rate in BPM (0–60). |
| `emotion` | dict | See below. |

**emotion** dict:

| Key | Type | Description |
|-----|------|-------------|
| `f0_hz` | float | F0 in Hz (0 if silence). |
| `top_emotion` | dict | `{"label": str, "score": float}`. |
| `emotion_scores` | dict | Label -> score for all mapped emotions. |
| `valence` | float | -1 to 1. |
| `arousal` | float | 0 to 1. |

The bridge reads **emotion.top_emotion.label** and **emotion.f0_hz** (and **breath_rate_bpm**) to build Breath outputs per 2 s segment.

---

## 5. Constants (for reference)

- **CHUNK_DURATION** = 0.1 s  
- **CHUNKS_PER_SECOND** = 10  
- **HISTORY_DURATION** = 15.0 s, **HISTORY_MAXLEN** = 150  
- **BPM_MIN** = 8.0, **BPM_MAX** = 40.0, **BPM_RANGE** = 32.0  
- **CONFIDENCE_MIN** = 0.3, **CONFIDENCE_MAX** = 0.8  
- **EMOTION_F0**: F0 ranges in Hz per emotion (Fear, Neutral, Sadness, Joy, Calm, Disgust, Anger, Love, Trust).  
- **VAL_AROUSAL**: Valence and arousal per emotion.  
- **gap_min_duration** = 0.5 s; **silence_min_duration** = 0.2 s.  
- **voice_energy_threshold**, **low_freq_threshold**, **rms_min_ratio**, **rms_max**, **breath_energy_threshold** – tunables for voice detection and breath fingerprint.

---

## 6. Integration with the bridge

The **bridge** (in the project’s `bridge/` package) is the layer that combines Breath with the English model. It:

1. Loads or receives 16 kHz mono audio.
2. Instantiates **BreathDetector(sample_rate=16000, use_opensmile=True)**.
3. Slices the waveform into consecutive chunks of **detector.chunk_size**.
4. Calls **detector.process_chunk(chunk)** for each slice.
5. Aggregates results over **2 s segments**: for each segment it collects chunk outputs, computes dominant Breath label (from **emotion.top_emotion.label**), mean F0 (from **emotion.f0_hz**), mean BPM, and **breath_emotion_pcts** (only over chunks with **emotion.f0_hz >= 50** so silence does not dominate as Neutral).
6. Passes these segment-level Breath outputs together with the English model outputs into the Ensemble (final_result).

So as an **external developer** you normally **do not** call `BreathDetector` directly for the combined pipeline; you call the bridge’s **run_combined**. If you need only breath (no English model), you can instantiate **BreathDetector** and feed chunks with **process_chunk** yourself, and use the output dict as described above.

---

## 7. Dependencies and installation

- **Python**: 3.9+ recommended.
- **Package deps** (see `requirements.txt`): numpy, scipy, librosa, soundfile, pandas, **opensmile** (or install from the project’s **opensmile-python-main**). The OpenSMILE wrapper also uses **audeer**, **audinterface**, **audobject** (usually pulled in with `opensmile`).
- **Install**: From project root, install the main project requirements and OpenSMILE (e.g. `pip install -e ./opensmile-python-main` or `pip install opensmile`). Ensure **ascended_intelligence_sxsw2026/src** is on `sys.path` when importing `ascended.breath_detector`.

---

## 8. Example usage

From the project root (so that `ascended_intelligence_sxsw2026/src` is on the path):

```python
import sys
from pathlib import Path

PROJECT_ROOT = Path("/path/to/projectmk")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ascended_intelligence_sxsw2026" / "src"))

from ascended.breath_detector import BreathDetector
import numpy as np

detector = BreathDetector(sample_rate=16000, use_opensmile=True)
chunk_size = detector.chunk_size  # 1600 at 16 kHz

# Simulate 1 second of audio (10 chunks)
for _ in range(10):
    chunk = np.zeros(chunk_size, dtype=np.float32)  # or load real audio
    result = detector.process_chunk(chunk)
    print(result["breath_rate_bpm"], result["emotion"]["top_emotion"]["label"])
```

For full pipeline (Breath + English model + Ensemble), use the bridge’s **run_combined** with an audio path or waveform; see [bridge/README.md](../bridge/README.md).

---

## For developers: where to look in the code

So another developer can follow the logic and change behaviour without guessing.

### Files and responsibilities

- **`src/ascended/breath_detector.py`** – All Breath logic. **BreathDetector** is the only class; **process_chunk()** is the only method the bridge calls. **opensmile_wrapper** is used only inside breath_detector for feature extraction.
- **`src/ascended/opensmile_wrapper.py`** – Thin wrapper around the OpenSMILE library: exposes **Smile** with eGeMAPSv02 functionals. **BreathDetector** imports **Smile** and **FeatureSet**, **FeatureLevel** from opensmile; it does not expose any other API. You only need to read opensmile_wrapper if you change the feature set or the way features are extracted.

### Call graph (what runs when the bridge calls process_chunk)

1. **process_chunk(audio_chunk)** – Validates/pads chunk to **chunk_size** (1600 at 16 kHz). Calls **._extract_opensmile_features(audio_chunk)** (normalises chunk, **self.smile.process_signal()**, parses into **REQUIRED_FEATURES**). **._validate_opensmile_output(features)**. Appends to **self.history_buffer**. **._analyze_breath_patterns()** (**._identify_gaps()**, **._match_breath_fingerprint()** per gap, BPM from breath events, **._calculate_confidence()**). **._map_f0_to_emotion(features)** (F0 semitone to Hz, **EMOTION_F0** ranges, **VAL_AROUSAL**). **._format_touchdesigner_output(breath_rate, confidence, emotion_info)**. Returns that dict.

2. **Voice activity**: **._detect_voice_activity_f0(features)** – True if F0 != 0 and not NaN and **pcm_RMSenergy_sma >= voice_energy_threshold**. Used in **._identify_gaps()**.

3. **Gaps and breath fingerprint**: **._identify_gaps()** – Consecutive chunks with no voice form a gap if duration >= silence_min_chunks. **._match_breath_fingerprint(gap_start, gap_end)** – Low-freq bands + RMS over gap, thresholds (low_freq_threshold, rms_min_ratio, rms_max, low_freq_dominance). Returns True if enough criteria pass.

4. **F0 to emotion**: **._map_f0_to_emotion(features)** – **._semitone_to_hz()**, if F0 < 50 or 0 return Neutral; else score each **EMOTION_F0** range, top emotion and **VAL_AROUSAL** set valence/arousal.

### Constants and tuning (where to change behaviour)

- **CHUNK_DURATION**, **HISTORY_MAXLEN**, **BPM_MIN/MAX**, **CONFIDENCE_MIN/MAX** – At top of **BreathDetector** class.
- **EMOTION_F0** – F0 ranges (Hz) per emotion; order and overlap determine which label wins.
- **VAL_AROUSAL** – Valence/arousal per emotion.
- **voice_energy_threshold**, **low_freq_threshold**, **rms_min_ratio**, **rms_max**, **breath_energy_threshold**, **silence_min_duration**, **gap_min_duration** – In **__init__**; tune for sensitivity and false positives.

### What the bridge reads from each chunk result

- **emotion.top_emotion.label** – Used to build breath_emotion_pcts and breath_dominant per segment.
- **emotion.f0_hz** – Used to filter voice-only chunks (f0_hz >= 50) and mean F0 per segment.
- **breath_rate_bpm** – Mean BPM per segment (or bridge uses BPM fallback when 0).

The bridge does not use anxiety_level, calm_level, emotion_scores, or valence/arousal; those are for TouchDesigner or other consumers.
