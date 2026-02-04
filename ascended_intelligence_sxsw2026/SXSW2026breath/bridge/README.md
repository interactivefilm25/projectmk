# Bridge – Integration Guide

The bridge runs the combined pipeline (Breath + emotion model) on audio, optionally prepares it (denoise + enhance), sends OSC to TouchDesigner, and returns one result per time segment.

---

## 1. Overview

1. **prepare_audio** – Denoise (DeepFilterNet2 or noisereduce) + enhance (rumble filter, light NR, peak norm).
2. **Breath pipeline** (OpenSMILE): 100 ms chunks, F0, breath rate (BPM), BPM-based state (Calm, Slightly elevated, Anxious, High anxiety).
3. **Emotion model** (emotion2vec): Full-audio inference, probabilities over six labels (angry, disgust, fear, happy, neutral, sad).
4. **OSC** – Sends /emotion, /frequency, /breath, /bpm to TouchDesigner (default 127.0.0.1:5005).

Outputs are kept separate per segment: breath state, Audio2Emotion, and target frequency.

---

## 2. API

### `prepare_audio(waveform, sample_rate)`

Cleans and enhances audio for the emotion model. Returns processed float32 waveform.

- **waveform** – float32 mono array
- **sample_rate** – Hz (typically 16000)

### `load_audio(audio_path, max_sec=None)`

Loads an audio file and returns float32 mono at 16 kHz.

### `run_combined(audio_path=None, waveform=None, sample_rate=None, segment_duration=None, full_audio_emotion=True, source="file", clean_audio=True)`

Runs the full pipeline.

- **audio_path** or **(waveform, sample_rate)** – Required. Pass a file path or raw waveform.
- **segment_duration** – Seconds per output segment. `None` = one result for full audio.
- **full_audio_emotion** – If True, run emotion model on entire audio once (default).
- **source** – `"file"` (default) or `"live"`.
- **clean_audio** – If True, run prepare_audio (denoise + enhance) before analysis (default).

**Returns:** `list[dict]` – one dict per segment.

### OSC

- **configure_osc(ip=None, port=None, enabled=None)** – Configure OSC before run_combined.
- **osc_client** – Send custom messages with `osc_client.send_raw(address, value)`.
- **get_osc_info()** – Current OSC config.

---

## 3. Segment output

| Key | Type | Description |
|-----|------|-------------|
| `start_time` | float | Segment start (seconds). |
| `end_time` | float | Segment end (seconds). |
| `breath_emotion` | str | BPM-based state: calm, slightly_elevated, anxious, high_anxiety. |
| `primary_emotion` | str | Human-readable breath state (e.g. "Calm and relaxed"). |
| `breath_f0` | float | Mean F0 (Hz). |
| `breath_bpm` | float | Mean breath rate (BPM). |
| `audio2emotion_emotion` | str | Dominant emotion: angry, disgust, fear, happy, neutral, sad. |
| `audio2emotion_probs` | dict | Probability per emotion. |
| `target_frequency_hz` | int | Target Hz for TouchDesigner (639 or 396). |

---

## 4. OSC messages (sent after each segment)

| Address | Type | Description |
|---------|------|-------------|
| /emotion | int | 0–6 (angry, disgust, fear, happy, neutral, sad, surprised), -1=unknown |
| /frequency | float | F0 normalized 0.0–1.0 (50–400 Hz) |
| /breath | int | 0–3 (calm, slightly_elevated, anxious, high_anxiety) |
| /bpm | float | BPM normalized 0.0–1.0 (10–40 BPM) |

---

## 5. Example

```python
import sys
from pathlib import Path

PROJECT_ROOT = Path("/path/to/projectmk")
sys.path.insert(0, str(PROJECT_ROOT))

from bridge import run_combined, load_audio, prepare_audio, configure_osc

# Configure OSC for TouchDesigner
configure_osc(ip="127.0.0.1", port=5005)

# From file
segments = run_combined(audio_path=PROJECT_ROOT / "test.ogg")
for seg in segments:
    print(seg["primary_emotion"], seg["audio2emotion_emotion"], seg["target_frequency_hz"])

# From waveform (e.g. live mic)
waveform, sr = load_audio("/path/to/audio.wav", max_sec=30.0)
waveform = prepare_audio(waveform, sr)  # optional if clean_audio=True
segments = run_combined(waveform=waveform, sample_rate=sr, segment_duration=None)
```

---

## 6. Dependencies

- **ascended.breath_detector** – Breath pipeline.
- **english_model.get_model()** – emotion2vec (FunASR).
- **english_model.get_noise_cleaner()** – Noise cleaning.
- **bridge.osc** – OSC client.
- **OpenSMILE** – from `opensmile-python-main` or `pip install opensmile`.
