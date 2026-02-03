# Bridge – Integration Guide

The bridge runs the combined pipeline (Breath + emotion model) on audio and returns one result per time segment.

---

## 1. Overview

1. **Breath pipeline** (OpenSMILE): 100 ms chunks, F0, breath rate (BPM), BPM-based state (Calm, Slightly elevated, Anxious, High anxiety).
2. **Emotion model** (emotion2vec): Full-audio inference, probabilities over six labels (angry, disgust, fear, happy, neutral, sad).

Outputs are kept separate per segment: breath state, Audio2Emotion, and target frequency.

---

## 2. API

### `load_audio(audio_path, max_sec=None)`

Loads an audio file and returns float32 mono at 16 kHz.

### `run_combined(audio_path=None, waveform=None, sample_rate=None, segment_duration=None, full_audio_emotion=True, source="file")`

Runs the full pipeline.

- **audio_path** or **(waveform, sample_rate)** – Required. Pass a file path or raw waveform.
- **segment_duration** – Seconds per output segment. `None` = one result for full audio.
- **full_audio_emotion** – If True, run emotion model on entire audio once (default).
- **source** – `"file"` (default) or `"live"`. For waveform input: `"live"` uses segment-based emotion when audio ≥ 5 s (3 s windows, picks most confident non-neutral).

**Returns:** `list[dict]` – one dict per segment.

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

## 4. Example

```python
import sys
from pathlib import Path

PROJECT_ROOT = Path("/path/to/projectmk")
sys.path.insert(0, str(PROJECT_ROOT))

from bridge import run_combined, load_audio

# From file
segments = run_combined(audio_path=PROJECT_ROOT / "test.ogg")
for seg in segments:
    print(seg["primary_emotion"], seg["audio2emotion_emotion"], seg["target_frequency_hz"])

# From waveform
waveform, sr = load_audio("/path/to/audio.wav", max_sec=30.0)
segments = run_combined(waveform=waveform, sample_rate=sr, segment_duration=None)
```

---

## 5. Dependencies

- **ascended.breath_detector** – Breath pipeline.
- **english_model.get_model()** – emotion2vec (FunASR).
- **OpenSMILE** – from `opensmile-python-main` or `pip install opensmile`.
