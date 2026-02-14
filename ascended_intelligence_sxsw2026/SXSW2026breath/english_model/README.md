# English Model

The emotion model is **Wav2Vec2** (same as TouchDesigner / `realtime_emotion_td.py`): **multilingual_emotion_model** with 5 labels. A separate noise model (DeepFilterNet2) cleans audio. Models are stored in `english_model/model/{model_name}/`.

## Module layout

| Module | Purpose |
|--------|---------|
| **english_model.py** | Emotion recognition (Wav2Vec2ForSequenceClassification, 5 labels) |
| **noise_model.py** | Noise cleaning (DeepFilterNet2) + `enhance_audio()` |
| **download_model.py** | Ensure model dirs; download noise model to `model/` |
| **load_model.py** | Load from `model/` |

## Model paths

- `model/multilingual_emotion_model/` – Emotion model (Wav2Vec2; place your model here or set `HF_EMOTION_ID` in `download_model.py`)
- `model/DeepFilterNet2/` – Noise model (optional)

## Usage

The **bridge** calls `english_model.get_model()` and `model.infer(chunk)` for emotion. It calls `get_noise_cleaner()` and `prepare_audio()` for cleaning.

### Emotion model

**Input:** float32 mono waveform, 16 kHz. Minimum 1 s (16000 samples); shorter audio is padded.

**Output:** `(probs, pred_idx)` – probs over 5 labels, pred_idx is argmax.

**Labels (TouchDesigner / OSC 0–4):** `anger_fear`, `joy_excited`, `sadness`, `curious_reflective`, `calm_content`.

### Noise model

- **NoiseCleaner.clean(waveform, sample_rate, target_sr=48000)** – Denoise (DeepFilterNet2)
- **enhance_audio(waveform, sample_rate)** – No-op (API compatibility).

### API

- **`get_model()`** – Returns singleton `Audio2EmotionModel`.
- **`model.emotions`** – List of 5 label strings.
- **`model.infer(waveform)`** – Returns `(probs, pred_idx)`.
- **`get_noise_cleaner()`** – Returns singleton `NoiseCleaner`.
- **`download_emotion_model()`** – Creates `model/multilingual_emotion_model/`; if `HF_EMOTION_ID` is set, downloads from HuggingFace.
- **`download_noise_model()`** – Download DeepFilterNet2 to `model/DeepFilterNet2/`.

## Installation

From project root:

```bash
pip install -r requirements.txt
conda install -y ffmpeg -c conda-forge
```

Place your Wav2Vec2 emotion model (same as used by `realtime_emotion_td.py`) in `english_model/model/multilingual_emotion_model/` (e.g. `config.json`, `pytorch_model.bin` or `model.safetensors`). If the path is missing, the code falls back to loading `multilingual_emotion_model` from the current working directory or HuggingFace.
