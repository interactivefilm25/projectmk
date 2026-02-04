# English Model

The emotion model is **emotion2vec/emotion2vec_plus_base** from Hugging Face. A separate noise model (DeepFilterNet2 or noisereduce) cleans and enhances audio. Models are stored in `english_model/model/{model_name}/`.

## Module layout

| Module | Purpose |
|--------|---------|
| **english_model.py** | Emotion recognition (emotion2vec_plus_base) |
| **noise_model.py** | Noise cleaning (DeepFilterNet2) + `enhance_audio()` |
| **download_model.py** | Download models to `model/` |
| **load_model.py** | Load from `model/` |

## Model paths

- `model/emotion2vec_plus_base/` – Emotion model
- `model/DeepFilterNet2/` – Noise model (optional)

## Usage

The **bridge** calls `english_model.get_model()` and `model.infer(chunk)` for emotion. It calls `get_noise_cleaner()` and `prepare_audio()` for cleaning.

### Emotion model

**Input:** float32 mono waveform, 16 kHz. Minimum 1 s (16000 samples); shorter audio is padded.

**Output:** `(probs, pred_idx)` – probs over 6 labels, pred_idx is argmax.

### Noise model

- **NoiseCleaner.clean(waveform, sample_rate, target_sr=48000)** – Denoise (DeepFilterNet2 or noisereduce fallback)
- **enhance_audio(waveform, sample_rate)** – Rumble filter + light noise reduction + peak normalization

### API

- **`get_model()`** – Returns singleton `Audio2EmotionModel`.
- **`model.emotions`** – List of 6 label strings.
- **`model.infer(waveform)`** – Returns `(probs, pred_idx)`.
- **`get_noise_cleaner()`** – Returns singleton `NoiseCleaner`.
- **`enhance_audio(waveform, sample_rate)`** – Gentle enhancement.
- **`download_emotion_model()`** – Download emotion model to `model/emotion2vec_plus_base/`.
- **`download_noise_model()`** – Download DeepFilterNet2 to `model/DeepFilterNet2/`.

## Installation

From project root, use `requirements.txt`:

```bash
pip install -r requirements.txt
conda install -y ffmpeg -c conda-forge
```

No Hugging Face token or `.env` required. Models are downloaded automatically on first use (or via `download_emotion_model()` / `download_noise_model()`).
