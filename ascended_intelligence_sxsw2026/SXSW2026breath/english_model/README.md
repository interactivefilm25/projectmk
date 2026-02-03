# English Model (emotion2vec)

The emotion model is **emotion2vec/emotion2vec_plus_base** from Hugging Face. It uses FunASR and outputs probabilities over six labels: angry, disgust, fear, happy, neutral, sad.

## Location

- `english_model/model_loader.py` – load model, run inference
- `english_model/__init__.py` – exports `get_model`

## Usage

The **bridge** calls `english_model.get_model()` and then `model.infer(chunk)` for each segment or full audio.

**Input:** float32 mono waveform, 16 kHz. Minimum 1 s (16000 samples); shorter audio is padded.

**Output:** `(probs, pred_idx)` – probs over 6 labels, pred_idx is argmax.

## Installation

From project root, use `requirements.txt`:

```bash
pip install -r requirements.txt
conda install -y ffmpeg -c conda-forge
```

No Hugging Face token or `.env` required. The model is downloaded automatically on first use.

## API

- **`get_model()`** – Returns singleton `Audio2EmotionModel`.
- **`model.emotions`** – List of 6 label strings.
- **`model.infer(waveform)`** – Returns `(probs, pred_idx)`.
