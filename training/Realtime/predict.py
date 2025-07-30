
import os
import joblib
import sounddevice as sd
import numpy as np
import pandas as pd
import subprocess
import time
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import soundfile as sf

# Constants
DURATION = 4  # seconds
SAMPLERATE = 16000
BASE_PATH = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_PATH / "model" / "emotion_model.pkl"
ENCODER_PATH = BASE_PATH / "model" / "encoder.pkl"
SMILE_BIN = Path("C:/opensmile-3.0-win-x64/bin/SMILExtract.exe")
SMILE_CONF = Path("C:/opensmile-3.0-win-x64/config/emobase/emobase.conf")
OUTPUT_CSV = BASE_PATH / "features" / "output.csv"
AUDIO_FILE = BASE_PATH / "audio" / "recording.wav"

# Create necessary directories
os.makedirs(BASE_PATH / "features", exist_ok=True)
os.makedirs(BASE_PATH / "audio", exist_ok=True)

print("🎙️ Recording voice for emotion prediction...")
recording = sd.rec(int(DURATION * SAMPLERATE), samplerate=SAMPLERATE, channels=1, dtype='int16')
sd.wait()
sf.write(AUDIO_FILE, recording, SAMPLERATE)
print("✅ Recording complete.")

print("🔍 Extracting features with OpenSMILE...")
command = [
    str(SMILE_BIN),
    "-C", str(SMILE_CONF),
    "-I", str(AUDIO_FILE),
    "-O", str(OUTPUT_CSV)
]
subprocess.run(command, check=True)

print("📊 Parsing features manually...")
df = pd.read_csv(OUTPUT_CSV, sep=';', comment='@')
df = df.dropna(axis=1, how='any')

print("🧠 Loading model and encoder...")
model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

expected_features = model.feature_names_in_
missing = set(expected_features) - set(df.columns)
if missing:
    print(f"⚠️ Warning: Missing features filled with 0: {missing}")
    for col in missing:
        df[col] = 0
df = df[expected_features]

prediction = model.predict(df)[0]

# Static class map instead of encoder.inverse_transform
if hasattr(encoder, 'classes_'):
    emotion_label = encoder.classes_[int(prediction)]
else:
    emotion_label = str(prediction)

print(f"🗣️ Predicted Emotion: {emotion_label}")
