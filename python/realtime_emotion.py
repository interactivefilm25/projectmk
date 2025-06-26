import uuid
import os
from pathlib import Path
import sounddevice as sd
import soundfile as sf
import opensmile
import joblib
import numpy as np

# —— CONFIG ——————————————————————————————————————————————————————————————————

# Where to save recordings
OUTPUT_DIR = Path(r"C:\Users\Me\Desktop\voice_gpt_chat\My recording")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# OpenSMILE setup (adjust path if needed)
CONF_FILE = r"C:\Users\Me\Desktop\Dissertassion\opensmile-3.0.2-windows-x86_64\config\emobase\emobase.conf"
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.emobase,
    feature_level=opensmile.FeatureLevel.Functionals
)

# Load trained classifier
MODEL_PATH = Path(__file__).parent / "model.pkl"
clf = joblib.load(MODEL_PATH)

# Audio settings
SR = 16_000   # sample rate
DUR = 5.0     # seconds per recording

# —— RECORD FUNCTION —————————————————————————————————————————————————————————————

def record_and_save():
    print("🎙️  START SPEAKING NOW…")
    audio = sd.rec(int(DUR * SR), samplerate=SR, channels=1, dtype='float32')
    sd.wait()
    # write out to your folder
    filename = f"{uuid.uuid4()}.wav"
    out_path = OUTPUT_DIR / filename
    sf.write(str(out_path), audio, SR)
    print(f"✅  Recording saved to {out_path}")
    return out_path

# —— FEATURE EXTRACTION & PREDICTION ——————————————————————————————————————————————

def predict(wav_path):
    # extract functionals
    features = smile.process_file(str(wav_path))
    X = features.to_numpy().reshape(1, -1)
    # align columns if needed (drop or pad)
    trained_cols = clf.feature_names_in_
    X_df = np.zeros((1, len(trained_cols)))
    for i, feat in enumerate(features.columns):
        if feat in trained_cols:
            idx = list(trained_cols).index(feat)
            X_df[0, idx] = X[0, i]
    probs = clf.predict_proba(X_df)[0]
    emo = clf.classes_[np.argmax(probs)]
    return emo, dict(zip(clf.classes_, (probs * 100).round(2)))

# —— MAIN ——————————————————————————————————————————————————————————————————————

def main():
    print("🚀  realtime_emotion.py starting…")
    print("✅  OpenSMILE initialized.")
    print(f"🔍  Loading model from {MODEL_PATH}")
    print("✅  Model loaded.")
    wav = record_and_save()
    emo, probs = predict(wav)
    print("\n🧠  Predicted Emotion:", emo)
    print("📊  Probabilities:")
    for e, p in probs.items():
        print(f"    {e:9s}: {p:6.2f}%")

if __name__ == "__main__":
    main()
