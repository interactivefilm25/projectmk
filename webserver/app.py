from quart import Quart, render_template, websocket, request
from pydub import AudioSegment
import os
import numpy as np
import soundfile as sf
from io import BytesIO
import joblib
import json
import opensmile
import pandas as pd

if not os.path.exists("uploads"):
    os.makedirs("uploads")

model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/rf_compare_model.pkl"))
model = joblib.load(model_path)

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.emobase,
    feature_level=opensmile.FeatureLevel.Functionals,
)

audio_file_path = "uploads/streamed_audio.wav"
# audio_file_path = "../data/audioSamples/neutral.wav" # test on trained files

app = Quart(__name__)

@app.route("/")
async def webserver():
    return await render_template("index.html")

@app.route("/api/upload-audio", methods=["POST"])
async def upload_audio():
    files = await request.files
    audio = files.get("audio")
    if audio:
        await audio.save(f"uploads/{audio.filename}")
        return {"status": "success", "message": "Audio uploaded successfully"}
    else:
        return {"status": "error", "message": "No audio file provided"}, 400

@app.websocket("/ws")
async def audio_stream():
    audio_data = []
    recording_started = False

    while True:
        data = await websocket.receive()
        # print(f"Received initial data: {data}")

        if data == "PING":
            continue

        elif data == "END":
            print("Recording ended")

            all_bytes = b''.join(audio_data)
            audio_stream = BytesIO(all_bytes)
            audio_segment = AudioSegment.from_file(audio_stream, format="webm")
            audio_segment = audio_segment.set_frame_rate(16000)
            audio_segment.export(audio_file_path, format="wav", parameters=["-acodec", "pcm_s16le"])
            print(f"Saved concatenated audio to {audio_file_path} at 16kHz")

            await predict()
            # Reset for next recording
            audio_data = []
            recording_started = False

        elif data == "KILL":
            os._exit(0)

        elif data == "TEST":
            await testAction()

        else: # Process audio data
            if not recording_started:
                # On first chunk, delete the old wav file if it exists
                if os.path.exists(audio_file_path):
                    try:
                        os.remove(audio_file_path)
                        print(f"Deleted old {audio_file_path}")
                    except Exception as e:
                        print(f"Could not delete {audio_file_path}: {e}")
                recording_started = True
            print(f"Received data chunk of size {len(data)} bytes")
            audio_data.append(data)

def map_vibration(f0_hz: float) -> str:
    if f0_hz <= 120:
        return "calm/reflective (75-120 Hz)"
    if f0_hz <= 180:
        return "stable/ordinary (120-180 Hz)"
    if f0_hz <= 250:
        return "energised/attentive (180-250 Hz)"
    if f0_hz <= 350:
        return "excited/anxious (250-350 Hz)"
    if f0_hz <= 500:
        return "overstimulated/fragile (350-500 Hz)"
    return "alarm/chaos (>500 Hz)"

async def predict():
    import time
    start_time = time.time()

    if os.path.exists(audio_file_path):
        features = smile.process_file(audio_file_path)
        expected_features = model.feature_names_in_
        X_df = pd.DataFrame([features.values.flatten()], columns=features.columns)
        X_df = X_df.reindex(columns=expected_features, fill_value=0)
        X = X_df.values

        probs = model.predict_proba(X)[0]
        labels = model.classes_
        ranked = sorted(zip(labels, probs), key=lambda x: x[1], reverse=True)

        emotion_probabilities = [
            {"label": lbl, "certainty": round(float(p), 2)}
            for lbl, p in sorted(ranked, key=lambda x: x[1], reverse=True)
        ]

        frequency = 0
        mean_vibration = 0
        if "F0_sma_amean" in features.columns:
            f0 = float(features["F0_sma_amean"].iloc[0])
            frequency = map_vibration(f0)
            mean_vibration = round(f0, 2)

        result = {
            "emotions_sorted": emotion_probabilities,
            "vibration": {
                "mean": mean_vibration,
                "frequency": frequency
            }
        }
        print(result)
    
        end_time = time.time()
        print(f"[predict] Processing time: {end_time - start_time:.3f} seconds")
    
        await websocket.send(json.dumps(result).encode())
    else:
        print(f"[predict] File {audio_file_path} does not exist.")

async def testAction():
    print("Test action triggered")
    if os.path.exists(audio_file_path):
        # Try soundfile
        data, samplerate = sf.read(audio_file_path)
        print(f"[soundfile] dtype: {data.dtype}, shape: {data.shape}, sample rate: {samplerate}")
        print(f"[soundfile] min: {np.min(data)}, max: {np.max(data)}, mean: {np.mean(data)}, std: {np.std(data)}")

        # Run the model prediction pipeline
        await predict()
    else:
        print(f"File {audio_file_path} does not exist.")

if __name__ == "__main__":
    app.run(debug=True)