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

        elif data == "KILL":
            os._exit(0)

        elif data == "TEST":
            await testAction()

        else: # Process audio data
            print(f"Received data chunk of size {len(data)} bytes")
            audio_data.append(data)

async def predict():
    if os.path.exists(audio_file_path):
        features = smile.process_file(audio_file_path)
        X = features.values.reshape(1, -1)
        probs = model.predict_proba(X)[0]
        classes = model.classes_
        top = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)

        result = {
            "top": top[0][0].upper(),
            "predictions": [{"label": label, "probability": float(prob)} for label, prob in top]
        }
        print(f"[predict] Model emotion prediction: {result}")
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