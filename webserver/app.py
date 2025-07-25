from quart import Quart, render_template, websocket, request
import os
import numpy as np
import soundfile as sf
from io import BytesIO
from pydub import AudioSegment
import joblib
import json
import opensmile
import pandas as pd

if not os.path.exists("uploads"):
    os.makedirs("uploads")

model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/voice_emotion_model.pkl"))
model = joblib.load(model_path)

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.emobase,
    feature_level=opensmile.FeatureLevel.Functionals,
)

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
    audio_file_path = "uploads/streamed_audio.wav"
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
            audio_segment.export("uploads/streamed_audio.wav", format="wav")
            print(f"Saved concatenated audio to uploads/streamed_audio.wav")

            await predict()

        elif data == "KILL":
            os._exit(0)

        else: # Process audio data
            print(f"Received data chunk of size {len(data)} bytes")
            audio_data.append(data)

async def predict():
    features = smile.process_file("uploads/streamed_audio.wav")
    trained_cols = model.feature_names_in_
    aligned_features = features.reindex(columns=trained_cols, fill_value=0.0)
    probs = model.predict_proba(aligned_features)[0]
    emo = model.classes_[np.argmax(probs)]
    probababilities = dict(zip(model.classes_, (probs * 100).round(2)))

    print(f"Model emotion prediction: {emo}")
    print(f"Probabilities: {probababilities}")

    await websocket.send(json.dumps(emo).encode())
    await websocket.send(json.dumps(probababilities).encode())