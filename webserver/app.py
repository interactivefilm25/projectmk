from quart import Quart, render_template, websocket, request
import os
import numpy as np
import soundfile as sf
from io import BytesIO
from pydub import AudioSegment
import joblib
import opensmile
import pandas as pd

if not os.path.exists("uploads"):
    os.makedirs("uploads")

model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../python/model.pkl"))
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

    try:
        with open(audio_file_path, "wb") as audio_file:
            while True:
                data = await websocket.receive()
                if data == "END":
                    print("Recording ended")
                    break

                audio_file.write(data)
                print(f"Received data chunk of size {len(data)} bytes")

                try:
                    audio_stream = BytesIO(data)
                    audio_segment = AudioSegment.from_file(audio_stream, format="webm")
                    wav_data = BytesIO()
                    audio_segment.export(wav_data, format="wav")
                    wav_data.seek(0)

                    audio_chunk, _ = sf.read(wav_data, dtype='float32')
                    audio_data.append(audio_chunk)
                    print(f"Processed Audio chunk shape: {audio_chunk.shape}")
                except Exception as e:
                    print(f"Error processing audio chunk: {e}")

        if audio_data:
            full_audio_array = np.concatenate(audio_data, axis=0)
            print(f"Full audio array shape: {full_audio_array.shape}")

            if full_audio_array.ndim > 1:
                signal = full_audio_array[:, 0]  # Use first channel if stereo
            else:
                signal = full_audio_array

            features: pd.DataFrame = smile.process_signal(signal, 16000)
            print(f"Extracted features shape: {features.shape}")

            trained_cols = model.feature_names_in_
            aligned_features = features.reindex(columns=trained_cols, fill_value=0.0)

            probs = model.predict_proba(aligned_features)[0]
            emo = model.classes_[probs.argmax()]

            print(f"Model emotion prediction: {emo}")
            print(f"Probabilities: {dict(zip(model.classes_, (probs * 100).round(2)))}")

            await websocket.send(str(emo))
        else:
            print(f"No valid audio data received")

        print(f"Audio saved to {audio_file_path}")
    except ConnectionResetError:
        print("WebSocket connection closed by client")
    except Exception as e:
        print(f"An error occurred: {e}")
