from quart import Quart, render_template, websocket, request
import os
import numpy as np
import soundfile as sf
from io import BytesIO
from pydub import AudioSegment

if not os.path.exists("uploads"):
    os.makedirs("uploads")

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
        else:
            print(f"No valid audio data received")

        print(f"Audio saved to {audio_file_path}")
    except ConnectionResetError:
        print("WebSocket connection closed by client")
    except Exception as e:
        print(f"An error occurred: {e}")
