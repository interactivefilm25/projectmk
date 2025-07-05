from quart import Quart, render_template, websocket, request
import os

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

# @app.route("/api")
# async def json():
#     return {"hello": "world"}

@app.websocket("/ws")
async def audio_stream():
    audio_file_path = "uploads/streamed_audio.webm"
    try:
        with open(audio_file_path, "wb") as audio_file:
            while True:
                data = await websocket.receive()
                if data == "END":
                    print("Recording ended")
                    break
                audio_file.write(data)
                print(f"Received data chunk of size {len(data)} bytes")
        print(f"Audio saved to {audio_file_path}")
    except ConnectionResetError:
        print("WebSocket connection closed by client")
    except Exception as e:
        print(f"An error occurred: {e}")

#     while True:
#         await websocket.send("hello")
#         await websocket.send_json({"hello": "world"})