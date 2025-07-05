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

# @app.websocket("/ws")
# async def ws():
#     while True:
#         await websocket.send("hello")
#         await websocket.send_json({"hello": "world"})