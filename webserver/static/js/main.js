let recordbutton
let mediaRecorder
let audioChunks = []
let socket

const init = async () => {
    console.log("recorder initialized")
    
    recordButton = document.getElementById("recordButton")
    recordButton.addEventListener("mousedown", startRecording)
    recordButton.addEventListener("mouseup", stopRecording)

    const stream = await getMicrophoneAccess()
    mediaRecorder = new MediaRecorder(stream)

    socket = new WebSocket("ws://127.0.0.1:5000/ws")
    
    socket.onopen = () => {
        console.log("WebSocket connection established")
    }

    socket.onerror = (error) => {
        console.error("WebSocket error:", error)
    }

    socket.onclose = () => {
        console.log("WebSocket connection closed")
    }

    mediaRecorder.ondataavailable = event => {
        const audioChunk = event.data

        if (socket.readyState === WebSocket.OPEN) {
            socket.send(audioChunk)
            console.log("Audio chunk sent: ", audioChunk)
        } else {
            console.error("WebSocket is not open. Cannot send audio chunk.")
        }
    }

    mediaRecorder.onstop = async () => {
        console.log("Recording stopped")

        if (socket.readyState === WebSocket.OPEN) {
            socket.send("END")
            console.log("Sent END message to WebSocket")
            socket.close()
            console.log("WebSocket connection closed")
        }
    }
}

const getMicrophoneAccess = async () => {
    try {
        const mic = await navigator.mediaDevices.getUserMedia({ audio: true });
        console.log("Microphone access granted:", mic);
        return mic;
    } catch (error) {
        console.error("Error accessing microphone:", error);
    }
}

const startRecording = async () => {
    console.log("start recording")
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        console.error("getUserMedia not supported on your browser!");
        return;
    }

    if (mediaRecorder && mediaRecorder.state === "recording") {
        console.warn("Already recording.");
        return;
    }

    if (mediaRecorder && mediaRecorder.state === "inactive") {
        mediaRecorder.start();
        console.log("Recording started");
    }
}

const stopRecording = async () => {
    console.log("stop recording")
    if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
        console.log("Recording stopped");
    } else {
        console.warn("MediaRecorder is not recording.");
    }

    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
        console.log("Recording stopped");
    }
}

document.addEventListener("DOMContentLoaded", init)