let recordbutton
let mediaRecorder
let audioChunks = []
let socket

let chunk_duration = 500
let pingInterval

const init = async () => {
    console.log("recorder initialized")
    
    recordButton = document.getElementById("recordButton")
    recordButton.addEventListener("mousedown", startRecording)
    recordButton.addEventListener("mouseup", stopRecording)
    
    killButton = document.getElementById("killButton")
    killButton.addEventListener("click", killServer)
    
    const stream = await getMicrophoneAccess()
    mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" })

    openWebSocket()

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
        }
    }
}

const openWebSocket = () => {
    socket = new WebSocket("ws://127.0.0.1:5000/ws")
    
    socket.onopen = () => {
        console.log("WebSocket connection established")

        pingInterval = setInterval(() => {
            if (socket.readyState === WebSocket.OPEN) {
                socket.send("PING")
            }
        }, 3000) // Send a ping every 30 seconds
    }

    socket.onmessage = async (event) => {
        let text;
        if (event.data instanceof Blob) {
            console.log("Received Blob data from server");
            text = await event.data.text();
        } else {
            text = event.data;
        }
        console.log("Decoded message:", text);

        if (text.includes("{")) { // Parse JSON
            const obj = JSON.parse(text); 
            document.getElementById("detectedProbabilities").innerText = "Probabilities: " + JSON.stringify(obj, null, 2);
        } else { // Parse text
            document.getElementById("detectedEmotion").innerText = text;
        }
    }

    socket.onerror = (error) => {
        console.error("WebSocket error:", error)
    }

    socket.onclose = () => {
        console.log("WebSocket connection closed")
        clearInterval(pingInterval) // Clear the ping interval when the socket closes
        
        openWebSocket() // Reconnect
        console.log("Reconnecting WebSocket...")
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

    if (mediaRecorder && mediaRecorder.state === "inactive") {
        mediaRecorder.start();
        console.log("Recording started");
    }

    document.getElementById("detectedEmotion").innerText = "Recording..."
    audioChunks = []; // Clear previous audio chunks
    console.log("Audio chunks cleared");
}

const stopRecording = async () => {
    console.log("stop recording")
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
        console.log("Recording stopped");
    }
}

const killServer = async () => {
    console.log("Killing server...")
    socket.send("KILL")
}

document.addEventListener("DOMContentLoaded", init)