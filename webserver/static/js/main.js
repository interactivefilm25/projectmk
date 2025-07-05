let recordbutton
let mediaRecorder
let audioChunks = []

const init = async () => {
    console.log("recorder initialized")
    
    recordButton = document.getElementById("recordButton")
    recordButton.addEventListener("mousedown", startRecording)
    recordButton.addEventListener("mouseup", stopRecording)

    const stream = await getMicrophoneAccess()
    mediaRecorder = new MediaRecorder(stream)
    mediaRecorder.ondataavailable = event => {
        audioChunks.push(event.data);
    }

    mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' })
        audioChunks = [] // Clear the chunks for the next recording
        console.log("Audio Blob created:", audioBlob)

        const formData = new FormData()
        formData.append("audio", audioBlob, "recording.webm")

        try {
            const response = await fetch("/api/upload-audio", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                throw new Error("Network response was not ok");
            }

            const data = await response.json();
            console.log("Server response:", data);
        } catch (error) {
            console.error("Error sending audio to server:", error);
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