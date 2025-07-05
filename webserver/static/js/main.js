let recordbutton
let recorder
let mediaRecorder
let audioChunks = []
let socket

let chunk_duration = 500

const init = async () => {
    console.log("recorder initialized")
    
    recordButton = document.getElementById("recordButton")
    recordButton.addEventListener("mousedown", startRecording)
    recordButton.addEventListener("mouseup", stopRecording)

    const stream = await getMicrophoneAccess()
    const audioContext = new(window.AudioContext || window.webkitAudioContext)()
    const input = audioContext.createMediaStreamSource(stream)

    recorder = new Recorder(input, { numChannels: 1 })

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

    // mediaRecorder.ondataavailable = event => {
    //     const audioChunk = event.data

    //     if (socket.readyState === WebSocket.OPEN) {
    //         socket.send(audioChunk)
    //         console.log("Audio chunk sent: ", audioChunk)
    //     } else {
    //         console.error("WebSocket is not open. Cannot send audio chunk.")
    //     }
    // }

    // mediaRecorder.onstop = async () => {
    //     console.log("Recording stopped")

    //     if (socket.readyState === WebSocket.OPEN) {
    //         socket.send("END")
    //         console.log("Sent END message to WebSocket")
    //         socket.close()
    //         console.log("WebSocket connection closed")
    //     }
    // }
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
    if (recorder) {
        recorder.record()
        console.log("Recording started")

        setInterval(() => {
            if (recorder.recording) {
                recorder.exportWAV(blob => {
                    if (socket.readyState === WebSocket.OPEN) {
                        socket.send(blob)
                        console.log("Audio blob sent to WebSocket")
                    } else {
                        console.error("WebSocket is not open. Cannot send audio blob.")
                    }
                })

                recorder.clear() // Clear the recorder buffer
            }
        }, chunk_duration); // Send audio data every n ms
    }
    // if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    //     console.error("getUserMedia not supported on your browser!");
    //     return;
    // }

    // if (mediaRecorder && mediaRecorder.state === "recording") {
    //     console.warn("Already recording.");
    //     return;
    // }

    // if (mediaRecorder && mediaRecorder.state === "inactive") {
    //     mediaRecorder.start();
    //     console.log("Recording started");
    // }
}

const stopRecording = async () => {
    console.log("stop recording")
    if (recorder) {
        recorder.stop();
        console.log("Recording stopped");
        
        recorder.exportWAV(blob => {
            if (socket.readyState === WebSocket.OPEN) {
                socket.send(blob);
                console.log("Send final WAV chunk to WebSocket");
            } else {
                console.error("WebSocket is not open. Cannot send audio blob.");
            }

            recorder.clear(); // Clear the recorder for the next recording

            if (socket.readyState === WebSocket.OPEN) {
                socket.send("END");
                console.log("Sent END message to WebSocket");
                socket.close();
                console.log("WebSocket connection closed");
            }
            
            // const reader = new FileReader();
            // reader.onload = () => {
            //     const audioData = reader.result;
            //     if (socket.readyState === WebSocket.OPEN) {
            //         socket.send(audioData);
            //         console.log("Audio data sent to WebSocket");
            //     } else {
            //         console.error("WebSocket is not open. Cannot send audio data.");
            //     }
            // };
            // reader.readAsArrayBuffer(blob);
        });
    }
    // if (mediaRecorder && mediaRecorder.state === "recording") {
    //     mediaRecorder.stop();
    //     console.log("Recording stopped");
    // } else {
    //     console.warn("MediaRecorder is not recording.");
    // }

    // if (mediaRecorder && mediaRecorder.state === 'recording') {
    //     mediaRecorder.stop();
    //     console.log("Recording stopped");
    // }
}

document.addEventListener("DOMContentLoaded", init)