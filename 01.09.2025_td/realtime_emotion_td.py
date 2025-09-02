import queue
import tempfile
import numpy as np
import sounddevice as sd
import torchaudio
import torch
import opensmile
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from pythonosc import udp_client
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", message="Segment too short, filling with NaN")
warnings.filterwarnings(
    "ignore",
    message="In 2.9, this function's implementation will be changed to use torchaudio.load_with_torchcodec"
)


# ----------------------------
# Frequency Analyzer
# ----------------------------
class FrequencyAnalyzer:
    def __init__(self):
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.emobase,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

    def extract_frequency(self, audio_chunk: np.ndarray, sr: int) -> float:
        try:
            signal = audio_chunk.reshape(1, -1) if audio_chunk.ndim == 1 else audio_chunk
            features = self.smile.process_signal(signal=signal, sampling_rate=sr)
            for col in features.columns:
                if "F0" in col and "amean" in col:
                    val = float(features[col].iloc[0])
                    if np.isfinite(val) and val > 0:
                        return val
        except Exception:
            pass
        return float("nan")


# ----------------------------
# Emotion Analyzer (Wav2Vec2)
# ----------------------------
class Wav2Vec2EmotionAnalyzer:
    def __init__(self, model_path="multilingual_emotion_model"):
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.id2label = {
            0: "anger_fear",
            1: "joy_excited",
            2: "sadness",
            3: "curious_reflective",
            4: "calm_content",
        }

    def analyze_emotion(self, audio_path: str) -> str:
        try:
            speech_array, sr = torchaudio.load(audio_path)
            if speech_array.shape[0] > 1:
                speech_array = torch.mean(speech_array, dim=0, keepdim=True)
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                speech_array = resampler(speech_array)
            speech = speech_array.squeeze().numpy()
            inputs = self.feature_extractor(
                speech, sampling_rate=16000, return_tensors="pt", padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self.model(**inputs).logits
            predicted_id = torch.argmax(logits, dim=-1).item()
            return self.id2label[predicted_id]
        except Exception as e:
            print(f"Emotion analysis error: {e}")
            return "Unknown"


# ----------------------------
# Live Audio Analyzer
# ----------------------------
class LiveAudioAnalyzer:
    def __init__(
        self,
        wav2vec_model_path="multilingual_emotion_model",
        chunk_size_sec=1.0,
        emotion_window_sec=5.0,
        sr=16000,
        osc_ip="127.0.0.1",
        osc_port=5005,
        rms_threshold=0.01,
    ):
        self.frequency_analyzer = FrequencyAnalyzer()
        self.emotion_analyzer = Wav2Vec2EmotionAnalyzer(wav2vec_model_path)
        self.chunk_size_sec = chunk_size_sec
        self.emotion_window_sec = emotion_window_sec
        self.sr = sr
        self.buffer = []
        self.q = queue.Queue()
        self.osc_client = udp_client.SimpleUDPClient(osc_ip, osc_port)
        self.rms_threshold = rms_threshold

        # Emotion → integer mapping for TouchDesigner
        self.emotion_map = {
            "anger_fear": 0,
            "joy_excited": 1,
            "sadness": 2,
            "curious_reflective": 3,
            "calm_content": 4,
            "Unknown": -1
        }

    def _process_chunk(self, chunk: np.ndarray):
        # --- Frequency Analysis ---
        freq = self.frequency_analyzer.extract_frequency(chunk, self.sr)
        if np.isfinite(freq):
            # Normalize frequency (50Hz–400Hz typical vocal range)
            freq_norm = min(max((freq - 50) / (400 - 50), 0.0), 1.0)
            print(f"Frequency: {freq:.2f} Hz (normalized: {freq_norm:.2f})")
            self.osc_client.send_message("/frequency", freq_norm)

        # --- Buffer audio for emotion analysis ---
        self.buffer.extend(chunk.flatten().tolist())
        if len(self.buffer) >= int(self.emotion_window_sec * self.sr):
            buffer_np = np.array(self.buffer, dtype=np.float32)
            rms = np.sqrt(np.mean(buffer_np ** 2))
            if rms < self.rms_threshold:
                print("Silence detected, skipping emotion analysis")
                self.buffer = []
                return

            # Save to temp WAV for Wav2Vec2
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                temp_path = tmpfile.name
                tensor_buffer = torch.tensor(buffer_np).unsqueeze(0)
                torchaudio.save(temp_path, tensor_buffer, self.sr)

            # Emotion analysis
            emotion_str = self.emotion_analyzer.analyze_emotion(temp_path)
            emotion_int = self.emotion_map.get(emotion_str, -1)
            print(f"Emotion: {emotion_str} → {emotion_int}")
            self.osc_client.send_message("/emotion", emotion_int)

            self.buffer = []

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.q.put(indata.copy())

    def run(self):
        chunk_size = int(self.chunk_size_sec * self.sr)
        with sd.InputStream(
            channels=1,
            samplerate=self.sr,
            blocksize=chunk_size,
            callback=self.audio_callback,
        ):
            print("🎤 Listening... Press Ctrl+C to stop.")
            try:
                while True:
                    chunk = self.q.get()
                    self._process_chunk(chunk)
            except KeyboardInterrupt:
                print("Stopping.")


# ----------------------------
# Main
# ----------------------------
def main():
    analyzer = LiveAudioAnalyzer(
        wav2vec_model_path="./multilingual_emotion_model",  # Path to your Wav2Vec2 model
        chunk_size_sec=1.0,       # mic processing chunk size
        emotion_window_sec=5.0,   # emotion analysis window
        sr=16000,
        osc_ip="127.0.0.1",       # TouchDesigner IP
        osc_port=5005,            # TouchDesigner port
        rms_threshold=0.001       # silence detection threshold
    )
    analyzer.run()


if __name__ == "__main__":
    main()
