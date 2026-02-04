#!/usr/bin/env python3
"""
Dynamic live microphone emotion detection with OSC output.

Features:
- Voice Activity Detection (VAD): automatically detects when person is speaking
- Audio denoising and enhancement using noisereduce (spectral gating)
- Real-time emotion prediction
- OSC output for TouchDesigner (sent from bridge)

Usage:
  conda activate ascending_intelligence
  pip install sounddevice noisereduce python-osc  # required packages
  python test_live_mic.py
  python test_live_mic.py --osc-ip 127.0.0.1 --osc-port 5005  # configure OSC
  python test_live_mic.py --no-osc                # disable OSC

To test OSC: run osc_receiver.py in another terminal, then run this script.
"""
import os
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import sys
import queue
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*version.*")
warnings.filterwarnings("ignore", message=".*CUDAExecutionProvider.*")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

MODEL_SR = 16000            # Emotion model expects 16kHz
RECORD_SR = 44100           # Recording sample rate (most compatible)
CHUNK_SIZE_SEC = 0.5        # Process audio in 0.5s chunks
EMOTION_WINDOW_SEC = 3.0    # Analyze emotion over 3s of speech
RMS_THRESHOLD = 0.01        # Voice activity detection threshold
SILENCE_TIMEOUT_SEC = 1.5   # Stop after this much silence


class LiveEmotionAnalyzer:
    """
    Dynamic live microphone emotion analyzer with VAD.
    Automatically detects speech, buffers it, and predicts emotion.
    OSC is configured before run() via bridge.configure_osc().
    """
    
    def __init__(self, show_osc_sent: bool = True):
        self.record_sr = RECORD_SR
        self.model_sr = MODEL_SR
        self.show_osc_sent = show_osc_sent
        self.chunk_size = int(CHUNK_SIZE_SEC * self.record_sr)
        self.emotion_window = int(EMOTION_WINDOW_SEC * self.record_sr)
        self.silence_chunks = int(SILENCE_TIMEOUT_SEC / CHUNK_SIZE_SEC)
        self.rms_threshold = RMS_THRESHOLD
        self.audio_queue = queue.Queue()
        self.speech_buffer = []
        self.silence_count = 0
        self.is_speaking = False
        
        self._emotion_model = None
        self._run_combined = None
    
    def _ensure_emotion_model(self):
        if self._run_combined is not None:
            return
        import logging
        logging.getLogger("ascended.breath_detector").setLevel(logging.WARNING)
        from bridge import run_combined
        self._run_combined = run_combined
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Called by sounddevice for each audio chunk."""
        # Ignore overflow warnings (common with high sample rates)
        self.audio_queue.put(indata.copy())
    
    def _compute_rms(self, audio: np.ndarray) -> float:
        """Compute RMS energy of audio chunk."""
        return float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
    
    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        if orig_sr == target_sr:
            return audio
        try:
            import librosa
            return librosa.resample(audio.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            from scipy import signal
            num = int(len(audio) * target_sr / orig_sr)
            return signal.resample(audio, num).astype(np.float32)
    
    def _process_speech(self, audio: np.ndarray):
        """Process accumulated speech buffer and predict emotion."""
        if len(audio) < self.record_sr:  # At least 1 second
            print("      Speech too short, skipping", flush=True)
            return
        
        duration = len(audio) / self.record_sr
        print(f"\n[Processing {duration:.1f}s of speech]", flush=True)
        
        # Resample to model sample rate
        audio = self._resample(audio, self.record_sr, self.model_sr)
        
        # Clean + enhance (bridge: noise_model + gentle enhancement)
        print("  Preparing audio (denoise + enhance)...", flush=True)
        from bridge import prepare_audio
        audio = prepare_audio(audio, self.model_sr)
        
        # Predict emotion (audio already prepared, skip bridge clean)
        print("  Predicting...", flush=True)
        self._ensure_emotion_model()
        segments = self._run_combined(waveform=audio, sample_rate=self.model_sr, segment_duration=None, clean_audio=False)
        
        if segments:
            s = segments[0]
            probs = s.get("audio2emotion_probs", {})
            if probs:
                top = max(probs.items(), key=lambda x: x[1])
                emotion = top[0]
                confidence = top[1] * 100
                breath_state = s.get("primary_emotion", s["breath_emotion"])
                f0 = s["breath_f0"]
                print(f"\n  ┌─────────────────────────────────────┐", flush=True)
                print(f"  │ Emotion: {emotion:12} ({confidence:5.1f}%)  │", flush=True)
                print(f"  │ Breath:  {breath_state:24} │", flush=True)
                print(f"  │ F0:      {f0:6.1f} Hz                  │", flush=True)
                print(f"  └─────────────────────────────────────┘", flush=True)
                if self.show_osc_sent:
                    try:
                        from bridge.osc import get_osc_info
                        info = get_osc_info()
                        if info.get("enabled"):
                            print(f"  OSC: sent to {info['ip']}:{info['port']}\n", flush=True)
                    except Exception:
                        pass
    
    def run(self):
        """Main loop: listen for speech, detect, analyze."""
        try:
            import sounddevice as sd
        except ImportError:
            print("ERROR: sounddevice not installed. Run: pip install sounddevice", flush=True)
            sys.exit(1)
        
        print("=" * 60, flush=True)
        print("LIVE EMOTION DETECTION (Voice Activated)", flush=True)
        print("=" * 60, flush=True)
        print(f"\nSettings:", flush=True)
        print(f"  - Recording: {self.record_sr} Hz", flush=True)
        print(f"  - Model: {self.model_sr} Hz", flush=True)
        print(f"  - Emotion window: {EMOTION_WINDOW_SEC}s", flush=True)
        print(f"  - Silence timeout: {SILENCE_TIMEOUT_SEC}s", flush=True)
        print(f"  - Enhancement: bridge prepare_audio (denoise + enhance)", flush=True)
        try:
            from bridge.osc import get_osc_info
            info = get_osc_info()
            if info.get("enabled"):
                print(f"  - OSC: {info['ip']}:{info['port']} (/emotion, /frequency, /breath, /bpm)", flush=True)
            else:
                print(f"  - OSC: disabled", flush=True)
        except Exception:
            print(f"  - OSC: not available", flush=True)
        print(f"\nListening... Speak or play audio. Press Ctrl+C to stop.\n", flush=True)
        
        with sd.InputStream(
            channels=1,
            samplerate=self.record_sr,
            blocksize=self.chunk_size,
            callback=self._audio_callback,
        ):
            try:
                while True:
                    chunk = self.audio_queue.get()
                    chunk = chunk.flatten().astype(np.float32)
                    rms = self._compute_rms(chunk)
                    
                    if rms > self.rms_threshold:
                        # Voice detected
                        if not self.is_speaking:
                            self.is_speaking = True
                            print("🎤 Speech detected...", end="", flush=True)
                        self.speech_buffer.extend(chunk)
                        self.silence_count = 0
                        
                        # Show activity
                        bars = int(min(rms * 100, 20))
                        print(f"\r🎤 Recording: {'█' * bars}{' ' * (20-bars)} | {len(self.speech_buffer)/self.record_sr:.1f}s", end="", flush=True)
                    else:
                        # Silence
                        if self.is_speaking:
                            self.silence_count += 1
                            if self.silence_count >= self.silence_chunks:
                                # End of speech
                                print("\r" + " " * 60 + "\r", end="", flush=True)
                                self.is_speaking = False
                                if self.speech_buffer:
                                    audio = np.array(self.speech_buffer, dtype=np.float32)
                                    self.speech_buffer = []
                                    self._process_speech(audio)
                                self.silence_count = 0
                            else:
                                # Still in speech pause
                                self.speech_buffer.extend(chunk)
                                
            except KeyboardInterrupt:
                print("\n\nStopped.", flush=True)


def main():
    use_osc = "--no-osc" not in sys.argv
    
    # Parse OSC configuration
    osc_ip = "127.0.0.1"
    osc_port = 5005
    for i, arg in enumerate(sys.argv):
        if arg == "--osc-ip" and i + 1 < len(sys.argv):
            osc_ip = sys.argv[i + 1]
        elif arg == "--osc-port" and i + 1 < len(sys.argv):
            try:
                osc_port = int(sys.argv[i + 1])
            except ValueError:
                print(f"Invalid port, using 5005", flush=True)
    
    # Configure OSC before running
    from bridge.osc import configure_osc
    configure_osc(ip=osc_ip, port=osc_port, enabled=use_osc)
    
    analyzer = LiveEmotionAnalyzer(
        show_osc_sent=use_osc,
    )
    analyzer.run()


if __name__ == "__main__":
    main()
