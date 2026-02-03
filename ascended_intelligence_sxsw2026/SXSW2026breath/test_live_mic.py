#!/usr/bin/env python3
"""
Test combined emotion pipeline on live microphone input.

Records for a fixed duration, then runs Breath + emotion model.
Use by: (1) speaking into the mic, or (2) playing an audio file through speakers.

Usage:
  conda activate ascending_intelligence
  pip install pyaudio tqdm   # if not already installed
  python test_live_mic.py
  python test_live_mic.py --duration 6
"""
import os
os.environ["PYTHONUNBUFFERED"] = "1"  # Disable output buffering
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*version.*")
warnings.filterwarnings("ignore", message=".*CUDAExecutionProvider.*")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

TARGET_SR = 16000
RECORD_RATE = 44100
DEFAULT_DURATION_SEC = 4.0


def record_from_mic(duration_sec: float, sample_rate: int = RECORD_RATE) -> np.ndarray:
    """Record from default microphone with tqdm progress bar. Returns float32 mono waveform."""
    try:
        import pyaudio
        from tqdm import tqdm
    except ImportError as e:
        raise ImportError("Install pyaudio and tqdm: pip install pyaudio tqdm") from e

    chunk = 1024
    total_samples = int(duration_sec * sample_rate)
    buffer = []

    # Suppress ALSA warnings
    stderr_fd = os.dup(2)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 2)
        p = pyaudio.PyAudio()
        try:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                input=True,
                frames_per_buffer=chunk,
            )
            needed = total_samples
            sys.stdout.flush()
            with tqdm(total=int(duration_sec), desc="Recording", unit="s", 
                      bar_format="{desc}: {bar}| {n}/{total}s", ncols=50, file=sys.stdout) as pbar:
                last_sec = 0
                while needed > 0:
                    to_read = min(needed, chunk)
                    data = stream.read(to_read, exception_on_overflow=False)
                    arr = np.frombuffer(data, dtype=np.int16)
                    buffer.append(arr)
                    needed -= len(arr)
                    current_sec = int((total_samples - needed) / sample_rate)
                    if current_sec > last_sec:
                        pbar.update(current_sec - last_sec)
                        last_sec = current_sec
            sys.stdout.flush()
            stream.stop_stream()
            stream.close()
        finally:
            p.terminate()
    finally:
        os.dup2(stderr_fd, 2)
        os.close(stderr_fd)

    waveform = np.concatenate(buffer)[:total_samples].astype(np.float32) / 32768.0
    return waveform


def resample_to_16k(waveform: np.ndarray, orig_sr: int) -> np.ndarray:
    """Resample to 16000 Hz for the bridge."""
    if orig_sr == TARGET_SR:
        return waveform
    try:
        import librosa
        return librosa.resample(waveform.astype(np.float32), orig_sr=orig_sr, target_sr=TARGET_SR)
    except ImportError:
        from scipy import signal
        num = int(len(waveform) * TARGET_SR / orig_sr)
        return signal.resample(waveform, num).astype(np.float32)


def preprocess_audio(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Simple preprocessing: just normalize volume.
    No filtering (can remove emotional cues).
    """
    w = np.asarray(waveform, dtype=np.float32)
    if w.ndim > 1:
        w = w.mean(axis=1)
    
    # Only normalize volume - don't filter or trim
    peak = np.max(np.abs(w))
    if peak > 1e-6:
        w = w * (0.95 / peak)
    
    return w


def main():
    duration = DEFAULT_DURATION_SEC
    save_recording = "--save" in sys.argv
    if "--duration" in sys.argv:
        i = sys.argv.index("--duration")
        if i + 1 < len(sys.argv):
            duration = float(sys.argv[i + 1])

    # Header - flush immediately so user sees it before recording
    print("=" * 80, flush=True)
    print("COMBINED EMOTION TEST (Live Microphone)", flush=True)
    print("=" * 80, flush=True)
    print(f"\nRecording for {duration:.1f} seconds at {RECORD_RATE} Hz.", flush=True)
    print("Speak into the microphone, or play an audio file through speakers.\n", flush=True)

    # Step 1: Record with tqdm
    print("[1/4] Recording...", flush=True)
    waveform_raw = record_from_mic(duration, sample_rate=RECORD_RATE)
    print(f"      Recorded {len(waveform_raw)} samples", flush=True)

    # Step 2: Resample to 16 kHz
    print("[2/4] Resampling to 16 kHz...", flush=True)
    waveform = resample_to_16k(waveform_raw, RECORD_RATE)
    print(f"      {len(waveform)} samples ({len(waveform)/TARGET_SR:.1f} s)", flush=True)

    # Save recording if requested (for debugging)
    if save_recording:
        import soundfile as sf
        save_path = PROJECT_ROOT / "live_recording.wav"
        sf.write(str(save_path), waveform, TARGET_SR)
        print(f"      Saved to: {save_path}", flush=True)

    # Step 3: Preprocess (normalize)
    print("[3/4] Preprocessing audio...", flush=True)
    waveform = preprocess_audio(waveform, TARGET_SR)
    duration_sec = len(waveform) / TARGET_SR
    print(f"      {len(waveform)} samples ({duration_sec:.1f} s)", flush=True)

    # Step 4: Run prediction (model loads here with progress bar)
    print("[4/4] Predicting...", flush=True)
    import logging
    logging.getLogger("ascended.breath_detector").setLevel(logging.WARNING)
    from bridge import run_combined
    segments = run_combined(waveform=waveform, sample_rate=TARGET_SR, segment_duration=None)

    # Print results (same format as test_recorded_audio.py)
    def _time_range(s):
        st, et = s["start_time"], s["end_time"]
        return f"{int(st)}–{int(et)} s" if et == int(et) else f"{int(st)}–{et:.1f} s"

    sep = " | "
    print("\nTime" + sep + "F0 (Hz)" + sep + "BPM" + sep + "Breath State" + sep + "Audio2Emotion" + sep + "Target Hz", flush=True)
    print("-" * 90, flush=True)
    for s in segments:
        t = _time_range(s)
        f0 = s["breath_f0"]
        bpm = s["breath_bpm"]
        breath_str = s.get("primary_emotion", s["breath_emotion"])
        probs = s.get("audio2emotion_probs", {})
        if probs:
            top = max(probs.items(), key=lambda x: x[1])
            a2e_str = f"{top[0]} {top[1]*100:.0f}%"
        else:
            a2e_str = "—"
        target_hz = s.get("target_frequency_hz", "—")
        print(t + sep + f"{f0:.1f}" + sep + f"{bpm:.1f}" + sep + breath_str + sep + a2e_str + sep + str(target_hz), flush=True)

    print("\nOVERALL (by segment)", flush=True)
    print("-" * 90, flush=True)
    primary_counts = {}
    a2e_counts = {}
    for s in segments:
        p = s.get("primary_emotion", s["breath_emotion"])
        primary_counts[p] = primary_counts.get(p, 0) + 1
        a2e_counts[s["audio2emotion_emotion"]] = a2e_counts.get(s["audio2emotion_emotion"], 0) + 1
    n = len(segments)
    primary_top = max(primary_counts.items(), key=lambda x: x[1]) if primary_counts else ("—", 0)
    a2e_top = max(a2e_counts.items(), key=lambda x: x[1]) if a2e_counts else ("—", 0)
    print(f"  Breath State (BPM map): {primary_top[0]}: {100*primary_top[1]/n:.1f}%", flush=True)
    print(f"  Audio2Emotion (model):  {a2e_top[0]}: {100*a2e_top[1]/n:.1f}%", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
