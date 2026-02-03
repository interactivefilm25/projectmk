#!/usr/bin/env python3
"""
Test combined emotion (Breath + Audio2Emotion) on a recorded audio file.

Usage:
  conda activate ascending_intelligence
  python test_recorded_audio.py
  python test_recorded_audio.py path/to/audio.wav
  python test_recorded_audio.py test.ogg
"""
import os
# Suppress HuggingFace/transformers verbose output before imports
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import sys
import warnings
from pathlib import Path

# Suppress noisy library warnings so output is readable
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*version.*")
warnings.filterwarnings("ignore", message=".*CUDAExecutionProvider.*")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bridge import load_audio, run_combined


def main():
    default_audio = PROJECT_ROOT / "test.ogg"
    audio_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_audio

    if not audio_path.exists():
        print(f"ERROR: File not found: {audio_path}")
        sys.exit(1)

    print("=" * 80)
    print("COMBINED EMOTION TEST (Recorded Audio)")
    print("=" * 80)
    print(f"\nAudio file: {audio_path}\n")

    # Full audio: one result for the entire recording (no segmentation)
    segments = run_combined(audio_path=audio_path, segment_duration=None)

    # Table: pipe-separated columns
    def _time_range(s):
        st, et = s["start_time"], s["end_time"]
        return f"{int(st)}–{int(et)} s" if et == int(et) else f"{int(st)}–{et:.1f} s"

    sep = " | "
    print("Time" + sep + "F0 (Hz)" + sep + "BPM" + sep + "Breath State" + sep + "Audio2Emotion" + sep + "Target Hz")
    print("-" * 90)
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
        print(t + sep + f"{f0:.1f}" + sep + f"{bpm:.1f}" + sep + breath_str + sep + a2e_str + sep + str(target_hz))

    # OVERALL: both breath state (strict BPM map) and Audio2Emotion (model prediction)
    print("\nOVERALL (by segment)")
    print("-" * 90)
    primary_counts = {}
    a2e_counts = {}
    for s in segments:
        p = s.get("primary_emotion", s["breath_emotion"])
        primary_counts[p] = primary_counts.get(p, 0) + 1
        a2e_counts[s["audio2emotion_emotion"]] = a2e_counts.get(s["audio2emotion_emotion"], 0) + 1
    n = len(segments)
    primary_top = max(primary_counts.items(), key=lambda x: x[1]) if primary_counts else ("—", 0)
    a2e_top = max(a2e_counts.items(), key=lambda x: x[1]) if a2e_counts else ("—", 0)
    print(f"  Breath State (BPM map): {primary_top[0]}: {100*primary_top[1]/n:.1f}%")
    print(f"  Audio2Emotion (model):  {a2e_top[0]}: {100*a2e_top[1]/n:.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
