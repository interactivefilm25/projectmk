# Bridge: combined BreathDetector + Audio2Emotion (Option A).
# No tests here; use test_recorded_audio.py and test_live_mic.py.

from .combined import load_audio, run_combined

__all__ = ["load_audio", "run_combined"]
