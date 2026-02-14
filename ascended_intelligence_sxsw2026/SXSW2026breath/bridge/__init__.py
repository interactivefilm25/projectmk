# Bridge: combined BreathDetector + Audio2Emotion (Option A).
# No tests here; use test_recorded_audio.py and test_live_mic.py.
# OSC output is sent automatically after each prediction.

from .combined import load_audio, run_combined, prepare_audio
from .osc import osc_client, configure_osc, get_osc_info

__all__ = [
    "load_audio", "run_combined", "prepare_audio",
    "osc_client", "configure_osc", "get_osc_info",
]
