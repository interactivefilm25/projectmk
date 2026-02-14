"""ASCENDED Intelligence - VBI (Vocal Breathiness Index) module."""

from .vbi import (
    compute_vbi,
    vbi_to_target_hz,
    vbi_to_breath_state,
    vbi_bloom_modifier,
    calculate_weighted_resonance,
    get_noise_floor_top_db,
    set_noise_floor_top_db,
    calibrate_noise_floor,
    adapt_noise_floor_for_silence,
    BREATH_STATE_LABELS,
    VBI_GAIN,
    VBI_WEIGHT,
    EMOTION_WEIGHT,
)

__all__ = [
    "compute_vbi",
    "vbi_to_target_hz",
    "vbi_to_breath_state",
    "vbi_bloom_modifier",
    "calculate_weighted_resonance",
    "get_noise_floor_top_db",
    "set_noise_floor_top_db",
    "calibrate_noise_floor",
    "adapt_noise_floor_for_silence",
    "BREATH_STATE_LABELS",
    "VBI_GAIN",
    "VBI_WEIGHT",
    "EMOTION_WEIGHT",
]
