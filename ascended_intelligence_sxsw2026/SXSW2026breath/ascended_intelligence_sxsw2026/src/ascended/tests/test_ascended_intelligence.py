#!/usr/bin/env python3
"""Test suite for ASCENDED Intelligence VBI module - SXSW 2026"""

import sys
from pathlib import Path

# Add ascended src to path
_ascended_src = Path(__file__).resolve().parent.parent
if str(_ascended_src) not in sys.path:
    sys.path.insert(0, str(_ascended_src.parent))

def test_vbi():
    """Test VBI computation"""
    import numpy as np
    from ascended.vbi import compute_vbi, vbi_to_target_hz, vbi_to_breath_state, calculate_weighted_resonance

    sr = 16000
    # Silence -> low VBI
    silence = np.zeros(sr * 2, dtype=np.float64)
    vbi_s = compute_vbi(silence, sr)
    assert vbi_s < 0.5, f"Silence VBI expected < 0.5, got {vbi_s}"
    assert vbi_to_target_hz(vbi_s) == 963
    assert vbi_to_breath_state(vbi_s) == "calm"

    # Weighted resonance
    w = calculate_weighted_resonance(0.5, 4)  # calm emotion
    assert 0 <= w <= 1

    print("✅ VBI tests passed")

if __name__ == "__main__":
    test_vbi()
