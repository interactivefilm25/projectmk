# ASCENDED Intelligence: Audio-Only Breath Detection

**For Karen Palmer's SXSW 2026 Interactive Installation**

## 🌬️ Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
python tests/test_ascended_intelligence.py

# Use breath detector
from src.ascended.breath_detector import BreathDetector
detector = BreathDetector()
result = detector.analyze_audio("audio_file.wav")
print(f"Breath rate: {result['breath_rate']} BPM")
