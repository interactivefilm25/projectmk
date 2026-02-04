"""
OSC (Open Sound Control) output for TouchDesigner / Max / other visual software.

Usage:
    from bridge.osc import osc_client, configure_osc
    
    # Configure (optional)
    configure_osc(ip="192.168.1.100", port=9000)
    
    # Send data
    osc_client.send(emotion="happy", confidence=0.95, f0=150.0, 
                    breath_state="calm", breath_bpm=18.0)
"""

# ============================================================================
# OSC Configuration
# ============================================================================
OSC_ENABLED = True           # Set to False to disable OSC
OSC_IP = "127.0.0.1"         # TouchDesigner IP
OSC_PORT = 5005              # TouchDesigner port

# Emotion to integer mapping (realtime_emotion_td.py format for TouchDesigner)
# anger_fear=0, joy_excited=1, sadness=2, curious_reflective=3, calm_content=4, unknown=-1
# Maps emotion2vec labels -> realtime 0-4
EMOTION_MAP_REALTIME = {
    "angry": 0,      # anger_fear
    "fear": 0,       # anger_fear
    "happy": 1,      # joy_excited
    "surprised": 1,  # joy_excited
    "sad": 2,        # sadness
    "disgust": 3,    # curious_reflective
    "neutral": 4,    # calm_content
    "unknown": -1,
}

# Legacy 8-label map (kept for reference)
EMOTION_MAP = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5,
    "surprised": 6,
    "unknown": -1,
}

# Breath state to integer mapping (used internally, not sent in realtime format)
BREATH_STATE_MAP = {
    "calm": 0,
    "slightly_elevated": 1,
    "anxious": 2,
    "high_anxiety": 3,
}


class OSCClient:
    """
    Singleton OSC client for sending emotion/breath data to TouchDesigner.
    Uses same format as realtime_emotion_td.py:
    
    OSC Messages:
      /emotion   - emotion as integer (0-4: anger_fear, joy_excited, sadness, curious_reflective, calm_content; -1=unknown)
      /frequency - F0 normalized to 0.0-1.0 (50-400Hz vocal range)
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._client = None
            cls._instance._initialized = False
        return cls._instance
    
    def _ensure_initialized(self):
        if self._initialized:
            return self._client is not None
        self._initialized = True
        if not OSC_ENABLED:
            return False
        try:
            from pythonosc import udp_client
            self._client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)
            return True
        except ImportError:
            return False
        except Exception:
            return False
    
    def reset(self):
        """Reset client to apply new configuration."""
        self._initialized = False
        self._client = None
    
    def send(self, emotion: str, confidence: float, f0: float, breath_state: str, breath_bpm: float):
        """
        Send emotion and frequency via OSC (realtime_emotion_td.py format).
        Only /emotion and /frequency are sent, matching TouchDesigner expectations.
        
        Args:
            emotion: Emotion name from emotion2vec (e.g., "happy", "angry")
            confidence: Confidence score (0.0-1.0) (unused, for API compatibility)
            f0: Fundamental frequency in Hz
            breath_state: Breath state label (unused, for API compatibility)
            breath_bpm: Breath rate in BPM (unused, for API compatibility)
        """
        if not self._ensure_initialized():
            return
        try:
            # Emotion as integer (0-4, same as realtime_emotion_td.py)
            emotion_int = EMOTION_MAP_REALTIME.get(emotion.lower(), -1)
            self._client.send_message("/emotion", emotion_int)
            
            # Normalized frequency (50-400Hz range -> 0.0-1.0)
            if f0 > 0:
                freq_norm = max(0.0, min(1.0, (f0 - 50) / (400 - 50)))
            else:
                freq_norm = 0.0
            self._client.send_message("/frequency", freq_norm)
            
        except Exception:
            pass  # Silently ignore OSC errors
    
    def send_raw(self, address: str, value):
        """Send a raw OSC message."""
        if not self._ensure_initialized():
            return
        try:
            self._client.send_message(address, value)
        except Exception:
            pass


# Global OSC client instance
osc_client = OSCClient()


def configure_osc(ip: str = None, port: int = None, enabled: bool = None):
    """
    Configure OSC settings. Call before using osc_client to change defaults.
    
    Args:
        ip: OSC target IP (default: 127.0.0.1)
        port: OSC target port (default: 5005)
        enabled: Enable/disable OSC (default: True)
    
    Example:
        from bridge.osc import configure_osc
        configure_osc(ip="192.168.1.100", port=9000)
    """
    global OSC_IP, OSC_PORT, OSC_ENABLED
    if ip is not None:
        OSC_IP = ip
    if port is not None:
        OSC_PORT = port
    if enabled is not None:
        OSC_ENABLED = enabled
    # Reset client to apply new settings
    osc_client.reset()


def get_osc_info() -> dict:
    """Get current OSC configuration."""
    return {
        "enabled": OSC_ENABLED,
        "ip": OSC_IP,
        "port": OSC_PORT,
        "emotion_map": EMOTION_MAP_REALTIME,
        "breath_state_map": BREATH_STATE_MAP,
    }
