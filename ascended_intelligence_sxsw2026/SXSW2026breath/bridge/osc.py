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

# Emotion to integer mapping for TouchDesigner
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

# Breath state to integer mapping
BREATH_STATE_MAP = {
    "calm": 0,
    "slightly_elevated": 1,
    "anxious": 2,
    "high_anxiety": 3,
}


class OSCClient:
    """
    Singleton OSC client for sending emotion/breath data to TouchDesigner.
    
    OSC Messages:
      /emotion   - emotion as integer (0-6, -1=unknown)
      /frequency - F0 normalized to 0.0-1.0 (50-400Hz vocal range)
      /breath    - breath state as integer (0-3)
      /bpm       - breath BPM normalized to 0.0-1.0 (10-40 BPM range)
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
        Send emotion and breath data via OSC.
        
        Args:
            emotion: Emotion name (e.g., "happy", "angry")
            confidence: Confidence score (0.0-1.0)
            f0: Fundamental frequency in Hz
            breath_state: Breath state label (e.g., "High anxiety", "Calm and relaxed")
            breath_bpm: Breath rate in BPM
        """
        if not self._ensure_initialized():
            return
        try:
            # Emotion as integer
            emotion_int = EMOTION_MAP.get(emotion.lower(), -1)
            self._client.send_message("/emotion", emotion_int)
            
            # Normalized frequency (50-400Hz range -> 0.0-1.0)
            if f0 > 0:
                freq_norm = max(0.0, min(1.0, (f0 - 50) / (400 - 50)))
            else:
                freq_norm = 0.0
            self._client.send_message("/frequency", freq_norm)
            
            # Breath state as integer
            breath_key = breath_state.lower().replace(" ", "_").replace("and_", "")
            if breath_key == "calm_relaxed":
                breath_key = "calm"
            breath_int = BREATH_STATE_MAP.get(breath_key, 1)
            self._client.send_message("/breath", breath_int)
            
            # Breath BPM normalized (10-40 BPM range -> 0.0-1.0)
            if breath_bpm > 0:
                bpm_norm = max(0.0, min(1.0, (breath_bpm - 10) / (40 - 10)))
            else:
                bpm_norm = 0.5
            self._client.send_message("/bpm", bpm_norm)
            
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
        "emotion_map": EMOTION_MAP,
        "breath_state_map": BREATH_STATE_MAP,
    }
