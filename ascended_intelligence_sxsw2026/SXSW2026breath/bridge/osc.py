"""
OSC output for TouchDesigner. Client spec: /vbi, /hz, /emotion, /heartbeat, /active, /glitch.
Port 5005.
"""

OSC_ENABLED = True
OSC_IP = "127.0.0.1"
OSC_PORT = 5005

EMOTION_MAP_REALTIME = {
    "anger_fear": 0,
    "joy_excited": 1,
    "sadness": 2,
    "curious_reflective": 3,
    "calm_content": 4,
    "angry": 0,
    "fear": 0,
    "happy": 1,
    "surprised": 1,
    "sad": 2,
    "disgust": 3,
    "neutral": 4,
    "unknown": -1,
}


class OSCClient:
    """Client spec: /vbi, /hz, /emotion, /heartbeat, /active, /glitch."""
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
    
    def send(
        self,
        emotion: str,
        target_hz: int | None = None,
        vbi: float | None = None,
        bloom: float | None = None,
    ):
        """Client spec: /vbi, /hz, /emotion, /heartbeat, /active, /glitch, /bloom."""
        if not self._ensure_initialized():
            return
        try:
            emotion_lower = emotion.lower()
            emotion_int = EMOTION_MAP_REALTIME.get(emotion_lower, -1)
            self._client.send_message("/emotion", emotion_int)
            if target_hz is not None:
                self._client.send_message("/hz", float(target_hz))
            if vbi is not None:
                self._client.send_message("/vbi", float(vbi))
            if bloom is not None:
                self._client.send_message("/bloom", float(bloom))
            is_calm = emotion_lower in ("calm_content", "calm")
            is_glitch = 1.0 if (vbi is not None and vbi > 0.7 and is_calm) else 0.0
            self._client.send_message("/glitch", is_glitch)
            self._client.send_message("/heartbeat", 1)
            self._client.send_message("/active", 1)
        except Exception:
            pass
    
    def send_raw(self, address: str, value):
        """Send a raw OSC message."""
        if not self._ensure_initialized():
            return
        try:
            self._client.send_message(address, value)
        except Exception:
            pass

    def send_heartbeat(self, active: int = 1):
        """Send /heartbeat and /active (1=alive, 0=severed). TD fades to black when 0."""
        if not self._ensure_initialized():
            return
        try:
            self._client.send_message("/heartbeat", int(active))
            self._client.send_message("/active", int(active))
            if active == 0:
                self._client.send_message("/hz", 528.0)  # Severing: grounding tone
        except Exception:
            pass

    def sever(self):
        """Severing Protocol: send /heartbeat 0, /active 0, /hz 528 for graceful ritual end."""
        self.send_heartbeat(0)


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
    return {"enabled": OSC_ENABLED, "ip": OSC_IP, "port": OSC_PORT}
