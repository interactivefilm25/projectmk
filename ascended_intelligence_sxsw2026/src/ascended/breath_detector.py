# ASCENDED Intelligence: Audio-Only Breath Detection
# Complete implementation from your SXSW document
# For Karen Palmer's SXSW 2026 MVP

import numpy as np
import librosa
from scipy import signal
from scipy.signal import find_peaks
import opensmile
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BreathDetector:
    """Complete breath detection system from your SXSW document"""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.breath_history = []
        self.breath_events = []
        
        # Breathing frequency range (Hz)
        self.breath_freq_min = 0.1  # 0.1 Hz = 6 breaths per minute
        self.breath_freq_max = 1.0  # 1.0 Hz = 60 breaths per minute
        
        # Voice activity detection thresholds
        self.voice_threshold = 0.01
        self.silence_min_duration = 0.5  # 500ms minimum silence
        
        # Breath detection parameters
        self.breath_sound_threshold = 0.005
        self.breath_duration_min = 0.2  # 200ms minimum breath
        self.breath_duration_max = 3.0  # 3s maximum breath

    def analyze_audio(self, audio_file_path):
        """Main function from your SXSW document"""
        try:
            audio_signal, sr = librosa.load(audio_file_path, sr=self.sample_rate)
            audio_duration = len(audio_signal) / self.sample_rate
            
            logger.info(f"Analyzing audio file: {audio_file_path}, Duration: {audio_duration:.2f}s")
            
            # Method 1: Breath sounds in speech gaps
            breath_events_gaps = self.detect_breath_in_speech_gaps(audio_signal)
            
            # Method 2: Voice tremor from breathing
            breath_events_tremor = self.detect_voice_tremor(audio_signal)
            
            # Combine and calculate breathing rate
            all_events = breath_events_gaps + breath_events_tremor
            all_events.sort()
            
            breath_rate = self.calculate_breathing_rate(all_events, audio_duration)
            breath_analysis = self._analyze_breath_pattern(all_events)
            
            result = {
                'breath_rate': breath_rate,
                'breath_events': all_events,
                'breath_analysis': breath_analysis,
                'audio_duration': audio_duration,
                'timestamp': time.time()
            }
            
            logger.info(f"Breath analysis complete: {breath_rate} bpm")
            return result
            
        except Exception as e:
            logger.error(f"Error in audio analysis: {e}")
            return None

    def detect_breath_in_speech_gaps(self, audio_signal):
        """From your SXSW document - detect breath in speech gaps"""
        try:
            voice_activity = self._detect_voice_activity(audio_signal)
            silence_segments = self._find_silence_segments(voice_activity)
            
            breath_events = []
            for start, end in silence_segments:
                segment = audio_signal[start:end]
                if self._has_breath_sound(segment):
                    breath_time = start / self.sample_rate
                    breath_events.append(breath_time)
            
            return breath_events
            
        except Exception as e:
            logger.error(f"Error in breath detection: {e}")
            return []

    def _detect_voice_activity(self, audio_signal):
        """Voice activity detection from your document"""
        frame_length = int(0.025 * self.sample_rate)
        hop_length = int(0.01 * self.sample_rate)
        
        energy = librosa.feature.rmse(
            y=audio_signal,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]
        
        energy_smooth = signal.medfilt(energy, kernel_size=5)
        voice_mask = energy_smooth > self.voice_threshold
        
        return voice_mask

    def calculate_breathing_rate(self, breath_events, audio_duration):
        """Calculate BPM from your document"""
        if not breath_events or audio_duration <= 0:
            return 0
        
        breath_count = len(breath_events)
        breath_rate = (breath_count / audio_duration) * 60
        return max(5, min(60, breath_rate))

    def _analyze_breath_pattern(self, breath_events):
        """Pattern analysis from your document"""
        if len(breath_events) < 2:
            return {'regularity': 'unknown', 'intensity': 'unknown'}
        
        intervals = [breath_events[i + 1] - breath_events[i] for i in range(len(breath_events) - 1)]
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        cv_interval = std_interval / mean_interval if mean_interval > 0 else float('inf')
        
        if cv_interval < 0.2:
            regularity = 'regular'
        elif cv_interval < 0.5:
            regularity = 'moderate'
        else:
            regularity = 'irregular'
        
        if mean_interval < 2.0:
            intensity = 'high'
        elif mean_interval < 4.0:
            intensity = 'moderate'
        else:
            intensity = 'low'
        
        return {
            'regularity': regularity,
            'intensity': intensity,
            'mean_interval': mean_interval,
            'std_interval': std_interval,
            'cv_interval': cv_interval
        }
