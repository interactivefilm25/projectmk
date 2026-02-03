# ASCENDED Intelligence: Audio-Only Breath Detection
# Complete implementation according to Breath_Sensing_Data_Pipeline_compressed.pdf
# For Karen Palmer's SXSW 2026 MVP

import numpy as np
import pandas as pd
from collections import deque
import time
import logging

# Import OpenSMILE wrapper for enhanced feature extraction
from .opensmile_wrapper import Smile
from opensmile.core.define import FeatureSet, FeatureLevel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BreathDetector:
    """
    Complete breath detection system according to PDF specification.
    Processes 100ms audio chunks in real-time with 15-second history buffer.
    """
    
    # Constants from PDF
    CHUNK_DURATION = 0.1  # 100ms chunks
    CHUNKS_PER_SECOND = 10
    HISTORY_DURATION = 15.0  # 15 seconds
    HISTORY_MAXLEN = 150  # 15 seconds * 10 chunks/sec
    
    # BPM range for normalization (from PDF)
    BPM_MIN = 8.0
    BPM_MAX = 40.0
    BPM_RANGE = BPM_MAX - BPM_MIN  # 32.0

    # Breath-rate emotional state bands per "The proposed solution" (breath_rate.jpg)
    # Under 20 = Calm and relaxed; 20-25 = Slightly elevated (normal conversation);
    # 25-30 = Anxious or stressed; Over 30 = High anxiety
    BPM_CALM_MAX = 20.0
    BPM_ELEVATED_MAX = 25.0
    BPM_ANXIOUS_MAX = 30.0
    
    # Confidence range
    CONFIDENCE_MIN = 0.3
    CONFIDENCE_MAX = 0.8
    
    # -------------------------------------------------------------------------
    # TARGET FREQUENCIES per PDF (target_frequency.jpg)
    #   396 Hz = Fear / Anxiety
    #   639 Hz = Love / Empathy
    #   963 Hz = Ascension / Joy
    # -------------------------------------------------------------------------
    TARGET_FREQUENCIES_HZ = (396, 639, 963)
    TARGET_FREQ_LABEL = {396: "Fear", 639: "Love", 963: "Ascension"}

    # BPM-based breath state → target frequency for TouchDesigner
    #   calm (< 20 BPM)           → 963 Hz Ascension
    #   slightly_elevated (20-25) → 639 Hz Love
    #   anxious (25-30)           → 396 Hz Fear
    #   high_anxiety (> 30)       → 396 Hz Fear
    BREATH_STATE_TO_TARGET_HZ = {
        "calm":               963,
        "slightly_elevated":  639,
        "anxious":            396,
        "high_anxiety":       396,
        "unknown":            639,
    }
    
    # Required OpenSMILE feature names (from PDF)
    REQUIRED_FEATURES = [
        'Loudness_sma3',
        'F0semitoneFrom27.5Hz_sma3nz',
        'pcm_RMSenergy_sma',
        'audSpec_Rfilt_sma3[0]',
        'audSpec_Rfilt_sma3[1]',
        'audSpec_Rfilt_sma3[2]',
        'jitterLocal_sma3nz'
    ]
    
    def __init__(self, sample_rate=16000, use_opensmile=True):
        self.sample_rate = sample_rate
        self.chunk_size = int(self.CHUNK_DURATION * sample_rate)  # 1600 samples at 16kHz
        self.use_opensmile = use_opensmile

        # Initialize OpenSMILE with eGeMAPSv02 (from PDF)
        if self.use_opensmile:
            try:
                self.smile = Smile(
                    feature_set=FeatureSet.eGeMAPSv02,
                    feature_level=FeatureLevel.Functionals,
                    sampling_rate=sample_rate
                )
                logger.info("OpenSMILE initialized for breath detection (eGeMAPSv02)")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenSMILE: {e}. Continuing without it.")
                self.use_opensmile = False
                self.smile = None
        else:
            self.smile = None
        
        # 15-second history buffer (from PDF: deque(maxlen=150))
        self.history_buffer = deque(maxlen=self.HISTORY_MAXLEN)
        
        # Breath detection state
        self.breath_events = []
        self.last_breath_time = None
        
        # Gap detection parameters (from PDF: "typically 0.2 to 3 seconds")
        self.gap_min_duration = 0.2  # 200ms minimum (2 chunks) per PDF
        self.gap_min_chunks = max(int(self.gap_min_duration * self.CHUNKS_PER_SECOND), 1)
        self.gap_max_duration = 3.0  # 3s maximum (30 chunks) per PDF
        self.gap_max_chunks = int(self.gap_max_duration * self.CHUNKS_PER_SECOND)
        
        # Background noise adaptation
        self.background_noise_floor = 0.0
        self.noise_adaptation_factor = 0.95
        
        # Breath detection sensitivity (tunable)
        # These thresholds were tuned based on the ASCENDED_Fixed_Breath_Detection analysis
        # to make the detector more sensitive to real-world audio and shorter clips.
        # - low_freq_threshold: minimum low-frequency energy for a "whoosh"
        # - rms_min_ratio: RMS must be this ratio above noise floor
        # - rms_max: maximum RMS for breath (breath is quiet)
        self.low_freq_threshold = 0.003
        self.rms_min_ratio = 1.05
        self.rms_max = 0.20

        # Additional energy-based thresholds (from error analysis)
        # Used for combined pitch+energy voice activity detection and breath matching.
        self.voice_energy_threshold = 0.005      # was effectively higher before
        self.breath_energy_threshold = 0.002     # low RMS for quiet breaths

        # Silence / gap tuning
        # PDF specifies breath duration "typically 0.2 to 3 seconds".
        # Both candidate gap detection and fingerprint confirmation now use 200ms minimum.
        self.silence_min_duration = 0.2  # 200ms minimum for a "silence" segment
        self.silence_min_chunks = max(int(self.silence_min_duration * self.CHUNKS_PER_SECOND), 1)
        
    def process_chunk(self, audio_chunk):
        """
        Process a single 100ms audio chunk (1600 samples at 16kHz).
        Returns TouchDesigner-formatted output.
        
        Args:
            audio_chunk: numpy array of 1600 samples (100ms at 16kHz)
            
        Returns:
            dict with TouchDesigner format:
            {
                'breath_rate_normalized': 0.0-1.0,
                'anxiety_level': 0.0 or 1.0,
                'calm_level': 0.0 or 1.0,
                'confidence': 0.3-0.8,
                'timestamp': unix timestamp
            }
        """
        try:
            # Validate chunk size
            if len(audio_chunk) != self.chunk_size:
                logger.warning(f"Chunk size mismatch: expected {self.chunk_size}, got {len(audio_chunk)}")
                # Pad or truncate if needed
                if len(audio_chunk) < self.chunk_size:
                    audio_chunk = np.pad(audio_chunk, (0, self.chunk_size - len(audio_chunk)))
                else:
                    audio_chunk = audio_chunk[:self.chunk_size]
            
            # Extract OpenSMILE features
            features = self._extract_opensmile_features(audio_chunk)
            
            # Validate features (will raise ValueError if empty, as per PDF spec)
            try:
                features = self._validate_opensmile_output(features)
            except ValueError as e:
                logger.error(f"OpenSMILE validation failed: {e}")
                # Return safe default features instead of crashing
                features = {key: 0.0 for key in self.REQUIRED_FEATURES}
            
            # Create chunk data structure
            chunk_data = {
                'timestamp': time.time(),
                'features': features,
                'audio_chunk': audio_chunk.copy()
            }
            
            # Add to history buffer
            self.history_buffer.append(chunk_data)
            
            # Analyze for breath patterns
            breath_rate, confidence = self._analyze_breath_patterns()

            # Extract F0 (Hz) for supplementary data
            f0_hz = self._extract_f0_hz(features)

            # Format output for TouchDesigner
            output = self._format_touchdesigner_output(breath_rate, confidence, f0_hz)
            
            return output
            
        except Exception as e:
            logger.error(f"Error processing chunk: {e}")
            import traceback
            traceback.print_exc()
            # Return safe default values (BPM=0 → breath_state="unknown")
            return self._format_touchdesigner_output(0.0, self.CONFIDENCE_MIN, None)
    
    def _extract_opensmile_features(self, audio_chunk):
        """
        Extract specific OpenSMILE features as specified in PDF.
        Uses actual eGeMAPSv02 feature names with _amean suffix.
        Calculates missing features (RMS energy) manually.
        """
        if self.smile is None:
            return None
        
        try:
            # Ensure signal is float32
            if audio_chunk.dtype != np.float32:
                audio_chunk = audio_chunk.astype(np.float32)

            # Compute RMS energy from the ORIGINAL signal BEFORE normalization
            # so that quiet speech and loud speech have different energy values.
            raw_rms_energy = float(np.sqrt(np.mean(audio_chunk ** 2)))

            # Normalize to [-1, 1] range for OpenSMILE
            # (pitch, jitter, spectral features are scale-invariant)
            max_val = np.max(np.abs(audio_chunk))
            if max_val > 0:
                audio_chunk = audio_chunk / max_val

            # Extract features using OpenSMILE
            features_df = self.smile.process_signal(audio_chunk, self.sample_rate)
            
            # Validate that features_df is not empty (as per PDF spec)
            if features_df.empty or len(features_df.columns) == 0:
                raise ValueError("OpenSMILE returned empty feature dataframe")
            
            # Extract specific features using actual eGeMAPSv02 names
            feature_dict = {}
            
            # Loudness_sma3 -> loudness_sma3_amean
            if 'loudness_sma3_amean' in features_df.columns:
                feature_dict['Loudness_sma3'] = float(features_df['loudness_sma3_amean'].iloc[0])
            else:
                # Fallback: search for any loudness feature
                for col in features_df.columns:
                    if 'loudness' in col.lower() and 'amean' in col:
                        feature_dict['Loudness_sma3'] = float(features_df[col].iloc[0])
                        break
            
            # F0semitoneFrom27.5Hz_sma3nz -> F0semitoneFrom27.5Hz_sma3nz_amean
            if 'F0semitoneFrom27.5Hz_sma3nz_amean' in features_df.columns:
                feature_dict['F0semitoneFrom27.5Hz_sma3nz'] = float(features_df['F0semitoneFrom27.5Hz_sma3nz_amean'].iloc[0])
            else:
                # Fallback
                for col in features_df.columns:
                    if 'F0semitoneFrom27.5Hz' in col and 'amean' in col:
                        feature_dict['F0semitoneFrom27.5Hz_sma3nz'] = float(features_df[col].iloc[0])
                        break
            
            # pcm_RMSenergy_sma - Use RMS computed from the original (pre-normalization)
            # signal so energy reflects actual loudness, not always ~0.5 after norm.
            feature_dict['pcm_RMSenergy_sma'] = raw_rms_energy
            
            # audSpec_Rfilt_sma3[0-2] - Low-frequency "whoosh" detection
            # eGeMAPSv02 doesn't have exact audSpec_Rfilt_sma3, so we use best available proxies:
            # - slopeV0-500: spectral slope in 0-500Hz (low-freq energy indicator)
            # - mfcc1_sma3: first MFCC coefficient (captures low-freq spectral shape)
            # - alphaRatioV: ratio of low to high frequency energy
            
            # Band 0: Very low frequency (0-500Hz) - breath "whoosh" primary band
            if 'slopeV0-500_sma3nz_amean' in features_df.columns:
                feature_dict['audSpec_Rfilt_sma3[0]'] = float(features_df['slopeV0-500_sma3nz_amean'].iloc[0])
            elif 'mfcc1_sma3_amean' in features_df.columns:
                # MFCC1 captures low-frequency spectral shape
                feature_dict['audSpec_Rfilt_sma3[0]'] = abs(float(features_df['mfcc1_sma3_amean'].iloc[0]))
            else:
                feature_dict['audSpec_Rfilt_sma3[0]'] = 0.0
            
            # Band 1: Low-mid frequency (500-1500Hz)
            if 'slopeV500-1500_sma3nz_amean' in features_df.columns:
                feature_dict['audSpec_Rfilt_sma3[1]'] = float(features_df['slopeV500-1500_sma3nz_amean'].iloc[0])
            elif 'mfcc2_sma3_amean' in features_df.columns:
                feature_dict['audSpec_Rfilt_sma3[1]'] = abs(float(features_df['mfcc2_sma3_amean'].iloc[0]))
            else:
                feature_dict['audSpec_Rfilt_sma3[1]'] = 0.0
            
            # Band 2: Mid frequency (1500-3000Hz) - breath harmonics
            if 'alphaRatioV_sma3nz_amean' in features_df.columns:
                # Alpha ratio: low-freq / high-freq energy ratio
                feature_dict['audSpec_Rfilt_sma3[2]'] = float(features_df['alphaRatioV_sma3nz_amean'].iloc[0])
            elif 'spectralFluxV_sma3nz_amean' in features_df.columns:
                feature_dict['audSpec_Rfilt_sma3[2]'] = float(features_df['spectralFluxV_sma3nz_amean'].iloc[0])
            else:
                feature_dict['audSpec_Rfilt_sma3[2]'] = 0.0
            
            # jitterLocal_sma3nz -> jitterLocal_sma3nz_amean
            if 'jitterLocal_sma3nz_amean' in features_df.columns:
                feature_dict['jitterLocal_sma3nz'] = float(features_df['jitterLocal_sma3nz_amean'].iloc[0])
            else:
                # Fallback
                for col in features_df.columns:
                    if 'jitterLocal' in col and 'amean' in col:
                        feature_dict['jitterLocal_sma3nz'] = float(features_df[col].iloc[0])
                        break
            
            return feature_dict
            
        except Exception as e:
            logger.error(f"Error extracting OpenSMILE features: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _validate_opensmile_output(self, features):
        """
        Validate OpenSMILE output as specified in PDF:
        - Check if empty -> raise ValueError (as per PDF spec)
        - Fill NaN values with 0.0
        - Validate required features
        """
        if features is None or len(features) == 0:
            # As per PDF: "if features_df.empty: raise ValueError"
            raise ValueError("OpenSMILE features are empty or None")
        
        # Fill NaN values with 0.0
        validated_features = {}
        for key, value in features.items():
            if pd.isna(value) or np.isnan(value):
                validated_features[key] = 0.0
            else:
                validated_features[key] = float(value)
        
        # Ensure all required features are present
        for feature_name in self.REQUIRED_FEATURES:
            if feature_name not in validated_features:
                # Try to find similar feature
                found = False
                for key in validated_features.keys():
                    if feature_name.split('[')[0] in key or feature_name.split('_')[0] in key:
                        validated_features[feature_name] = validated_features[key]
                        found = True
                        break
                if not found:
                    validated_features[feature_name] = 0.0
                    logger.debug(f"Missing required feature: {feature_name}, using 0.0")
        
        return validated_features
    
    def _detect_voice_activity_f0(self, features):
        """
        Detect voice activity using F0 as specified in PDF.
        If F0 == 0, no voice detected.

        Enhanced version:
        - Combines pitch (F0) and energy (pcm_RMSenergy_sma)
        - Matches the error-analysis recommendation:
          voice_active = (pitch > 0) AND (energy > voice_threshold)
        """
        f0_key = 'F0semitoneFrom27.5Hz_sma3nz'
        if f0_key not in features:
            return False

        f0_value = features[f0_key]
        rms_energy = features.get('pcm_RMSenergy_sma', 0.0)

        # Pitch present?
        has_pitch = f0_value != 0.0 and not np.isnan(f0_value)
        # Enough energy to consider this "speech", not just noise
        has_energy = rms_energy is not None and rms_energy >= self.voice_energy_threshold

        return bool(has_pitch and has_energy)
    
    def _analyze_breath_patterns(self):
        """
        Analyze 15-second history buffer for breath patterns.
        Implements gap identification and acoustic fingerprint matching.
        """
        if len(self.history_buffer) < self.gap_min_chunks:
            return 0.0, self.CONFIDENCE_MIN
        
        # Identify gaps where voice_activity == False
        gaps = self._identify_gaps()
        
        # Pattern match: Confirm acoustic fingerprint within gaps
        breath_events = []
        evidence_count = 0
        
        for gap_start, gap_end in gaps:
            if self._match_breath_fingerprint(gap_start, gap_end):
                # Calculate breath time (middle of gap) in seconds from buffer start
                # Use relative time within buffer
                gap_center_chunk = (gap_start + gap_end) / 2.0
                # Convert to absolute time (chunks are 0.1s apart)
                breath_time = gap_center_chunk * self.CHUNK_DURATION
                breath_events.append(breath_time)
                evidence_count += 1
        
        # Calculate breath rate from events in history window
        if len(breath_events) >= 2:
            # Sort events by time
            breath_events.sort()
            
            # Calculate intervals between consecutive events
            intervals = [breath_events[i+1] - breath_events[i] for i in range(len(breath_events)-1)]
            
            # Filter out unrealistic intervals (too short or too long)
            valid_intervals = [i for i in intervals if 1.0 <= i <= 12.0]  # 5-60 BPM range
            
            if len(valid_intervals) > 0:
                mean_interval = np.mean(valid_intervals)
                if mean_interval > 0:
                    breath_rate = 60.0 / mean_interval
                else:
                    breath_rate = 0.0
            else:
                breath_rate = 0.0
        elif len(breath_events) == 1:
            # Single event - estimate from buffer duration
            buffer_duration = len(self.history_buffer) * self.CHUNK_DURATION
            if buffer_duration > 0:
                breath_rate = 60.0 / buffer_duration  # Rough estimate
            else:
                breath_rate = 0.0
        else:
            breath_rate = 0.0
        
        # Clamp to valid range
        breath_rate = np.clip(breath_rate, 0.0, 60.0)
        
        # Calculate confidence based on evidence quality
        confidence = self._calculate_confidence(evidence_count, len(gaps), breath_rate)
        
        return breath_rate, confidence
    
    def _identify_gaps(self):
        """
        Find gaps where voice_activity == False.
        Minimum duration: > silence_min_duration (default 200ms, more lenient than 500ms).
        """
        gaps = []
        in_gap = False
        gap_start = 0
        
        for i, chunk_data in enumerate(self.history_buffer):
            features = chunk_data.get('features', {})
            has_voice = self._detect_voice_activity_f0(features)
            
            if not has_voice:  # Silence/gap
                if not in_gap:
                    gap_start = i
                    in_gap = True
            else:  # Voice detected
                if in_gap:
                    gap_end = i
                    gap_duration = gap_end - gap_start
                    # Use more permissive silence_min_chunks for detecting candidate gaps.
                    if gap_duration >= self.silence_min_chunks:
                        gaps.append((gap_start, gap_end))
                    in_gap = False
        
        # Handle case where buffer ends in gap
        if in_gap:
            gap_end = len(self.history_buffer)
            gap_duration = gap_end - gap_start
            if gap_duration >= self.silence_min_chunks:
                gaps.append((gap_start, gap_end))
        
        return gaps
    
    def _match_breath_fingerprint(self, gap_start, gap_end):
        """
        Pattern match: Confirm acoustic fingerprint within gap.
        Uses low-frequency "whoosh" detection (audSpec_Rfilt_sma3[0-2])
        and RMS energy (pcm_RMSenergy_sma).
        Per PDF: breath duration "typically 0.2 to 3 seconds".
        """
        gap_length = gap_end - gap_start
        if gap_length < self.gap_min_chunks or gap_length > self.gap_max_chunks:
            return False
        
        # Collect features from gap chunks
        gap_features = []
        for i in range(gap_start, min(gap_end, len(self.history_buffer))):
            chunk_data = self.history_buffer[i]
            features = chunk_data.get('features', {})
            gap_features.append(features)
        
        if not gap_features:
            return False
        
        # Check for low-frequency "whoosh" pattern
        # Breath sounds have significant energy in low-frequency bands
        low_freq_energy_sum = 0.0
        rms_energy_sum = 0.0
        
        for features in gap_features:
            # Sum low-frequency bands
            for band_idx in range(3):
                band_key = f'audSpec_Rfilt_sma3[{band_idx}]'
                if band_key in features:
                    low_freq_energy_sum += abs(features[band_key])
            
            # RMS energy (breath strength)
            if 'pcm_RMSenergy_sma' in features:
                rms_energy_sum += features['pcm_RMSenergy_sma']
        
        # Normalize by number of chunks
        num_chunks = len(gap_features)
        avg_low_freq = low_freq_energy_sum / num_chunks if num_chunks > 0 else 0.0
        avg_rms = rms_energy_sum / num_chunks if num_chunks > 0 else 0.0
        
        # Update noise floor adaptively
        if avg_rms > 0:
            self.background_noise_floor = (
                self.background_noise_floor * self.noise_adaptation_factor +
                avg_rms * (1 - self.noise_adaptation_factor)
            )
        
        # Pattern matching criteria:
        # 1. Low-frequency energy should be significant
        # 2. RMS energy should be above noise floor but not too high (breath is quiet)
        # 3. Multiple evidence points (sound + time + duration)
        
        # Use a slightly more permissive minimum and explicit breath_energy_threshold
        rms_min = max(self.background_noise_floor * self.rms_min_ratio, self.breath_energy_threshold)
        rms_max = self.rms_max
        
        # Check criteria
        has_low_freq = avg_low_freq > self.low_freq_threshold
        has_rms_match = rms_min < avg_rms < rms_max
        has_duration = self.gap_min_chunks <= (gap_end - gap_start) <= self.gap_max_chunks
        
        # Additional check: low-frequency should dominate
        # Breath sounds have more energy in low frequencies
        total_spectral = avg_low_freq + avg_rms
        low_freq_ratio = avg_low_freq / total_spectral if total_spectral > 0 else 0.0
        has_low_freq_dominance = low_freq_ratio > 0.3  # At least 30% low-freq
        
        # Require multiple evidence points
        evidence_score = sum([has_low_freq, has_rms_match, has_duration, has_low_freq_dominance])
        
        # Need at least 2-3 criteria to confirm breath
        return evidence_score >= 2
    
    def _calculate_confidence(self, evidence_count, gap_count, breath_rate):
        """
        Calculate confidence score (0.3-0.8) based on evidence quality.
        """
        if gap_count == 0:
            return self.CONFIDENCE_MIN
        
        # Base confidence from evidence ratio
        evidence_ratio = evidence_count / gap_count if gap_count > 0 else 0.0
        
        # Adjust based on breath rate validity
        rate_valid = self.BPM_MIN <= breath_rate <= self.BPM_MAX
        
        # Calculate confidence
        base_confidence = self.CONFIDENCE_MIN + (
            (self.CONFIDENCE_MAX - self.CONFIDENCE_MIN) * evidence_ratio
        )
        
        if not rate_valid and breath_rate > 0:
            base_confidence *= 0.7  # Reduce confidence for out-of-range rates
        
        # Clamp to range
        confidence = np.clip(base_confidence, self.CONFIDENCE_MIN, self.CONFIDENCE_MAX)
        
        return confidence
    
    def _semitone_to_hz(self, semitone_value):
        """
        Convert F0 semitone value to Hz.
        Formula: Hz = 27.5 * 2^(semitone/12)
        """
        if semitone_value is None or np.isnan(semitone_value) or semitone_value == 0:
            return 0.0
        
        # Convert semitones to Hz
        hz = 27.5 * (2 ** (semitone_value / 12.0))
        return float(hz)

    def _extract_f0_hz(self, features):
        """
        Extract fundamental frequency (F0) in Hz from OpenSMILE features.
        Returns f0_hz float (0.0 if no voice detected).
        """
        f0_key = 'F0semitoneFrom27.5Hz_sma3nz'
        f0_semitone = features.get(f0_key, 0.0)
        if f0_semitone == 0.0:
            return 0.0
        f0_hz = self._semitone_to_hz(f0_semitone)
        return round(f0_hz, 2) if f0_hz >= 50.0 else 0.0
    
    def _normalize_bpm(self, breath_rate):
        """
        Normalize BPM to 0.0-1.0 range as specified in PDF.
        Formula: normalized = (BreathRate - 8.0) / 32.0
        Clamped to [0.0, 1.0]
        """
        if breath_rate <= 0:
            return 0.0
        
        normalized = (breath_rate - self.BPM_MIN) / self.BPM_RANGE
        return np.clip(normalized, 0.0, 1.0)
    
    def _format_touchdesigner_output(self, breath_rate, confidence, f0_hz=0.0):
        """
        Format output for TouchDesigner per PDF specification.
        Emotion = BPM-based breath state (calm / slightly_elevated / anxious / high_anxiety).
        Target frequency per PDF: 396 Hz Fear, 639 Hz Love, 963 Hz Ascension.

        Args:
            breath_rate: Breath rate in BPM
            confidence: Confidence score (0.3-0.8)
            f0_hz: Raw fundamental frequency in Hz (supplementary)

        Returns:
            {
                'breath_rate_normalized': 0.0-1.0,
                'anxiety_level': 0.0 or 1.0,
                'calm_level': 0.0 or 1.0,
                'breath_state': 'calm' | 'slightly_elevated' | 'anxious' | 'high_anxiety',
                'confidence': 0.3-0.8,
                'target_frequency_hz': 396 | 639 | 963,
                'breath_rate_bpm': float,
                'emotion': { 'top_emotion': {...}, 'f0_hz': float, 'target_frequency_hz': int },
                'timestamp': unix timestamp
            }
        """
        normalized_bpm = self._normalize_bpm(breath_rate)

        # BPM bands per "The proposed solution":
        # Under 20 = Calm and relaxed; 20-25 = Slightly elevated;
        # 25-30 = Anxious or stressed; Over 30 = High anxiety
        if breath_rate <= 0:
            breath_state = "unknown"
        elif breath_rate < self.BPM_CALM_MAX:
            breath_state = "calm"
        elif breath_rate < self.BPM_ELEVATED_MAX:
            breath_state = "slightly_elevated"
        elif breath_rate < self.BPM_ANXIOUS_MAX:
            breath_state = "anxious"
        else:
            breath_state = "high_anxiety"
        anxiety_level = 1.0 if breath_rate > self.BPM_ANXIOUS_MAX else 0.0
        calm_level = 1.0 if breath_rate < self.BPM_CALM_MAX and breath_rate > 0 else 0.0

        # Target frequency from BPM state per PDF (396/639/963)
        target_hz = self.BREATH_STATE_TO_TARGET_HZ.get(breath_state, 639)

        output = {
            'breath_rate_normalized': float(normalized_bpm),
            'anxiety_level': float(anxiety_level),
            'calm_level': float(calm_level),
            'breath_state': breath_state,
            'confidence': float(confidence),
            'target_frequency_hz': target_hz,
            'timestamp': time.time(),
            'breath_rate_bpm': float(breath_rate),
            'emotion': {
                'top_emotion': {'label': breath_state, 'score': float(confidence)},
                'f0_hz': float(f0_hz) if f0_hz else 0.0,
                'target_frequency_hz': target_hz,
            },
        }

        return output
    
    # Legacy methods for backward compatibility
    def analyze_audio(self, audio_file_path):
        """Analyze entire audio file (legacy method)"""
        try:
            import librosa
            audio_signal, sr = librosa.load(audio_file_path, sr=self.sample_rate)
            return self.analyze_signal(audio_signal, audio_file_path=audio_file_path)
        except Exception as e:
            logger.error(f"Error loading audio file: {e}")
            return None
    
    def analyze_signal(self, audio_signal, audio_file_path=None):
        """
        Analyze entire audio signal by processing as chunks (legacy method).
        For real-time use, use process_chunk() instead.
        """
        try:
            audio_duration = len(audio_signal) / self.sample_rate
            
            if audio_file_path:
                logger.info(f"Analyzing audio file: {audio_file_path}, Duration: {audio_duration:.2f}s")
            else:
                logger.info(f"Analyzing audio signal, Duration: {audio_duration:.2f}s")
            
            # Process signal as chunks
            chunk_results = []
            num_chunks = len(audio_signal) // self.chunk_size
            
            for i in range(num_chunks):
                start_idx = i * self.chunk_size
                end_idx = start_idx + self.chunk_size
                chunk = audio_signal[start_idx:end_idx]
                
                result = self.process_chunk(chunk)
                chunk_results.append(result)
            
            # Aggregate results
            if chunk_results:
                # Use last result as final output
                final_result = chunk_results[-1]
                
                # Add additional analysis
                breath_rates = [r['breath_rate_normalized'] * self.BPM_RANGE + self.BPM_MIN 
                               for r in chunk_results if r['breath_rate_normalized'] > 0]
                
                avg_breath_rate = np.mean(breath_rates) if breath_rates else 0.0
                final_result['breath_rate'] = avg_breath_rate
                final_result['audio_duration'] = audio_duration
                
                logger.info(f"Breath analysis complete: {avg_breath_rate:.2f} bpm")
                return final_result
            else:
                return self._format_touchdesigner_output(0.0, self.CONFIDENCE_MIN, None)
                
        except Exception as e:
            logger.error(f"Error in audio analysis: {e}")
            import traceback
            traceback.print_exc()
            return None
