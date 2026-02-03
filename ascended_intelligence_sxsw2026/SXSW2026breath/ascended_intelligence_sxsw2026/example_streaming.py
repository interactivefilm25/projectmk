#!/usr/bin/env python3
"""
Example: Real-time breath detection with chunk-based processing
Demonstrates the PDF specification implementation
"""

import sys
import os
import numpy as np
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ascended.breath_detector import BreathDetector

def simulate_audio_stream(detector, duration_seconds=5.0):
    """
    Simulate an audio stream by generating chunks in real-time.
    In production, this would be replaced with actual microphone input.
    """
    print("="*60)
    print("Simulating Real-Time Audio Stream")
    print("="*60)
    print(f"Duration: {duration_seconds}s")
    print(f"Chunk size: {detector.chunk_size} samples (100ms)")
    print(f"Processing rate: {detector.CHUNKS_PER_SECOND} chunks/second\n")
    
    num_chunks = int(duration_seconds * detector.CHUNKS_PER_SECOND)
    
    print("Processing chunks...")
    print("-" * 60)
    
    results = []
    start_time = time.time()
    
    for i in range(num_chunks):
        # Simulate audio chunk (in production, get from microphone)
        chunk = np.random.randn(detector.chunk_size).astype(np.float32) * 0.1
        
        # Add some periodic variation to simulate breathing
        t = np.linspace(0, 0.1, detector.chunk_size)
        breathing_freq = 0.25  # 15 breaths per minute
        chunk += 0.02 * np.sin(2 * np.pi * breathing_freq * t)
        
        # Process chunk
        result = detector.process_chunk(chunk)
        results.append(result)
        
        # Display every 10 chunks (1 second)
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"[{elapsed:.1f}s] Chunk {i+1:3d}/{num_chunks}")
            print(f"  breath_rate_normalized: {result['breath_rate_normalized']:.4f}")
            print(f"  anxiety_level: {result['anxiety_level']:.1f}")
            print(f"  calm_level: {result['calm_level']:.1f}")
            print(f"  confidence: {result['confidence']:.4f}")
            print()
        
        # Simulate real-time delay (100ms between chunks)
        time.sleep(0.1)
    
    elapsed = time.time() - start_time
    print("-" * 60)
    print(f"Processing complete: {elapsed:.2f}s")
    print(f"Average processing time per chunk: {elapsed/num_chunks*1000:.2f}ms")
    
    return results

def main():
    """Main example"""
    print("="*60)
    print("ASCENDED Breath Detection - Real-Time Example")
    print("="*60)
    print()
    
    # Initialize detector
    print("Initializing BreathDetector...")
    detector = BreathDetector(sample_rate=16000, use_opensmile=True)
    print("Detector initialized\n")
    
    # Simulate streaming
    results = simulate_audio_stream(detector, duration_seconds=5.0)
    
    # Summary
    print("="*60)
    print("Summary")
    print("="*60)
    
    if results:
        # Calculate statistics
        normalized_rates = [r['breath_rate_normalized'] for r in results]
        confidences = [r['confidence'] for r in results]
        
        # Convert normalized back to BPM for display
        bpm_values = [n * detector.BPM_RANGE + detector.BPM_MIN 
                     for n in normalized_rates if n > 0]
        
        if bpm_values:
            avg_bpm = np.mean(bpm_values)
            print(f"Average breath rate: {avg_bpm:.2f} BPM")
            print(f"Average confidence: {np.mean(confidences):.4f}")
            
            # Count anxiety/calm periods
            anxiety_count = sum(1 for r in results if r['anxiety_level'] == 1.0)
            calm_count = sum(1 for r in results if r['calm_level'] == 1.0)
            
            print(f"Anxiety periods: {anxiety_count} chunks")
            print(f"Calm periods: {calm_count} chunks")
        
        print(f"\nTotal chunks processed: {len(results)}")
        print(f"History buffer size: {len(detector.history_buffer)}/{detector.history_buffer.maxlen}")
    
    print("\nExample complete.")

if __name__ == "__main__":
    main()
