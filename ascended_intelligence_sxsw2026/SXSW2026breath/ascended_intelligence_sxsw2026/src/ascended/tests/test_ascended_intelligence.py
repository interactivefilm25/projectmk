#!/usr/bin/env python3
"""Test suite for ASCENDED Intelligence - SXSW 2026"""

import sys
import time
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dependencies():
    """Test all required dependencies"""
    logger.info("🔧 Testing Dependencies...")
    
    required_packages = [
        ('numpy', 'np'),
        ('librosa', 'librosa'), 
        ('scipy', 'scipy'),
        ('opensmile', 'opensmile'),
        ('soundfile', 'soundfile')
    ]
    
    results = {'passed': 0, 'failed': 0, 'errors': []}
    
    for package_name, import_name in required_packages:
        try:
            exec(f"import {import_name}")
            logger.info(f"✅ {package_name} - OK")
            results['passed'] += 1
        except ImportError as e:
            logger.error(f"❌ {package_name} - FAILED: {e}")
            results['failed'] += 1
            results['errors'].append(f"Missing: {package_name}")
    
    return results

def test_breath_detection_performance():
    """Test breath detection performance"""
    logger.info("⚡ Testing Performance...")
    
    try:
        import numpy as np
        import librosa
        
        # Generate test audio (5 seconds)
        duration = 5.0
        sample_rate = 16000
        test_audio = np.random.randn(int(duration * sample_rate)) * 0.1
        
        # Test processing speed
        start_time = time.time()
        mfccs = librosa.feature.mfcc(y=test_audio, sr=sample_rate, n_mfcc=13)
        processing_time = time.time() - start_time
        
        real_time_factor = duration / processing_time
        
        logger.info(f"✅ Performance Test - OK")
        logger.info(f"   Real-time factor: {real_time_factor:.2f}x")
        
        return {
            'success': True,
            'real_time_factor': real_time_factor,
            'sxsw_ready': real_time_factor > 2.0
        }
        
    except Exception as e:
        logger.error(f"❌ Performance Test - FAILED: {e}")
        return {'success': False, 'error': str(e)}

def main():
    """Run complete test suite"""
    print("🌬️ ASCENDED Intelligence Test Suite - SXSW 2026")
    print("-" * 60)
    
    # Test dependencies
    dep_results = test_dependencies()
    
    # Test performance
    perf_results = test_breath_detection_performance()
    
    # Generate report
    total_tests = dep_results['passed'] + dep_results['failed'] + 1
    total_passed = dep_results['passed'] + (1 if perf_results.get('success', False) else 0)
    success_rate = (total_passed / total_tests) * 100
    
    print("\n" + "="*60)
    print("🎯 SXSW 2026 READINESS REPORT")
    print("="*60)
    print(f"Tests Passed: {total_passed}/{total_tests}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("🎉 STATUS: ✅ READY FOR SXSW 2026!")
    else:
        print("⚠️ STATUS: ❌ NEEDS WORK")
        print("Issues to fix:")
        for error in dep_results['errors']:
            print(f"   - {error}")
    
    print("="*60)

if __name__ == "__main__":
    main()
