"""
Compare Whisper vs Qwen-Audio Transcription

This script processes the same video with both transcribers and shows:
- Processing time comparison
- Output quality comparison
- Accuracy metrics
- Feature comparison

Usage:
    python compare_transcribers.py "video.mp4"
"""

import time
from pathlib import Path
from typing import Dict
import json


def compare_transcribers(video_path: str):
    """Compare Whisper and Qwen-Audio on the same video."""
    
    video_path = Path(video_path)
    
    if not video_path.exists():
        print(f"Error: File not found: {video_path}")
        return
    
    print("=" * 70)
    print("TRANSCRIBER COMPARISON")
    print("=" * 70)
    print(f"Video: {video_path.name}\n")
    
    results = {}
    
    # Test 1: Whisper
    print("\n" + "=" * 70)
    print("TEST 1: WHISPER LARGE")
    print("=" * 70)
    
    try:
        from transcriber import SimpleTranscriber
        
        whisper = SimpleTranscriber(model_size="large", device="auto")
        
        start_time = time.time()
        whisper_result = whisper.transcribe_video(str(video_path))
        whisper_time = time.time() - start_time
        
        results['whisper'] = {
            'result': whisper_result,
            'time': whisper_time,
            'model': 'Whisper Large',
            'output_dir': Path("processed/transcripts/Whisper-Large") / video_path.stem
        }
        
        print(f"\n✓ Whisper completed in {whisper_time:.1f}s")
        
    except Exception as e:
        print(f"\n✗ Whisper failed: {e}")
        results['whisper'] = None
    
    # Test 2: Qwen-Audio
    print("\n" + "=" * 70)
    print("TEST 2: QWEN-AUDIO")
    print("=" * 70)
    
    try:
        from qwen_asr_transcriber import QwenTranscriber
        
        qwen = QwenTranscriber(model_name="qwen2-audio", device="auto")
        
        start_time = time.time()
        qwen_result = qwen.transcribe_video(str(video_path))
        qwen_time = time.time() - start_time
        
        results['qwen'] = {
            'result': qwen_result,
            'time': qwen_time,
            'model': 'Qwen2-Audio',
            'output_dir': Path("processed/transcripts/Qwen-qwen2-audio") / video_path.stem
        }
        
        print(f"\n✓ Qwen-Audio completed in {qwen_time:.1f}s")
        
    except Exception as e:
        print(f"\n✗ Qwen-Audio failed: {e}")
        results['qwen'] = None
    
    # Comparison Report
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    
    if results['whisper'] and results['qwen']:
        # Speed comparison
        speedup = results['whisper']['time'] / results['qwen']['time']
        
        print(f"\n📊 PROCESSING TIME:")
        print(f"  Whisper:     {results['whisper']['time']:.1f}s")
        print(f"  Qwen-Audio:  {results['qwen']['time']:.1f}s")
        print(f"  Speedup:     {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")
        
        # Text length comparison
        whisper_text = results['whisper']['result'].get('text', '')
        qwen_text = results['qwen']['result'].get('text', '')
        
        print(f"\n📝 OUTPUT LENGTH:")
        print(f"  Whisper:     {len(whisper_text)} characters")
        print(f"  Qwen-Audio:  {len(qwen_text)} characters")
        
        # Segment count
        whisper_segs = len(results['whisper']['result'].get('segments', []))
        qwen_segs = len(results['qwen']['result'].get('segments', []))
        
        print(f"\n🎯 SEGMENTS:")
        print(f"  Whisper:     {whisper_segs} segments")
        print(f"  Qwen-Audio:  {qwen_segs} segments")
        
        # Output locations
        print(f"\n📁 OUTPUT LOCATIONS:")
        print(f"  Whisper:")
        print(f"    {results['whisper']['output_dir'] / 'transcript.txt'}")
        print(f"  Qwen-Audio:")
        print(f"    {results['qwen']['output_dir'] / 'transcript.txt'}")
        
        # Sample comparison
        print(f"\n📄 TEXT PREVIEW (first 200 chars):")
        print(f"\n  Whisper:")
        print(f"  {whisper_text[:200]}...")
        print(f"\n  Qwen-Audio:")
        print(f"  {qwen_text[:200]}...")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if speedup > 1.5:
            print(f"  ✓ Qwen-Audio is {speedup:.1f}x faster - good for batch processing")
        
        if qwen_segs > whisper_segs:
            print(f"  ✓ Qwen-Audio has more detailed segmentation")
        
        # Save comparison report
        report_path = Path("processed") / f"comparison_{video_path.stem}.json"
        report = {
            'video': str(video_path),
            'whisper': {
                'time': results['whisper']['time'],
                'text_length': len(whisper_text),
                'segments': whisper_segs,
                'model': results['whisper']['model']
            },
            'qwen': {
                'time': results['qwen']['time'],
                'text_length': len(qwen_text),
                'segments': qwen_segs,
                'model': results['qwen']['model']
            },
            'speedup': speedup
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 Comparison report saved: {report_path}")
        
    else:
        if not results['whisper']:
            print("\n✗ Whisper test failed")
        if not results['qwen']:
            print("\n✗ Qwen-Audio test failed")
    
    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python compare_transcribers.py <video_file>")
        print("\nExample:")
        print("  python compare_transcribers.py 'videos/AkerBP 1.mp4'")
        sys.exit(1)
    
    video_path = sys.argv[1]
    compare_transcribers(video_path)
