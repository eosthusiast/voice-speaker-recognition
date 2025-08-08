#!/usr/bin/env python3
"""
Speaker Recognition Only - Minimal Example

This example shows how to use just the speaker recognition functionality
without the voice assistant, transcription, or AI components.
"""

import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from speaker_recognition import (
    initialize_speaker_recognition, 
    identify_speaker,
    get_session_stats
)

def main():
    """Minimal speaker recognition example"""
    
    print("🎙️  Speaker Recognition - Minimal Example")
    print("=" * 45)
    
    # Check for HuggingFace token
    if not os.getenv('HF_TOKEN'):
        print("❌ Error: HF_TOKEN not found in environment")
        print("Please set your HuggingFace token:")
        print("  export HF_TOKEN='hf_your_token_here'")
        print("\nGet token from: https://huggingface.co/settings/tokens")
        return
    
    # Initialize speaker recognition system
    print("🔄 Initializing speaker recognition...")
    if not initialize_speaker_recognition():
        print("❌ Failed to initialize speaker recognition")
        return
    
    print("✅ Speaker recognition ready!")
    
    # Example audio files (replace with your own)
    audio_files = [
        "example1.wav",  # Replace with actual audio files
        "example2.wav",
        "example1.wav",  # Same as first - should match
    ]
    
    print(f"\n🎯 Processing {len(audio_files)} audio files...")
    
    results = []
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n--- Audio File {i}: {audio_file} ---")
        
        # Check if file exists (in real usage)
        if not os.path.exists(audio_file):
            print(f"⚠️  File not found: {audio_file} (skipping)")
            print("   In real usage, provide actual audio files")
            continue
        
        # Identify speaker
        try:
            speaker_id, confidence = identify_speaker(audio_file)
            
            if speaker_id:
                print(f"✅ Recognized: {speaker_id}")
                print(f"   Confidence: {confidence:.3f}")
                status = "MATCH" if confidence < 0.6 else "STRONG MATCH"
                print(f"   Status: {status}")
            else:
                print("🆕 New speaker detected")
                print("   Will be assigned ID on future interactions")
                status = "NEW"
            
            results.append({
                'file': audio_file,
                'speaker': speaker_id,
                'confidence': confidence,
                'status': status
            })
            
        except Exception as e:
            print(f"❌ Error processing {audio_file}: {e}")
    
    # Show session statistics
    print(f"\n📊 Session Statistics:")
    stats = get_session_stats()
    print(f"   Total speakers: {stats.get('total_speakers', 0)}")
    print(f"   Total interactions: {stats.get('total_interactions', 0)}")
    
    # Summary
    if results:
        print(f"\n📋 Summary:")
        for result in results:
            status_icon = "✅" if result['status'] != "NEW" else "🆕"
            confidence_str = f" ({result['confidence']:.3f})" if result['speaker'] else ""
            print(f"   {status_icon} {result['file']}: {result['speaker'] or 'NEW_SPEAKER'}{confidence_str}")
    
    print(f"\n🎉 Done! Speaker recognition data saved to:")
    print(f"   - speaker_embeddings.npy (voice fingerprints)")
    print(f"   - speaker_hashtags.txt (speaker database)")

if __name__ == "__main__":
    main()