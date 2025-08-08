#!/usr/bin/env python3
"""
Integration Patterns for Speaker Recognition

Common patterns for integrating speaker recognition into existing pipelines.
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from speaker_recognition import initialize_speaker_recognition, identify_speaker

class SpeakerDatabase:
    """Manage speaker database with custom metadata"""
    
    def __init__(self, db_path: str = "speakers.json"):
        self.db_path = db_path
        self.speakers = self.load_database()
    
    def load_database(self) -> Dict:
        """Load speaker metadata from JSON file"""
        if os.path.exists(self.db_path):
            with open(self.db_path, 'r') as f:
                return json.load(f)
        return {}
    
    def save_database(self):
        """Save speaker metadata to JSON file"""
        with open(self.db_path, 'w') as f:
            json.dump(self.speakers, f, indent=2)
    
    def add_speaker_metadata(self, speaker_id: str, name: str = None, notes: str = None):
        """Add custom metadata for a speaker"""
        if speaker_id not in self.speakers:
            self.speakers[speaker_id] = {}
        
        if name:
            self.speakers[speaker_id]['name'] = name
        if notes:
            self.speakers[speaker_id]['notes'] = notes
        
        self.speakers[speaker_id]['interactions'] = self.speakers[speaker_id].get('interactions', 0) + 1
        self.save_database()
    
    def get_speaker_info(self, speaker_id: str) -> Dict:
        """Get all info about a speaker"""
        return self.speakers.get(speaker_id, {})

def batch_process_audio_files(audio_files: List[str]) -> List[Dict]:
    """
    Pattern 1: Batch Processing
    Process multiple audio files and return results
    """
    print("🔄 Batch processing audio files...")
    
    if not initialize_speaker_recognition():
        raise RuntimeError("Failed to initialize speaker recognition")
    
    results = []
    for audio_file in audio_files:
        if not os.path.exists(audio_file):
            continue
            
        speaker_id, confidence = identify_speaker(audio_file)
        results.append({
            'file': audio_file,
            'speaker_id': speaker_id,
            'confidence': confidence,
            'timestamp': None  # Add your timestamp logic
        })
    
    return results

def streaming_audio_handler(audio_chunk_path: str, threshold: float = 0.52) -> Optional[str]:
    """
    Pattern 2: Streaming/Real-time Processing
    Handle individual audio chunks from a stream
    """
    try:
        speaker_id, confidence = identify_speaker(audio_chunk_path, threshold=threshold)
        
        # Only return speaker if confidence is high enough
        if speaker_id and confidence <= threshold:
            return speaker_id
        
        return None
    except Exception as e:
        print(f"Error processing audio chunk: {e}")
        return None

def pipeline_integration(audio_file: str, existing_pipeline_data: Dict) -> Dict:
    """
    Pattern 3: Pipeline Integration
    Add speaker recognition to existing audio processing pipeline
    """
    # Your existing pipeline results
    pipeline_results = existing_pipeline_data.copy()
    
    # Add speaker recognition
    try:
        speaker_id, confidence = identify_speaker(audio_file)
        
        pipeline_results.update({
            'speaker_recognition': {
                'speaker_id': speaker_id,
                'confidence': confidence,
                'status': 'recognized' if speaker_id else 'new_speaker'
            }
        })
    except Exception as e:
        pipeline_results['speaker_recognition'] = {
            'error': str(e),
            'status': 'failed'
        }
    
    return pipeline_results

def conversation_tracker():
    """
    Pattern 4: Conversation/Session Tracking
    Track speakers across a conversation or session
    """
    class ConversationSession:
        def __init__(self):
            self.speakers = {}
            self.timeline = []
            self.db = SpeakerDatabase()
        
        def process_utterance(self, audio_file: str, timestamp: float = None):
            """Process a single utterance in the conversation"""
            speaker_id, confidence = identify_speaker(audio_file)
            
            # Track speaker in session
            if speaker_id:
                if speaker_id not in self.speakers:
                    self.speakers[speaker_id] = {
                        'first_seen': timestamp,
                        'utterance_count': 0,
                        'total_confidence': 0.0
                    }
                
                self.speakers[speaker_id]['utterance_count'] += 1
                self.speakers[speaker_id]['total_confidence'] += confidence
                self.speakers[speaker_id]['avg_confidence'] = (
                    self.speakers[speaker_id]['total_confidence'] / 
                    self.speakers[speaker_id]['utterance_count']
                )
            
            # Add to timeline
            self.timeline.append({
                'timestamp': timestamp,
                'speaker_id': speaker_id,
                'confidence': confidence,
                'audio_file': audio_file
            })
            
            # Update database
            if speaker_id:
                self.db.add_speaker_metadata(speaker_id)
            
            return speaker_id, confidence
        
        def get_session_summary(self):
            """Get summary of the conversation session"""
            return {
                'unique_speakers': len(self.speakers),
                'total_utterances': len(self.timeline),
                'speakers': self.speakers,
                'timeline': self.timeline
            }
    
    return ConversationSession()

def custom_speaker_ids(audio_file: str, custom_format: callable = None) -> Tuple[str, float]:
    """
    Pattern 5: Custom Speaker ID Format
    Use custom speaker ID formats instead of default hashtags
    """
    speaker_id, confidence = identify_speaker(audio_file)
    
    if speaker_id and custom_format:
        # Convert default #abcd format to custom format
        # Example: #abcd -> SPEAKER_001, USER_abcd, etc.
        custom_id = custom_format(speaker_id)
        return custom_id, confidence
    
    return speaker_id, confidence

# Example custom ID formatters
def corporate_id_format(speaker_id: str) -> str:
    """Convert #abcd to SPEAKER_001 format"""
    # You'd maintain your own mapping
    speaker_num = hash(speaker_id) % 1000
    return f"SPEAKER_{speaker_num:03d}"

def user_id_format(speaker_id: str) -> str:
    """Convert #abcd to USER_abcd format"""
    return f"USER_{speaker_id[1:]}" if speaker_id.startswith('#') else f"USER_{speaker_id}"

def main():
    """Demonstrate integration patterns"""
    print("🔧 Speaker Recognition - Integration Patterns")
    print("=" * 50)
    
    # Example usage of different patterns
    audio_files = ["example1.wav", "example2.wav"]  # Replace with real files
    
    # Pattern 1: Batch processing
    try:
        results = batch_process_audio_files(audio_files)
        print(f"📊 Batch results: {len(results)} files processed")
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
    
    # Pattern 2: Streaming
    print("\n🌊 Streaming pattern example:")
    for audio_file in audio_files:
        if os.path.exists(audio_file):
            speaker = streaming_audio_handler(audio_file)
            print(f"   {audio_file}: {speaker or 'No speaker detected'}")
    
    # Pattern 3: Pipeline integration
    print("\n🔄 Pipeline integration example:")
    existing_data = {"transcription": "Hello world", "language": "en"}
    enhanced_data = pipeline_integration("example1.wav", existing_data)
    print(f"   Enhanced data keys: {list(enhanced_data.keys())}")
    
    # Pattern 4: Conversation tracking
    print("\n💬 Conversation tracking example:")
    session = conversation_tracker()
    # session.process_utterance("utterance1.wav", timestamp=1.0)
    # session.process_utterance("utterance2.wav", timestamp=2.5)
    summary = session.get_session_summary()
    print(f"   Session: {summary['unique_speakers']} speakers, {summary['total_utterances']} utterances")
    
    # Pattern 5: Custom ID formats
    print("\n🏷️  Custom ID format examples:")
    if os.path.exists("example1.wav"):
        corporate_id, conf = custom_speaker_ids("example1.wav", corporate_id_format)
        user_id, _ = custom_speaker_ids("example1.wav", user_id_format)
        print(f"   Corporate format: {corporate_id}")
        print(f"   User format: {user_id}")

if __name__ == "__main__":
    main()