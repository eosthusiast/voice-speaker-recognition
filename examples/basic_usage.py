#!/usr/bin/env python3
"""
Basic Usage Example - Voice Assistant with Speaker Recognition

This example demonstrates how to use the voice assistant programmatically
without the PTT interface. Useful for integration into other applications.
"""

import os
import sys
import tempfile
import wave

# Add parent directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice_assistant import VoiceAssistant
from speaker_recognition import initialize_speaker_recognition, identify_speaker, get_session_stats
from colors import Colors

def create_sample_audio():
    """Create a simple audio file for testing (silent audio)
    In a real application, you would record from microphone or load existing files.
    """
    duration = 2  # seconds
    sample_rate = 44100
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        with wave.open(tmp_file.name, 'w') as wav_file:
            wav_file.setnchannels(1)  # mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            # Write silent audio (zeros)
            silent_frames = b'\x00\x00' * (sample_rate * duration)
            wav_file.writeframes(silent_frames)
        
        return tmp_file.name

def example_basic_usage():
    """Basic voice assistant usage example"""
    print(f"{Colors.CYAN}=== Basic Voice Assistant Usage ==={Colors.RESET}")
    
    # Check for required API key
    if not os.getenv('ANTHROPIC_API_KEY'):
        print(f"{Colors.RED}Error: ANTHROPIC_API_KEY not found in environment{Colors.RESET}")
        print("Please set your API key:")
        print("  export ANTHROPIC_API_KEY='your-key-here'")
        return
    
    # Initialize the voice assistant
    print("Initializing voice assistant...")
    assistant = VoiceAssistant(
        use_speaker_recognition=True,
        use_honcho=True  # Enable memory if Honcho is available
    )
    
    # Simulate a conversation
    user_messages = [
        "Hello, I'm testing the voice assistant",
        "What's the weather like today?",
        "Can you remember what I said earlier?"
    ]
    
    for i, message in enumerate(user_messages):
        print(f"\n--- Turn {i+1} ---")
        print(f"{Colors.BLUE}User: {message}{Colors.RESET}")
        
        # In a real app, you would have actual audio here
        # For this example, we'll just use the text directly
        fake_speaker_id = "#test_user"  # Simulate a speaker ID
        
        # Get AI response
        response = assistant.get_ai_response(message, fake_speaker_id)
        print(f"{Colors.GREEN}Assistant: {response}{Colors.RESET}")
    
    # Show conversation history
    print(f"\n{Colors.YELLOW}Conversation History:{Colors.RESET}")
    for j, msg in enumerate(assistant.conversation_history[-4:]):  # Show last 4 messages
        role_color = Colors.BLUE if msg['role'] == 'user' else Colors.GREEN
        print(f"{role_color}{msg['role'].title()}: {msg['content'][:100]}...{Colors.RESET}")
    
    assistant.cleanup()

def example_speaker_recognition():
    """Speaker recognition specific example"""
    print(f"\n{Colors.CYAN}=== Speaker Recognition Example ==={Colors.RESET}")
    
    # Initialize speaker recognition
    if not initialize_speaker_recognition():
        print(f"{Colors.RED}Speaker recognition failed to initialize{Colors.RESET}")
        return
    
    print("Speaker recognition initialized successfully!")
    
    # Create sample audio file (in practice, use real recordings)
    audio_file = create_sample_audio()
    print(f"Created sample audio file: {audio_file}")
    
    try:
        # Attempt speaker identification
        print("Identifying speaker...")
        speaker_id, confidence = identify_speaker(audio_file)
        
        if speaker_id:
            print(f"{Colors.GREEN}Speaker identified: {speaker_id} (confidence: {confidence:.3f}){Colors.RESET}")
        else:
            print(f"{Colors.YELLOW}No speaker match found - this would create a new speaker profile{Colors.RESET}")
        
        # Show session statistics
        stats = get_session_stats()
        print(f"\nSession Statistics:")
        print(f"  Total speakers: {stats.get('total_speakers', 0)}")
        print(f"  Total interactions: {stats.get('total_interactions', 0)}")
        
    finally:
        # Clean up
        if os.path.exists(audio_file):
            os.unlink(audio_file)

def example_memory_integration():
    """Example showing memory/context features"""
    print(f"\n{Colors.CYAN}=== Memory Integration Example ==={Colors.RESET}")
    
    # Check if Honcho is available
    try:
        from honcho import Honcho
        honcho_available = bool(os.getenv('HONCHO_API_KEY'))
    except ImportError:
        honcho_available = False
    
    if not honcho_available:
        print(f"{Colors.YELLOW}Honcho not available - memory features disabled{Colors.RESET}")
        print("To enable memory:")
        print("  1. Install: pip install honcho-ai")
        print("  2. Set HONCHO_API_KEY in environment")
        return
    
    print("Testing memory integration...")
    
    assistant = VoiceAssistant(use_honcho=True)
    
    # Simulate storing and retrieving memory
    speaker_id = "#memory_test_user"
    
    # First interaction - establish some context
    response1 = assistant.get_ai_response("My name is Alice and I love Python programming", speaker_id)
    print(f"{Colors.BLUE}User: My name is Alice and I love Python programming{Colors.RESET}")
    print(f"{Colors.GREEN}Assistant: {response1}{Colors.RESET}")
    
    # Second interaction - test memory recall
    response2 = assistant.get_ai_response("What's my name and what do I like?", speaker_id)
    print(f"\n{Colors.BLUE}User: What's my name and what do I like?{Colors.RESET}")
    print(f"{Colors.GREEN}Assistant: {response2}{Colors.RESET}")
    
    assistant.cleanup()

def main():
    """Run all examples"""
    print(f"{Colors.BOLD_CYAN}Voice Assistant with Speaker Recognition - Examples{Colors.RESET}")
    print("=" * 60)
    
    try:
        # Run examples
        example_basic_usage()
        example_speaker_recognition() 
        example_memory_integration()
        
        print(f"\n{Colors.GREEN}All examples completed!{Colors.RESET}")
        print(f"\nTo use the full voice assistant with push-to-talk:")
        print(f"  python voice_assistant.py")
        
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Examples interrupted by user{Colors.RESET}")
    except Exception as e:
        print(f"\n{Colors.RED}Error running examples: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()