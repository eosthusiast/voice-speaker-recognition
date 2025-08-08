#!/usr/bin/env python3
"""
Voice Assistant with Speaker Recognition
A simplified voice assistant using MLX Whisper, speaker recognition, and Anthropic chat.

Features:
- Push-to-talk (Option+Space) for voice input
- MLX Whisper for fast transcription
- Speaker recognition with voice fingerprinting
- Basic Anthropic chat integration
- Optional Honcho memory integration
"""

import os
import json
import wave
import tempfile
import pyaudio
import numpy as np
import anthropic
import threading
import time
from datetime import datetime
from typing import Optional, List, Dict, Tuple
import mlx_whisper
from pynput import keyboard
from pynput.keyboard import Key

# Optional Honcho import
try:
    from honcho import Honcho
    HONCHO_AVAILABLE = True
except ImportError:
    HONCHO_AVAILABLE = False
    print("Honcho not available - memory features disabled")

# Local imports
from colors import Colors
from speaker_recognition import (
    initialize_speaker_recognition, 
    identify_speaker, 
    get_speaker_context,
    DEFAULT_SIMILARITY_THRESHOLD
)

# Audio recording configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

# MLX-Whisper configuration
WHISPER_MODEL_NAME = "mlx-community/whisper-large-v3-turbo"

# Initialize Anthropic client
anthropic_client = None
if os.getenv('ANTHROPIC_API_KEY'):
    anthropic_client = anthropic.Client(api_key=os.getenv('ANTHROPIC_API_KEY'))
else:
    print(f"{Colors.YELLOW}Warning: ANTHROPIC_API_KEY not set{Colors.RESET}")

# Initialize Honcho client (optional)
honcho_client = None
if HONCHO_AVAILABLE and os.getenv('HONCHO_API_KEY'):
    try:
        honcho_client = Honcho(
            api_key=os.getenv('HONCHO_API_KEY'),
            base_url="https://api.honcho.dev",
            workspace_id=os.getenv('HONCHO_WORKSPACE', 'default')
        )
        print(f"{Colors.GREEN}Honcho memory system initialized{Colors.RESET}")
    except Exception as e:
        print(f"{Colors.YELLOW}Honcho initialization failed: {e}{Colors.RESET}")

class VoiceAssistant:
    """Main voice assistant class"""
    
    def __init__(self, use_speaker_recognition=True, use_honcho=True):
        """Initialize the voice assistant
        
        Args:
            use_speaker_recognition: Enable speaker recognition features
            use_honcho: Enable Honcho memory features (if available)
        """
        self.is_listening = False
        self.is_recording = False
        self.audio_stream = None
        self.audio_chunks = []
        self.p = None
        
        # Configuration
        self.speaker_recognition_enabled = False
        self.honcho_enabled = use_honcho and honcho_client is not None
        
        # Initialize audio system
        self.initialize_audio()
        
        # Initialize speaker recognition
        if use_speaker_recognition:
            self.speaker_recognition_enabled = initialize_speaker_recognition()
            if self.speaker_recognition_enabled:
                print(f"{Colors.GREEN}Speaker recognition ready!{Colors.RESET}")
            else:
                print(f"{Colors.YELLOW}Speaker recognition unavailable{Colors.RESET}")
        
        # Track conversation
        self.conversation_history = []
        self.current_speaker_id = None
        
    def initialize_audio(self):
        """Initialize PyAudio"""
        try:
            if self.p is None:
                self.p = pyaudio.PyAudio()
            if self.audio_stream is None:
                self.audio_stream = self.p.open(
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    start=False
                )
            print("Audio system initialized")
        except Exception as e:
            print(f"Error initializing audio: {e}")
    
    def start_recording(self):
        """Start recording audio"""
        if self.is_listening:
            return
        
        self.is_listening = True
        self.is_recording = True
        self.audio_chunks = []
        
        # Start recording thread
        thread = threading.Thread(target=self._recording_thread, daemon=True)
        thread.start()
        
        print(f"{Colors.GREEN}Recording started (release key to process)...{Colors.RESET}")
    
    def stop_recording(self):
        """Stop recording and process audio"""
        self.is_listening = False
        time.sleep(0.05)  # Give recording thread time to finish
        
        if self.is_recording and self.audio_chunks:
            self.process_recorded_audio()
        
        self.is_recording = False
    
    def _recording_thread(self):
        """Background thread for audio recording"""
        try:
            if not self.audio_stream.is_active():
                self.audio_stream.start_stream()
            
            # Clear buffer
            for _ in range(3):
                try:
                    self.audio_stream.read(CHUNK, exception_on_overflow=False)
                except:
                    pass
            
            while self.is_listening:
                try:
                    audio_data = self.audio_stream.read(CHUNK, exception_on_overflow=False)
                    self.audio_chunks.append(audio_data)
                except Exception as e:
                    print(f"Error in recording: {e}")
                    break
                    
        finally:
            if self.audio_stream and self.audio_stream.is_active():
                self.audio_stream.stop_stream()
    
    def process_recorded_audio(self):
        """Process the recorded audio: transcribe and identify speaker"""
        if not self.audio_chunks:
            return None, None
        
        try:
            # Combine audio chunks
            audio_data = b''.join(self.audio_chunks)
            
            # Create temporary wav file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
                with wave.open(temp_audio.name, 'wb') as wav_file:
                    wav_file.setnchannels(CHANNELS)
                    wav_file.setsampwidth(self.p.get_sample_size(FORMAT))
                    wav_file.setframerate(RATE)
                    wav_file.writeframes(audio_data)
                
                # Transcribe with MLX-Whisper
                transcribed_text = self.transcribe_audio(temp_audio.name)
                
                # Speaker recognition
                speaker_id = None
                speaker_context = None
                if self.speaker_recognition_enabled and transcribed_text:
                    speaker_id, confidence = identify_speaker(
                        temp_audio.name, 
                        threshold=DEFAULT_SIMILARITY_THRESHOLD
                    )
                    if speaker_id:
                        speaker_context = get_speaker_context(speaker_id)
                        print(f"{Colors.CYAN}Speaker: {speaker_context} (confidence: {confidence:.3f}){Colors.RESET}")
                        self.current_speaker_id = speaker_id
                
                # Clean up temp file
                os.unlink(temp_audio.name)
                
                return transcribed_text, speaker_id
                
        except Exception as e:
            print(f"Error processing audio: {e}")
            return None, None
    
    def transcribe_audio(self, audio_path):
        """Transcribe audio using MLX-Whisper
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text or None
        """
        try:
            print("Transcribing with MLX-Whisper...")
            
            # Two-pass approach: detect language first, then transcribe
            # First pass - detect language
            result = mlx_whisper.transcribe(
                audio_path,
                path_or_hf_repo=WHISPER_MODEL_NAME,
                task="transcribe",
                language=None,  # Auto-detect
                verbose=False,
                max_initial_timestamp=0.5
            )
            detected_language = result.get("language", "unknown")
            
            # Second pass - transcribe with explicit language
            if detected_language != "unknown":
                result = mlx_whisper.transcribe(
                    audio_path,
                    path_or_hf_repo=WHISPER_MODEL_NAME,
                    task="transcribe",
                    language=detected_language,
                    verbose=False
                )
            
            transcribed_text = result["text"].strip()
            
            if transcribed_text:
                print(f"Language: {detected_language}")
                print(f"Transcribed: '{transcribed_text}'")
                
            return transcribed_text
            
        except Exception as e:
            print(f"{Colors.RED}Transcription error: {e}{Colors.RESET}")
            return None
    
    def get_ai_response(self, user_message, speaker_id=None):
        """Get response from Anthropic Claude
        
        Args:
            user_message: The user's message
            speaker_id: Optional speaker ID for context
            
        Returns:
            AI response text
        """
        if not anthropic_client:
            return "Anthropic API key not configured"
        
        try:
            # Get Honcho context if available
            context_messages = []
            if self.honcho_enabled and speaker_id:
                context_messages = self.get_honcho_context(speaker_id, user_message)
            
            # Add conversation history
            messages = context_messages + self.conversation_history
            
            # Add current message
            messages.append({"role": "user", "content": user_message})
            
            # Get AI response
            response = anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                temperature=0.7,
                system="You are a helpful voice assistant with speaker recognition capabilities.",
                messages=messages
            )
            
            ai_response = response.content[0].text
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": ai_response})
            
            # Keep history manageable
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
            
            # Store in Honcho if available
            if self.honcho_enabled and speaker_id:
                self.store_honcho_interaction(speaker_id, user_message, ai_response)
            
            return ai_response
            
        except Exception as e:
            print(f"{Colors.RED}AI response error: {e}{Colors.RESET}")
            return f"Error getting AI response: {e}"
    
    def get_honcho_context(self, speaker_id, query):
        """Get relevant context from Honcho memory system
        
        Args:
            speaker_id: Speaker identifier
            query: Current query for context
            
        Returns:
            List of context messages
        """
        if not honcho_client:
            return []
        
        try:
            # Convert speaker hashtag to peer_id
            peer_id = f"speaker_{speaker_id[1:]}" if speaker_id.startswith('#') else f"speaker_{speaker_id}"
            
            # Get or create session
            session_id = os.getenv('HONCHO_SESSION_ID', 'voice_assistant_session')
            peer = honcho_client.peer(peer_id)
            session = honcho_client.session(session_id)
            session.add_peers([peer])
            
            # Get context
            context_result = session.get_context(
                tokens=2000,
                summary=True
            )
            
            if not context_result:
                return []
            
            context_messages = []
            
            # Add summary if available
            if hasattr(context_result, 'summary') and context_result.summary:
                context_messages.extend([
                    {"role": "user", "content": "[MEMORY] Summarize recent interactions"},
                    {"role": "assistant", "content": context_result.summary}
                ])
            
            # Add individual messages
            if hasattr(context_result, 'messages') and context_result.messages:
                for msg in context_result.messages:
                    peer_name = getattr(msg, 'peer_id', 'unknown')
                    content = getattr(msg, 'content', str(msg))
                    
                    if peer_name == 'assistant':
                        role = "assistant"
                    else:
                        role = "user"
                        content = f"[{peer_name}] {content}"
                    
                    context_messages.append({"role": role, "content": content})
            
            return context_messages
            
        except Exception as e:
            print(f"Honcho context error: {e}")
            return []
    
    def store_honcho_interaction(self, speaker_id, user_message, ai_response):
        """Store interaction in Honcho memory
        
        Args:
            speaker_id: Speaker identifier
            user_message: User's message
            ai_response: AI's response
        """
        if not honcho_client:
            return
        
        try:
            peer_id = f"speaker_{speaker_id[1:]}" if speaker_id.startswith('#') else f"speaker_{speaker_id}"
            session_id = os.getenv('HONCHO_SESSION_ID', 'voice_assistant_session')
            
            peer = honcho_client.peer(peer_id)
            session = honcho_client.session(session_id)
            
            # Add messages to session
            session.add_messages([
                {"peer_id": peer_id, "content": user_message},
                {"peer_id": "assistant", "content": ai_response}
            ])
            
            print(f"{Colors.CYAN}Stored in Honcho memory{Colors.RESET}")
            
        except Exception as e:
            print(f"Honcho storage error: {e}")
    
    def cleanup(self):
        """Clean up resources"""
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        if self.p:
            self.p.terminate()

class PTTHandler:
    """Push-to-talk keyboard handler"""
    
    def __init__(self, assistant):
        """Initialize PTT handler
        
        Args:
            assistant: VoiceAssistant instance
        """
        self.assistant = assistant
        self.is_key_held = False
        self.pressed_keys = set()
        
    def start(self):
        """Start listening for hotkey"""
        def on_press(key):
            self.pressed_keys.add(key)
            
            # Check for Option+Space (Alt+Space on non-Mac)
            if ((Key.alt_l in self.pressed_keys or Key.alt_r in self.pressed_keys) 
                and Key.space in self.pressed_keys):
                if not self.is_key_held:
                    self.is_key_held = True
                    self.assistant.start_recording()
        
        def on_release(key):
            if self.is_key_held and key in [Key.alt_l, Key.alt_r, Key.space]:
                self.is_key_held = False
                self.assistant.stop_recording()
                
                # Process the recording
                text, speaker_id = self.assistant.process_recorded_audio()
                if text:
                    print(f"\n{Colors.BLUE}You: {text}{Colors.RESET}")
                    
                    # Get AI response
                    response = self.assistant.get_ai_response(text, speaker_id)
                    print(f"{Colors.GREEN}Assistant: {response}{Colors.RESET}\n")
            
            self.pressed_keys.discard(key)
        
        # Set up listener
        listener = keyboard.Listener(
            on_press=on_press,
            on_release=on_release
        )
        
        listener.start()
        print(f"{Colors.GREEN}PTT ready! Hold Option+Space (Alt+Space) to speak{Colors.RESET}")
        
        # Keep listener alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
            listener.stop()

def main():
    """Main entry point"""
    print(f"{Colors.CYAN}Voice Assistant with Speaker Recognition{Colors.RESET}")
    print("=" * 50)
    
    # Check API keys
    if not os.getenv('ANTHROPIC_API_KEY'):
        print(f"{Colors.RED}Error: ANTHROPIC_API_KEY not set in environment{Colors.RESET}")
        print("Please set your Anthropic API key:")
        print("  export ANTHROPIC_API_KEY='your-key-here'")
        return
    
    # Initialize assistant
    assistant = VoiceAssistant(
        use_speaker_recognition=True,
        use_honcho=HONCHO_AVAILABLE
    )
    
    # Set up PTT handler
    ptt = PTTHandler(assistant)
    
    try:
        # Start listening for hotkey
        ptt.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        assistant.cleanup()

if __name__ == "__main__":
    main()