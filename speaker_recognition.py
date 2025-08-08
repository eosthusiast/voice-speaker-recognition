"""
Speaker Recognition Module for SAN Voice Interface

This module provides speaker identification capabilities using pyannote/embedding
to recognize returning speakers within a single session at Burning Man.
"""

import os
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from scipy.spatial.distance import cosine
import warnings
import hashlib
import random
import json

# Suppress pyannote warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")

try:
    from pyannote.audio import Model, Inference
except ImportError:
    print("Warning: pyannote.audio not available. Speaker recognition disabled.")
    Model = None
    Inference = None

from colors import Colors

# Configuration - defaults can be overridden by environment variables
DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv('SPEAKER_THRESHOLD', '0.52'))  # Cosine distance threshold for speaker matching (increased for better matching)
MAX_SPEAKERS = int(os.getenv('MAX_SPEAKERS', '50'))  # Maximum number of speakers to track in session
MAX_SAMPLES_PER_SPEAKER = int(os.getenv('MAX_SAMPLES_PER_SPEAKER', '5'))  # Maximum audio samples to store per speaker
MIN_AUDIO_LENGTH = float(os.getenv('MIN_AUDIO_LENGTH', '1.5'))  # Minimum audio length in seconds for embedding extraction (reduced from 2.0)
MERGE_SIMILARITY_THRESHOLD = float(os.getenv('MERGE_THRESHOLD', '0.48'))  # More lenient threshold for merging existing speakers (increased)
MIN_INTERACTIONS_BEFORE_MERGE = int(os.getenv('MIN_INTERACTIONS_BEFORE_MERGE', '1'))  # Wait for some data before considering merges (reduced)
RECENT_SPEAKER_BONUS = float(os.getenv('RECENT_SPEAKER_BONUS', '0.10'))  # Distance bonus for speakers who spoke recently (doubled)
RECENT_TIME_WINDOW = float(os.getenv('RECENT_TIME_WINDOW', '60.0'))  # Seconds to consider a speaker "recent" (doubled)

# Burning Man mode - harsh environment settings
BURNING_MAN_MODE = os.getenv('BURNING_MAN_MODE', 'false').lower() in ['true', '1', 'yes']

if BURNING_MAN_MODE:
    print(f"{Colors.YELLOW}🔥 BURNING MAN MODE ACTIVATED 🔥{Colors.RESET}")
    # Override settings for harsh desert environment
    DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv('SPEAKER_THRESHOLD', '0.60'))  # More lenient for dust/noise (increased)
    MERGE_SIMILARITY_THRESHOLD = float(os.getenv('MERGE_THRESHOLD', '0.55'))  # More lenient merging (increased)
    RECENT_SPEAKER_BONUS = float(os.getenv('RECENT_SPEAKER_BONUS', '0.15'))  # Higher bonus for continuity (increased)
    RECENT_TIME_WINDOW = float(os.getenv('RECENT_TIME_WINDOW', '90.0'))  # Longer window (increased)
    MAX_SPEAKERS = int(os.getenv('MAX_SPEAKERS', '200'))  # Expect many more people at festival
    MIN_AUDIO_LENGTH = float(os.getenv('MIN_AUDIO_LENGTH', '1.2'))  # Even shorter clips acceptable in noisy environment
    print(f"{Colors.YELLOW}Dust-adjusted thresholds: similarity={DEFAULT_SIMILARITY_THRESHOLD:.3f}, merge={MERGE_SIMILARITY_THRESHOLD:.3f}{Colors.RESET}")

# Random alphanumeric characters for hashtag generation
HASHTAG_CHARS = 'abcdefghjkmnpqrstuvwxyz23456789'  # Avoid confusing chars like 0,o,1,l,i

# File paths for persistence
HASHTAGS_FILE = "./speaker_hashtags.txt"
RELATIONSHIPS_FILE = "./speaker_relationships.txt"
EMBEDDINGS_FILE = "./speaker_embeddings.npy"  # Binary file for embedding vectors

# Global session storage
session_speakers: List[Dict] = []
hashtag_mappings: Dict[str, Dict] = {}  # hashtag -> speaker info
model = None
inference = None

# Audio preprocessing configuration - DISABLED by default as it destroys speaker characteristics
ENABLE_PREPROCESSING = os.getenv('ENABLE_AUDIO_PREPROCESSING', 'false').lower() in ['true', '1', 'yes']  # Disabled - preprocessing hurts speaker recognition
ENABLE_PITCH_NORMALIZATION = os.getenv('ENABLE_PITCH_NORMALIZATION', 'false').lower() in ['true', '1', 'yes']  # Disabled - pitch is a key speaker characteristic!
ENABLE_VAD = os.getenv('ENABLE_VAD', 'false').lower() in ['true', '1', 'yes']  # Disabled - speaking rhythm matters for identification

def load_speaker_embeddings():
    """Load speaker embeddings from disk and populate session_speakers."""
    global session_speakers
    
    if not os.path.exists(EMBEDDINGS_FILE) or not os.path.exists(HASHTAGS_FILE):
        return
    
    try:
        # Load embeddings array
        embeddings_data = np.load(EMBEDDINGS_FILE, allow_pickle=True).item()
        
        # Rebuild session_speakers from saved data
        for hashtag, info in hashtag_mappings.items():
            embedding_hash = info['embedding_hash']
            if embedding_hash in embeddings_data:
                embedding = embeddings_data[embedding_hash]['embedding']
                session_speakers.append({
                    'speaker_id': hashtag,  # For compatibility with existing code
                    'hashtag': hashtag,
                    'embedding': embedding,  # Main embedding for compatibility
                    'embeddings': [embedding],
                    'embedding_hash': embedding_hash,
                    'first_heard': datetime.fromisoformat(info['first_seen']),
                    'last_heard': datetime.fromisoformat(info['last_seen']),
                    'first_seen': info['first_seen'],  # Keep string version for file writing
                    'last_seen': info['last_seen'],
                    'interaction_count': info['interaction_count']
                })
        
        print(f"{Colors.BLUE}Loaded {len(session_speakers)} speaker embeddings from disk{Colors.RESET}")
        
    except Exception as e:
        print(f"{Colors.YELLOW}Error loading speaker embeddings: {e}{Colors.RESET}")

def save_speaker_embeddings():
    """Save all speaker embeddings to disk."""
    try:
        embeddings_data = {}
        
        for speaker in session_speakers:
            if speaker['embeddings']:  # Only save if there are embeddings
                # Average all embeddings for this speaker
                avg_embedding = np.mean(speaker['embeddings'], axis=0)
                embeddings_data[speaker['embedding_hash']] = {
                    'embedding': avg_embedding,
                    'hashtag': speaker['hashtag']
                }
        
        # Save to binary file
        np.save(EMBEDDINGS_FILE, embeddings_data)
        print(f"{Colors.BLUE}Saved {len(embeddings_data)} speaker embeddings to disk{Colors.RESET}")
        
    except Exception as e:
        print(f"{Colors.YELLOW}Error saving speaker embeddings: {e}{Colors.RESET}")

def load_hashtag_mappings():
    """Load existing hashtag mappings from file."""
    global hashtag_mappings
    
    if os.path.exists(HASHTAGS_FILE):
        try:
            with open(HASHTAGS_FILE, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and '|' in line:
                        parts = line.split('|')
                        if len(parts) >= 5:
                            hashtag, embedding_hash, first_seen, last_seen, interaction_count = parts[:5]
                            hashtag_mappings[hashtag] = {
                                'embedding_hash': embedding_hash,
                                'first_seen': first_seen,
                                'last_seen': last_seen,
                                'interaction_count': int(interaction_count)
                            }
            print(f"{Colors.BLUE}Loaded {len(hashtag_mappings)} existing speaker hashtags{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.YELLOW}Error loading hashtag mappings: {e}{Colors.RESET}")

def save_hashtag_mapping(hashtag: str, embedding_hash: str, interaction_count: int = 1):
    """Save or update a hashtag mapping to file."""
    timestamp = datetime.now().isoformat()
    
    # Update in-memory mapping
    if hashtag in hashtag_mappings:
        hashtag_mappings[hashtag]['last_seen'] = timestamp
        hashtag_mappings[hashtag]['interaction_count'] = interaction_count
    else:
        hashtag_mappings[hashtag] = {
            'embedding_hash': embedding_hash,
            'first_seen': timestamp,
            'last_seen': timestamp,
            'interaction_count': interaction_count
        }
    
    # Write all mappings to file
    try:
        with open(HASHTAGS_FILE, 'w') as f:
            for tag, info in hashtag_mappings.items():
                f.write(f"{tag}|{info['embedding_hash']}|{info['first_seen']}|{info['last_seen']}|{info['interaction_count']}\n")
    except Exception as e:
        print(f"{Colors.YELLOW}Error saving hashtag mapping: {e}{Colors.RESET}")

def generate_embedding_hash(embedding: np.ndarray) -> str:
    """Generate a stable hash from embedding for collision detection."""
    # Use first 32 elements of embedding for hash (stable across similar embeddings)
    embedding_subset = embedding.flatten()[:32]
    embedding_bytes = embedding_subset.tobytes()
    return hashlib.md5(embedding_bytes).hexdigest()[:8]

def generate_speaker_hashtag(embedding: np.ndarray) -> str:
    """Generate a unique random alphanumeric hashtag for a speaker."""
    embedding_hash = generate_embedding_hash(embedding)
    
    # Check if we already have a hashtag for this embedding
    for hashtag, info in hashtag_mappings.items():
        if info['embedding_hash'] == embedding_hash:
            print(f"{Colors.BLUE}Found existing hashtag for embedding: {hashtag}{Colors.RESET}")
            return hashtag
    
    # Generate new random hashtag
    max_attempts = 100
    for _ in range(max_attempts):
        # Generate 4 random alphanumeric characters
        random_chars = ''.join(random.choices(HASHTAG_CHARS, k=4))
        hashtag = f"#{random_chars}"
        
        # Check for collisions
        if hashtag not in hashtag_mappings:
            save_hashtag_mapping(hashtag, embedding_hash)
            print(f"{Colors.GREEN}Generated new speaker hashtag: {hashtag}{Colors.RESET}")
            return hashtag
    
    # Fallback if we can't find a unique hashtag (very unlikely)
    fallback_hashtag = f"#{embedding_hash[:4]}"
    save_hashtag_mapping(fallback_hashtag, embedding_hash)
    print(f"{Colors.YELLOW}Used fallback hashtag: {fallback_hashtag}{Colors.RESET}")
    return fallback_hashtag

def update_relationship_context(hashtag: str, user_message: str = "", san_response: str = ""):
    """Update relationship tracking for a speaker."""
    timestamp = datetime.now().isoformat()
    
    # Simple topic extraction (can be enhanced later)
    topics = []
    nature_keywords = ['tree', 'forest', 'fungal', 'mycel', 'root', 'network', 'nature', 'wisdom', 'earth']
    for keyword in nature_keywords:
        if keyword.lower() in user_message.lower() or keyword.lower() in san_response.lower():
            topics.append(keyword)
    
    # Extract emotional tone (basic sentiment)
    emotional_tone = "neutral"
    if any(word in san_response.lower() for word in ['contemplative', 'wise', 'ancient', 'deep']):
        emotional_tone = "contemplative"
    elif any(word in san_response.lower() for word in ['curious', 'wonder', 'explore']):
        emotional_tone = "curious"
    
    # Append to relationships file
    try:
        with open(RELATIONSHIPS_FILE, 'a') as f:
            topics_str = ','.join(topics) if topics else 'general'
            f.write(f"{hashtag}|{timestamp}|topics:{topics_str}|tone:{emotional_tone}|user:\"{user_message[:100]}...\"|san:\"{san_response[:100]}...\"\n")
    except Exception as e:
        print(f"{Colors.YELLOW}Error updating relationship context: {e}{Colors.RESET}")

def get_relationship_context(hashtag: str) -> str:
    """Get simplified context for a speaker - just the hashtag."""
    return f"[{hashtag}]"

def preprocess_audio(audio_file_path: str) -> Optional[str]:
    """
    Apply audio preprocessing for better speaker recognition.
    
    NOTE: Preprocessing is DISABLED by default because it actually HURTS speaker recognition:
    - Pitch normalization removes a key speaker characteristic (fundamental frequency)
    - VAD removes speaking rhythm and breath patterns that help identify speakers
    - The pyannote model was trained on raw audio and handles variations better than our preprocessing
    
    Args:
        audio_file_path: Path to input audio file
        
    Returns:
        Path to preprocessed audio file, or original path if preprocessing fails
    """
    if not ENABLE_PREPROCESSING:
        return audio_file_path
    
    try:
        import librosa
        import soundfile as sf
        import tempfile
        import shutil
        
        # Create timestamp for debug files
        timestamp = datetime.now().strftime("%H%M%S")
        
        # Load audio
        audio_original, sr = librosa.load(audio_file_path, sr=16000)  # Standard rate for pyannote
        
        # Save original audio for comparison
        original_debug_path = f"./tempaudio/original_{timestamp}.wav"
        sf.write(original_debug_path, audio_original, sr)
        print(f"{Colors.CYAN}Saved original audio: {original_debug_path}{Colors.RESET}")
        
        # Copy for processing
        audio = audio_original.copy()
        
        # Apply preprocessing steps with detailed logging
        preprocessing_steps = []
        
        if ENABLE_VAD:
            audio_before_vad = audio.copy()
            audio = apply_voice_activity_detection(audio, sr)
            duration_before = len(audio_before_vad) / sr
            duration_after = len(audio) / sr
            preprocessing_steps.append(f"VAD: {duration_before:.2f}s → {duration_after:.2f}s")
        
        if ENABLE_PITCH_NORMALIZATION:
            audio = apply_pitch_normalization(audio, sr)  # This function logs its own pitch shift
        
        # Apply spectral normalization (always enabled if preprocessing is on)
        audio_before_norm = audio.copy()
        audio = apply_spectral_normalization(audio)
        std_before = np.std(audio_before_norm)
        std_after = np.std(audio)
        preprocessing_steps.append(f"Spectral norm: std {std_before:.3f} → {std_after:.3f}")
        
        # Save preprocessed audio for comparison
        processed_debug_path = f"./tempaudio/processed_{timestamp}.wav"
        sf.write(processed_debug_path, audio, sr)
        print(f"{Colors.CYAN}Saved processed audio: {processed_debug_path}{Colors.RESET}")
        
        # Log all preprocessing steps
        if preprocessing_steps:
            print(f"{Colors.BLUE}Preprocessing steps: {', '.join(preprocessing_steps)}{Colors.RESET}")
        
        # Save preprocessed audio to temporary file for embedding extraction
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
            sf.write(temp_path, audio, sr)
            
        print(f"{Colors.BLUE}Applied audio preprocessing to {audio_file_path}{Colors.RESET}")
        return temp_path
        
    except ImportError:
        print(f"{Colors.YELLOW}librosa/soundfile not available, skipping preprocessing{Colors.RESET}")
        return audio_file_path
    except Exception as e:
        print(f"{Colors.YELLOW}Audio preprocessing failed: {e}, using original audio{Colors.RESET}")
        return audio_file_path

def apply_voice_activity_detection(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Simple voice activity detection to remove silence.
    
    Args:
        audio: Input audio signal
        sr: Sample rate
        
    Returns:
        Audio with silence removed
    """
    try:
        import librosa
        
        # Simple energy-based VAD
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)   # 10ms hop
        
        # Calculate short-time energy
        energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Threshold for voice activity (adaptive based on signal statistics)
        energy_threshold = np.percentile(energy, 30)  # Bottom 30% likely silence
        
        # Create voice activity mask
        voice_frames = energy > energy_threshold
        
        # Convert frame indices to sample indices
        voice_samples = np.zeros(len(audio), dtype=bool)
        for i, is_voice in enumerate(voice_frames):
            start_sample = i * hop_length
            end_sample = min(start_sample + frame_length, len(audio))
            if is_voice:
                voice_samples[start_sample:end_sample] = True
        
        # Keep only voice segments (with small padding)
        if np.any(voice_samples):
            # Add small padding around voice segments
            padding = int(0.1 * sr)  # 100ms padding
            voice_samples = np.convolve(voice_samples.astype(float), np.ones(padding)/padding, mode='same') > 0.1
            
            return audio[voice_samples]
        else:
            # If no voice detected, return original (may be all voice)
            return audio
            
    except Exception as e:
        print(f"{Colors.YELLOW}VAD failed: {e}, using original audio{Colors.RESET}")
        return audio

def apply_pitch_normalization(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Apply pitch normalization to reduce speaker-specific vocal characteristics.
    
    Args:
        audio: Input audio signal
        sr: Sample rate
        
    Returns:
        Pitch-normalized audio
    """
    try:
        import librosa
        
        # Extract fundamental frequency (F0)
        f0 = librosa.yin(audio, fmin=50, fmax=400, sr=sr)
        
        # Calculate median F0 (ignoring unvoiced frames)
        voiced_f0 = f0[f0 > 0]
        if len(voiced_f0) > 0:
            median_f0 = np.median(voiced_f0)
            target_f0 = 150.0  # Target F0 in Hz (neutral pitch)
            
            # Calculate pitch shift ratio
            pitch_shift_ratio = target_f0 / median_f0
            
            # Apply pitch shifting (limited to reasonable range)
            if 0.7 <= pitch_shift_ratio <= 1.4:  # Limit to ±40% shift
                semitone_shift = 12 * np.log2(pitch_shift_ratio)
                audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitone_shift)
                print(f"{Colors.BLUE}Applied pitch shift: {semitone_shift:.1f} semitones{Colors.RESET}")
        
        return audio
        
    except Exception as e:
        print(f"{Colors.YELLOW}Pitch normalization failed: {e}, using original audio{Colors.RESET}")
        return audio

def apply_spectral_normalization(audio: np.ndarray) -> np.ndarray:
    """
    Apply spectral normalization (similar to cepstral mean normalization).
    
    Args:
        audio: Input audio signal
        
    Returns:
        Spectrally normalized audio
    """
    try:
        # Simple amplitude normalization
        if np.std(audio) > 0:
            audio = (audio - np.mean(audio)) / np.std(audio)
        
        # Soft limiting to prevent clipping
        audio = np.tanh(audio * 0.9)
        
        return audio
        
    except Exception as e:
        print(f"{Colors.YELLOW}Spectral normalization failed: {e}, using original audio{Colors.RESET}")
        return audio

def initialize_speaker_recognition(hf_token: Optional[str] = None) -> bool:
    """
    Initialize the pyannote speaker recognition model.
    
    Args:
        hf_token: HuggingFace access token for pyannote/embedding
        
    Returns:
        bool: True if initialization successful, False otherwise
    """
    global model, inference
    
    if Model is None or Inference is None:
        print(f"{Colors.YELLOW}pyannote.audio not available. Speaker recognition disabled.{Colors.RESET}")
        return False
    
    try:
        # Get HF token from environment if not provided
        if hf_token is None:
            hf_token = os.getenv('HUGGINGFACE_TOKEN') or os.getenv('HF_TOKEN')
        
        if hf_token is None:
            print(f"{Colors.YELLOW}No HuggingFace token found. Set HUGGINGFACE_TOKEN environment variable for speaker recognition.{Colors.RESET}")
            return False
        
        print(f"{Colors.BLUE}Loading pyannote/embedding model...{Colors.RESET}")
        
        # Load the pretrained model
        model = Model.from_pretrained("pyannote/embedding", use_auth_token=hf_token)
        inference = Inference(model, window="whole")
        
        print(f"{Colors.GREEN}Speaker recognition initialized successfully{Colors.RESET}")
        
        # Load existing hashtag mappings and embeddings
        load_hashtag_mappings()
        load_speaker_embeddings()
        
        return True
        
    except Exception as e:
        print(f"{Colors.RED}Failed to initialize speaker recognition: {e}{Colors.RESET}")
        model = None
        inference = None
        return False

def extract_embedding(audio_file_path: str, use_averaging: bool = True) -> Optional[np.ndarray]:
    """
    Extract speaker embedding from audio file with optional averaging.
    
    Args:
        audio_file_path: Path to the audio file
        use_averaging: If True, extract multiple overlapping windows and average
        
    Returns:
        numpy array of speaker embedding, or None if extraction fails
    """
    global inference
    
    if inference is None:
        return None
    
    try:
        # Check if file exists
        if not os.path.exists(audio_file_path):
            print(f"{Colors.YELLOW}Audio file not found: {audio_file_path}{Colors.RESET}")
            return None
        
        # Check audio duration to determine extraction method
        try:
            import librosa
            audio, sr = librosa.load(audio_file_path, sr=16000)
            duration = len(audio) / sr
            
            # For short audio (< 3s), use single embedding for consistency
            if duration < 3.0:
                print(f"{Colors.BLUE}Short audio ({duration:.1f}s), using single embedding extraction{Colors.RESET}")
                return _extract_single_embedding(audio_file_path)
        except:
            # If we can't check duration, default to single embedding
            pass
        
        # For very short audio or when averaging is disabled, use the whole file
        if not use_averaging:
            return _extract_single_embedding(audio_file_path)
        
        # Try to extract multiple embeddings and average them for better stability
        embeddings = _extract_windowed_embeddings(audio_file_path)
        
        if not embeddings:
            # Fall back to single embedding
            return _extract_single_embedding(audio_file_path)
        
        if len(embeddings) == 1:
            return embeddings[0]
        
        # Average the embeddings for better stability
        averaged_embedding = np.mean(embeddings, axis=0)
        print(f"{Colors.BLUE}Averaged {len(embeddings)} embeddings for better stability{Colors.RESET}")
        
        return averaged_embedding.flatten()
        
    except Exception as e:
        print(f"{Colors.YELLOW}Error extracting embedding: {e}{Colors.RESET}")
        return None

def _extract_single_embedding(audio_file_path: str) -> Optional[np.ndarray]:
    """Extract a single embedding from the entire audio file."""
    global inference
    
    try:
        # Extract embedding using pyannote
        embedding = inference(audio_file_path)
        
        if embedding is None or len(embedding) == 0:
            print(f"{Colors.YELLOW}Failed to extract embedding from {audio_file_path}{Colors.RESET}")
            return None
        
        # Convert to numpy array if needed
        if hasattr(embedding, 'data'):
            embedding = embedding.data
        
        # Handle memoryview objects
        if isinstance(embedding, memoryview):
            embedding = np.array(embedding)
        
        # Ensure it's a numpy array
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)
        
        return embedding.flatten()  # Ensure 1D array
        
    except Exception as e:
        print(f"{Colors.YELLOW}Error extracting single embedding: {e}{Colors.RESET}")
        return None

def _extract_windowed_embeddings(audio_file_path: str, window_size: float = 2.0, overlap: float = 0.5) -> List[np.ndarray]:
    """
    Extract embeddings from overlapping windows of audio for better stability.
    
    Args:
        audio_file_path: Path to audio file
        window_size: Size of each window in seconds
        overlap: Overlap between windows (0.0 to 1.0)
        
    Returns:
        List of embeddings from each window
    """
    try:
        import librosa
        
        # Load audio to check duration
        audio, sr = librosa.load(audio_file_path, sr=16000)  # pyannote expects 16kHz
        duration = len(audio) / sr
        
        # If audio is shorter than window size, just return single embedding
        if duration < window_size:
            single_emb = _extract_single_embedding(audio_file_path)
            return [single_emb] if single_emb is not None else []
        
        # Calculate window positions
        step_size = window_size * (1.0 - overlap)
        embeddings = []
        
        start = 0.0
        while start + window_size <= duration:
            end = start + window_size
            
            # Extract window audio
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            window_audio = audio[start_sample:end_sample]
            
            # Save window to temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
                import soundfile as sf
                sf.write(temp_path, window_audio, sr)
            
            try:
                # Extract embedding from window
                window_embedding = _extract_single_embedding(temp_path)
                if window_embedding is not None:
                    embeddings.append(window_embedding)
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_path)
                except:
                    pass
            
            start += step_size
        
        return embeddings
        
    except ImportError:
        print(f"{Colors.YELLOW}librosa not available, using single embedding{Colors.RESET}")
        single_emb = _extract_single_embedding(audio_file_path)
        return [single_emb] if single_emb is not None else []
    except Exception as e:
        print(f"{Colors.YELLOW}Error in windowed embedding extraction: {e}{Colors.RESET}")
        single_emb = _extract_single_embedding(audio_file_path)
        return [single_emb] if single_emb is not None else []

def calculate_speaker_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two speaker embeddings.
    
    Args:
        embedding1: First speaker embedding
        embedding2: Second speaker embedding
        
    Returns:
        float: Cosine distance (0 = identical, 1 = completely different)
    """
    try:
        # Ensure embeddings are 1D
        if len(embedding1.shape) > 1:
            embedding1 = embedding1.flatten()
        if len(embedding2.shape) > 1:
            embedding2 = embedding2.flatten()
        
        # Calculate cosine distance
        distance = cosine(embedding1, embedding2)
        return float(distance)
        
    except Exception as e:
        print(f"{Colors.YELLOW}Error calculating similarity: {e}{Colors.RESET}")
        return 1.0  # Return maximum distance on error

def find_matching_speaker(new_embedding: np.ndarray, 
                         threshold: float = DEFAULT_SIMILARITY_THRESHOLD) -> Optional[str]:
    """
    Find if new embedding matches any existing speaker.
    
    Args:
        new_embedding: Speaker embedding to match
        threshold: Maximum cosine distance for positive match
        
    Returns:
        str: Speaker ID if match found, None otherwise
    """
    global session_speakers
    
    best_match = None
    best_distance = float('inf')
    all_distances = []  # For debugging
    current_time = datetime.now()
    
    print(f"{Colors.YELLOW}Searching among {len(session_speakers)} existing speakers with threshold {threshold:.3f}{Colors.RESET}")
    
    # Debug: Show characteristics of the new embedding
    if len(session_speakers) > 0:
        print(f"{Colors.CYAN}New embedding stats: mean={np.mean(new_embedding):.4f}, std={np.std(new_embedding):.4f}, norm={np.linalg.norm(new_embedding):.4f}{Colors.RESET}")
    
    for speaker in session_speakers:
        # Calculate distances against all stored embeddings for this speaker
        embeddings_to_check = speaker.get('embeddings', [speaker['embedding']])
        distances = []
        
        for stored_embedding in embeddings_to_check:
            distance = calculate_speaker_similarity(new_embedding, stored_embedding)
            distances.append(distance)
        
        # Use the minimum distance (best match) among all embeddings
        min_distance = min(distances)
        avg_distance = sum(distances) / len(distances)
        
        # Apply temporal bias - recent speakers get a distance bonus
        adjusted_distance = min_distance
        if 'last_heard' in speaker:
            time_since_last = (current_time - speaker['last_heard']).total_seconds()
            if time_since_last <= RECENT_TIME_WINDOW:
                # Apply bonus (reduce distance) for recent speakers
                recency_factor = 1.0 - (time_since_last / RECENT_TIME_WINDOW)
                distance_bonus = RECENT_SPEAKER_BONUS * recency_factor
                adjusted_distance = max(0.0, min_distance - distance_bonus)
                
                if distance_bonus > 0.001:  # Only log significant bonuses
                    print(f"{Colors.BLUE}Recent speaker bonus for {speaker['speaker_id']}: -{distance_bonus:.3f} (spoke {time_since_last:.1f}s ago){Colors.RESET}")
        
        # Log for debugging
        bonus_info = f", adjusted={adjusted_distance:.3f}" if adjusted_distance != min_distance else ""
        print(f"{Colors.CYAN}Checking {speaker['speaker_id']}: min_dist={min_distance:.3f}, avg_dist={avg_distance:.3f}, all_dists={[f'{d:.3f}' for d in distances]}{bonus_info}{Colors.RESET}")
        all_distances.append((speaker['speaker_id'], adjusted_distance))
        
        if adjusted_distance < threshold and adjusted_distance < best_distance:
            best_distance = adjusted_distance
            best_match = speaker['speaker_id']
    
    if best_match:
        # Update interaction count
        for speaker in session_speakers:
            if speaker['speaker_id'] == best_match:
                speaker['interaction_count'] += 1
                speaker['last_heard'] = datetime.now()
                
                # Update persistent hashtag mapping
                embedding_hash = generate_embedding_hash(new_embedding)
                save_hashtag_mapping(best_match, embedding_hash, speaker['interaction_count'])
                
                # Store additional embedding for improved matching (up to limit)
                embeddings_updated = False
                if len(speaker.get('embeddings', [])) < MAX_SAMPLES_PER_SPEAKER:
                    if 'embeddings' not in speaker:
                        speaker['embeddings'] = [speaker['embedding']]
                    speaker['embeddings'].append(new_embedding)
                    embeddings_updated = True
                
                # Save embeddings to disk if we added a new one
                if embeddings_updated:
                    save_speaker_embeddings()
                
                print(f"{Colors.GREEN}Matched {best_match} (adjusted distance: {best_distance:.3f}){Colors.RESET}")
                break
    
    # Log all distances for analysis if no match found
    if best_match is None and all_distances:
        sorted_distances = sorted(all_distances, key=lambda x: x[1])
        print(f"{Colors.YELLOW}No match found. Closest speakers:{Colors.RESET}")
        for speaker_id, dist in sorted_distances[:3]:  # Show top 3 closest
            print(f"{Colors.YELLOW}  - {speaker_id}: distance={dist:.3f} (threshold={threshold:.3f}, gap={dist-threshold:.3f}){Colors.RESET}")
        
        # If the closest speaker was very close to threshold, log it as a near-miss
        if sorted_distances and sorted_distances[0][1] < threshold + 0.05:
            print(f"{Colors.RED}NEAR MISS: {sorted_distances[0][0]} was only {sorted_distances[0][1] - threshold:.3f} above threshold!{Colors.RESET}")
        
        # Check for potential speaker consolidation opportunity
        if len(session_speakers) >= 2:
            consolidate_similar_speakers()
    
    return best_match

def consolidate_similar_speakers():
    """
    Check for speakers that might be the same person and merge them.
    Only runs after speakers have accumulated some interaction data.
    """
    global session_speakers
    
    if len(session_speakers) < 2:
        return
    
    speakers_to_remove = set()
    
    # Compare each pair of speakers
    for i, speaker1 in enumerate(session_speakers):
        if speaker1['speaker_id'] in speakers_to_remove:
            continue
            
        # Only consider speakers with some interaction history
        if speaker1['interaction_count'] < MIN_INTERACTIONS_BEFORE_MERGE:
            continue
            
        for j, speaker2 in enumerate(session_speakers[i+1:], i+1):
            if speaker2['speaker_id'] in speakers_to_remove:
                continue
                
            if speaker2['interaction_count'] < MIN_INTERACTIONS_BEFORE_MERGE:
                continue
            
            # Calculate similarity between speakers
            embeddings1 = speaker1.get('embeddings', [speaker1['embedding']])
            embeddings2 = speaker2.get('embeddings', [speaker2['embedding']])
            
            min_distance = float('inf')
            for emb1 in embeddings1:
                for emb2 in embeddings2:
                    distance = calculate_speaker_similarity(emb1, emb2)
                    min_distance = min(min_distance, distance)
            
            # If speakers are very similar, merge them
            if min_distance < MERGE_SIMILARITY_THRESHOLD:
                print(f"{Colors.BLUE}🔄 Consolidating speakers {speaker1['speaker_id']} and {speaker2['speaker_id']} (distance: {min_distance:.3f}){Colors.RESET}")
                
                # Merge speaker2 into speaker1 (keep the one with more interactions)
                if speaker2['interaction_count'] > speaker1['interaction_count']:
                    # Swap so speaker1 is the one we keep
                    speaker1, speaker2 = speaker2, speaker1
                
                # Merge embeddings
                speaker1['embeddings'].extend(speaker2.get('embeddings', [speaker2['embedding']]))
                # Keep only the best embeddings up to limit
                if len(speaker1['embeddings']) > MAX_SAMPLES_PER_SPEAKER:
                    speaker1['embeddings'] = speaker1['embeddings'][:MAX_SAMPLES_PER_SPEAKER]
                
                # Merge other data
                speaker1['interaction_count'] += speaker2['interaction_count']
                if speaker2['last_heard'] > speaker1['last_heard']:
                    speaker1['last_heard'] = speaker2['last_heard']
                
                # Mark speaker2 for removal
                speakers_to_remove.add(speaker2['speaker_id'])
                
                # Update persistent storage
                save_hashtag_mapping(speaker1['speaker_id'], speaker1['embedding_hash'], speaker1['interaction_count'])
                
                print(f"{Colors.GREEN}✓ Merged {speaker2['speaker_id']} into {speaker1['speaker_id']} ({speaker1['interaction_count']} total interactions){Colors.RESET}")
    
    # Remove merged speakers
    if speakers_to_remove:
        session_speakers = [s for s in session_speakers if s['speaker_id'] not in speakers_to_remove]
        
        # Clean up hashtag mappings for removed speakers
        for speaker_id in speakers_to_remove:
            if speaker_id in hashtag_mappings:
                del hashtag_mappings[speaker_id]
        
        # Save updated embeddings
        save_speaker_embeddings()
        
        print(f"{Colors.GREEN}🔄 Speaker consolidation complete. Removed {len(speakers_to_remove)} duplicate speaker(s){Colors.RESET}")

def force_speaker_consolidation():
    """Force immediate speaker consolidation, ignoring minimum interaction requirements."""
    global session_speakers
    
    if len(session_speakers) < 2:
        print(f"{Colors.BLUE}Only {len(session_speakers)} speaker(s), no consolidation needed{Colors.RESET}")
        return
    
    speakers_to_remove = set()
    
    print(f"{Colors.BLUE}🔄 Force consolidating {len(session_speakers)} speakers...{Colors.RESET}")
    
    # Compare each pair of speakers (more aggressive - no minimum interaction requirement)
    for i, speaker1 in enumerate(session_speakers):
        if speaker1['speaker_id'] in speakers_to_remove:
            continue
            
        for j, speaker2 in enumerate(session_speakers[i+1:], i+1):
            if speaker2['speaker_id'] in speakers_to_remove:
                continue
            
            # Calculate similarity between speakers
            embeddings1 = speaker1.get('embeddings', [speaker1['embedding']])
            embeddings2 = speaker2.get('embeddings', [speaker2['embedding']])
            
            min_distance = float('inf')
            for emb1 in embeddings1:
                for emb2 in embeddings2:
                    distance = calculate_speaker_similarity(emb1, emb2)
                    min_distance = min(min_distance, distance)
            
            # More aggressive merging threshold for force consolidation
            force_merge_threshold = MERGE_SIMILARITY_THRESHOLD + 0.05  # Slightly more lenient
            
            if min_distance < force_merge_threshold:
                print(f"{Colors.BLUE}🔄 Force merging {speaker1['speaker_id']} and {speaker2['speaker_id']} (distance: {min_distance:.3f}){Colors.RESET}")
                
                # Merge speaker2 into speaker1 (keep the one with more interactions)
                if speaker2['interaction_count'] > speaker1['interaction_count']:
                    speaker1, speaker2 = speaker2, speaker1
                
                # Merge embeddings and data (same as regular consolidation)
                speaker1['embeddings'].extend(speaker2.get('embeddings', [speaker2['embedding']]))
                if len(speaker1['embeddings']) > MAX_SAMPLES_PER_SPEAKER:
                    speaker1['embeddings'] = speaker1['embeddings'][:MAX_SAMPLES_PER_SPEAKER]
                
                speaker1['interaction_count'] += speaker2['interaction_count']
                if speaker2['last_heard'] > speaker1['last_heard']:
                    speaker1['last_heard'] = speaker2['last_heard']
                
                speakers_to_remove.add(speaker2['speaker_id'])
                save_hashtag_mapping(speaker1['speaker_id'], speaker1['embedding_hash'], speaker1['interaction_count'])
    
    # Remove merged speakers
    if speakers_to_remove:
        session_speakers = [s for s in session_speakers if s['speaker_id'] not in speakers_to_remove]
        
        for speaker_id in speakers_to_remove:
            if speaker_id in hashtag_mappings:
                del hashtag_mappings[speaker_id]
        
        save_speaker_embeddings()
        print(f"{Colors.GREEN}🔄 Force consolidation complete. Merged {len(speakers_to_remove)} duplicate speaker(s). Now have {len(session_speakers)} unique speakers.{Colors.RESET}")
    else:
        print(f"{Colors.BLUE}No speakers were similar enough to merge.{Colors.RESET}")

def add_new_speaker(embedding: np.ndarray) -> str:
    """
    Add a new speaker to the session.
    
    Args:
        embedding: Speaker embedding
        
    Returns:
        str: New speaker ID
    """
    global session_speakers
    
    # Clean up old speakers if we've hit the limit
    if len(session_speakers) >= MAX_SPEAKERS:
        # Remove oldest speaker by last_heard time
        session_speakers.sort(key=lambda x: x.get('last_heard', x['first_heard']))
        removed_speaker = session_speakers.pop(0)
        print(f"{Colors.YELLOW}Removed oldest speaker {removed_speaker['speaker_id']} to make room{Colors.RESET}")
    
    # Generate nature-themed hashtag for speaker ID
    speaker_id = generate_speaker_hashtag(embedding)
    
    # Create speaker record
    embedding_hash = generate_embedding_hash(embedding)
    speaker_record = {
        'speaker_id': speaker_id,
        'hashtag': speaker_id,  # For consistency with loaded speakers
        'embedding': embedding,
        'embeddings': [embedding],  # Store multiple embeddings for better matching
        'embedding_hash': embedding_hash,
        'first_heard': datetime.now(),
        'last_heard': datetime.now(),
        'interaction_count': 1,
        'confidence_scores': [1.0]  # Track confidence over time
    }
    
    session_speakers.append(speaker_record)
    print(f"{Colors.GREEN}Added new speaker: {speaker_id}{Colors.RESET}")
    
    # Save embeddings to disk for persistence across sessions
    save_speaker_embeddings()
    
    return speaker_id

def identify_speaker(audio_file_path: str, 
                    threshold: float = DEFAULT_SIMILARITY_THRESHOLD) -> Tuple[Optional[str], float]:
    """
    Main function to identify speaker from audio file.
    
    Args:
        audio_file_path: Path to audio file
        threshold: Similarity threshold for matching
        
    Returns:
        Tuple of (speaker_id or None, confidence_score)
    """
    if inference is None:
        return None, 0.0
    
    try:
        # Check if audio file exists
        if not os.path.exists(audio_file_path):
            print(f"{Colors.YELLOW}Audio file not found: {audio_file_path}{Colors.RESET}")
            return None, 0.0
        
        # Check audio duration to avoid low-quality embeddings from short clips
        try:
            import librosa
            audio, sr = librosa.load(audio_file_path, sr=None)
            duration = len(audio) / sr
            
            if duration < MIN_AUDIO_LENGTH:
                print(f"{Colors.YELLOW}Audio too short ({duration:.1f}s < {MIN_AUDIO_LENGTH}s), skipping speaker recognition{Colors.RESET}")
                return None, 0.0
        except ImportError:
            print(f"{Colors.YELLOW}librosa not available, unable to check audio duration{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.YELLOW}Error checking audio duration: {e}{Colors.RESET}")
        
        # Apply audio preprocessing for better recognition
        preprocessed_audio_path = preprocess_audio(audio_file_path)
        cleanup_preprocessed = preprocessed_audio_path != audio_file_path
        
        try:
            # Extract embedding
            start_time = datetime.now()
            embedding = extract_embedding(preprocessed_audio_path)
        finally:
            # Clean up preprocessed file if it was created
            if cleanup_preprocessed:
                try:
                    os.unlink(preprocessed_audio_path)
                except:
                    pass
        
        if embedding is None:
            return None, 0.0
        
        # Try to match existing speaker
        speaker_id = find_matching_speaker(embedding, threshold)
        
        if speaker_id is None:
            # Add as new speaker
            speaker_id = add_new_speaker(embedding)
            confidence = 1.0  # High confidence for new speaker
        else:
            # Calculate confidence based on similarity
            best_distance = float('inf')
            for speaker in session_speakers:
                if speaker['speaker_id'] == speaker_id:
                    distance = calculate_speaker_similarity(embedding, speaker['embedding'])
                    best_distance = min(best_distance, distance)
                    break
            
            confidence = max(0.0, 1.0 - (best_distance / threshold))
        
        processing_time = (datetime.now() - start_time).total_seconds()
        print(f"{Colors.BLUE}Speaker identification completed in {processing_time:.3f}s: {speaker_id} (confidence: {confidence:.3f}){Colors.RESET}")
        
        # Check for speakers to merge more aggressively - after every new speaker or periodically
        if speaker_id:
            if confidence == 1.0:  # New speaker added
                print(f"{Colors.BLUE}Running consolidation after adding new speaker{Colors.RESET}")
                check_and_merge_speakers()
            elif len(session_speakers) >= 2 and len(session_speakers) % 2 == 0:  # Every 2 speakers
                print(f"{Colors.BLUE}Running periodic consolidation (have {len(session_speakers)} speakers){Colors.RESET}")
                check_and_merge_speakers()
        
        return speaker_id, confidence
        
    except Exception as e:
        print(f"{Colors.RED}Error in speaker identification: {e}{Colors.RESET}")
        return None, 0.0

def get_speaker_context(speaker_id: str) -> str:
    """
    Get contextual information about a speaker for SAN.
    
    Args:
        speaker_id: Speaker identifier (hashtag)
        
    Returns:
        str: Context string for SAN prompt with relationship history
    """
    if speaker_id is None:
        return "[New speaker]"
    
    # Use the enhanced relationship context system
    return get_relationship_context(speaker_id)

def find_speakers_to_merge(threshold: float = MERGE_SIMILARITY_THRESHOLD) -> List[Tuple[str, str, float]]:
    """
    Find pairs of speakers that might be the same person and should be merged.
    
    Args:
        threshold: Maximum distance for considering speakers for merging
        
    Returns:
        List of tuples: (speaker_id_1, speaker_id_2, distance)
    """
    global session_speakers
    merge_candidates = []
    
    # Only consider speakers with enough interactions
    eligible_speakers = [s for s in session_speakers if s['interaction_count'] >= MIN_INTERACTIONS_BEFORE_MERGE]
    
    if len(eligible_speakers) < 2:
        return merge_candidates
    
    print(f"{Colors.BLUE}Checking for speakers to merge (threshold: {threshold:.3f})...{Colors.RESET}")
    
    # Compare each speaker with every other speaker
    for i in range(len(eligible_speakers)):
        for j in range(i + 1, len(eligible_speakers)):
            speaker1 = eligible_speakers[i]
            speaker2 = eligible_speakers[j]
            
            # Find minimum distance between all embeddings of both speakers
            min_distance = float('inf')
            for emb1 in speaker1.get('embeddings', [speaker1['embedding']]):
                for emb2 in speaker2.get('embeddings', [speaker2['embedding']]):
                    distance = calculate_speaker_similarity(emb1, emb2)
                    min_distance = min(min_distance, distance)
            
            if min_distance < threshold:
                merge_candidates.append((speaker1['speaker_id'], speaker2['speaker_id'], min_distance))
                print(f"{Colors.YELLOW}Merge candidate: {speaker1['speaker_id']} + {speaker2['speaker_id']} (distance: {min_distance:.3f}){Colors.RESET}")
    
    return merge_candidates

def merge_speakers(speaker_id_1: str, speaker_id_2: str) -> bool:
    """
    Merge two speakers into one, combining their data.
    The speaker with more interactions becomes the primary.
    
    Args:
        speaker_id_1: First speaker ID
        speaker_id_2: Second speaker ID
        
    Returns:
        bool: True if merge successful, False otherwise
    """
    global session_speakers
    
    # Find the speaker objects
    speaker1_obj = None
    speaker2_obj = None
    speaker1_idx = None
    speaker2_idx = None
    
    for i, speaker in enumerate(session_speakers):
        if speaker['speaker_id'] == speaker_id_1:
            speaker1_obj = speaker
            speaker1_idx = i
        elif speaker['speaker_id'] == speaker_id_2:
            speaker2_obj = speaker
            speaker2_idx = i
    
    if speaker1_obj is None or speaker2_obj is None:
        print(f"{Colors.RED}Cannot merge: speakers not found{Colors.RESET}")
        return False
    
    # Determine primary speaker (most interactions)
    if speaker1_obj['interaction_count'] >= speaker2_obj['interaction_count']:
        primary = speaker1_obj
        secondary = speaker2_obj
        secondary_idx = speaker2_idx
        kept_id = speaker_id_1
        merged_id = speaker_id_2
    else:
        primary = speaker2_obj
        secondary = speaker1_obj
        secondary_idx = speaker1_idx
        kept_id = speaker_id_2
        merged_id = speaker_id_1
    
    # Merge data into primary speaker
    primary['interaction_count'] += secondary['interaction_count']
    
    # Combine embeddings (keep best ones up to limit)
    primary_embeddings = primary.get('embeddings', [primary['embedding']])
    secondary_embeddings = secondary.get('embeddings', [secondary['embedding']])
    
    all_embeddings = primary_embeddings + secondary_embeddings
    # Keep up to MAX_SAMPLES_PER_SPEAKER best quality embeddings
    # For now, just keep most recent ones
    if len(all_embeddings) > MAX_SAMPLES_PER_SPEAKER:
        all_embeddings = all_embeddings[-MAX_SAMPLES_PER_SPEAKER:]
    
    primary['embeddings'] = all_embeddings
    primary['embedding'] = all_embeddings[0]  # Update primary embedding
    
    # Update timestamps
    primary['last_heard'] = max(primary['last_heard'], secondary['last_heard'])
    primary['first_heard'] = min(primary['first_heard'], secondary['first_heard'])
    
    # Combine confidence scores
    primary_scores = primary.get('confidence_scores', [1.0])
    secondary_scores = secondary.get('confidence_scores', [1.0])
    primary['confidence_scores'] = primary_scores + secondary_scores
    
    # Remove secondary speaker
    session_speakers.pop(secondary_idx)
    
    print(f"{Colors.GREEN}Merged {merged_id} into {kept_id} ({primary['interaction_count']} total interactions){Colors.RESET}")
    return True

def check_and_merge_speakers(threshold: float = MERGE_SIMILARITY_THRESHOLD) -> int:
    """
    Check for speakers that should be merged and merge them automatically.
    
    Args:
        threshold: Distance threshold for merging
        
    Returns:
        int: Number of merges performed
    """
    merge_candidates = find_speakers_to_merge(threshold)
    merges_performed = 0
    
    # Sort by distance (merge most similar first)
    merge_candidates.sort(key=lambda x: x[2])
    
    for speaker1_id, speaker2_id, distance in merge_candidates:
        # Check if both speakers still exist (might have been merged already)
        speaker1_exists = any(s['speaker_id'] == speaker1_id for s in session_speakers)
        speaker2_exists = any(s['speaker_id'] == speaker2_id for s in session_speakers)
        
        if speaker1_exists and speaker2_exists:
            if merge_speakers(speaker1_id, speaker2_id):
                merges_performed += 1
    
    if merges_performed > 0:
        print(f"{Colors.GREEN}Performed {merges_performed} speaker merges{Colors.RESET}")
    
    return merges_performed

def get_session_stats() -> Dict:
    """
    Get statistics about the current speaker recognition session.
    
    Returns:
        dict: Session statistics
    """
    total_speakers = len(session_speakers)
    total_interactions = sum(speaker['interaction_count'] for speaker in session_speakers)
    
    # Calculate inter-speaker distances for analysis
    distance_matrix = {}
    if len(session_speakers) > 1:
        for i, speaker1 in enumerate(session_speakers):
            for j, speaker2 in enumerate(session_speakers):
                if i < j:  # Only calculate upper triangle
                    distance = calculate_speaker_similarity(
                        speaker1['embedding'], 
                        speaker2['embedding']
                    )
                    key = f"{speaker1['speaker_id']}-{speaker2['speaker_id']}"
                    distance_matrix[key] = round(distance, 3)
    
    return {
        'total_speakers': total_speakers,
        'total_interactions': total_interactions,
        'active_speakers': [s['speaker_id'] for s in session_speakers],
        'model_initialized': model is not None,
        'inter_speaker_distances': distance_matrix,
        'configuration': {
            'burning_man_mode': BURNING_MAN_MODE,
            'similarity_threshold': DEFAULT_SIMILARITY_THRESHOLD,
            'merge_threshold': MERGE_SIMILARITY_THRESHOLD,
            'recent_speaker_bonus': RECENT_SPEAKER_BONUS,
            'recent_time_window': RECENT_TIME_WINDOW,
            'max_speakers': MAX_SPEAKERS,
            'max_samples_per_speaker': MAX_SAMPLES_PER_SPEAKER
        }
    }

def reset_session():
    """Reset all speaker data for new session."""
    global session_speakers
    session_speakers.clear()
    print(f"{Colors.BLUE}Speaker recognition session reset{Colors.RESET}")

# Test function for debugging
def test_speaker_recognition():
    """Test speaker recognition with sample audio (for debugging)."""
    print(f"{Colors.BLUE}Testing speaker recognition...{Colors.RESET}")
    
    # Try to initialize
    if initialize_speaker_recognition():
        stats = get_session_stats()
        print(f"Session stats: {stats}")
        print(f"{Colors.GREEN}Speaker recognition test completed{Colors.RESET}")
    else:
        print(f"{Colors.YELLOW}Speaker recognition test skipped - not initialized{Colors.RESET}")

if __name__ == "__main__":
    test_speaker_recognition()