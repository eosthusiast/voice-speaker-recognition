# Voice Assistant with Speaker Recognition

A real-time voice assistant combining MLX Whisper transcription with speaker recognition for personalized conversations. Built for Apple Silicon but adaptable to other platforms.

## What It Does

- **Voice Input**: Push-to-talk interface (Option+Space) for hands-free interaction
- **Speaker ID**: Recognizes returning speakers using [PyAnnote embeddings](https://huggingface.co/pyannote/embedding) - the core voice fingerprinting component
- **Fast Transcription**: MLX Whisper optimized for Apple Silicon (M1/M2/M3/M4)
- **AI Chat**: Anthropic Claude integration with speaker-aware context
- **Memory**: Optional conversation persistence via Honcho

## Quick Start

1. **Install (recommended - uv)**
   ```bash
   # Install uv if needed: curl -LsSf https://astral.sh/uv/install.sh | sh
   uv sync
   ```

2. **Configure**
   ```bash
   cp .env.example .env
   # Add your ANTHROPIC_API_KEY and HF_TOKEN to .env
   ```

3. **Run**
   ```bash
   uv run voice_assistant.py
   # Hold Option+Space, speak, release to get AI response
   ```

## Installation Options

**With uv (fastest, recommended):**
```bash
uv sync
uv run voice_assistant.py
```

**With pip:**
```bash
pip install -r requirements.txt
```

**System dependencies:**
- **macOS**: `brew install portaudio`
- **Linux**: `sudo apt install portaudio19-dev python3-dev`
- **Windows**: Usually works out of the box

## Platform Compatibility

### MLX Whisper (Apple Silicon Only)
This project uses **MLX Whisper Large-v3-Turbo** for multilingual support and language auto-detection on Apple Silicon (M1/M2/M3/M4). **MLX only runs on macOS**.

**For English-only and faster performance:**
```python
# In voice_assistant.py, change:
WHISPER_MODEL_NAME = "mlx-community/whisper-base"  # Much faster for English
```

**For non-Mac users, replace MLX Whisper with:**

**OpenAI Whisper (CPU/GPU):**
```bash
# Replace in requirements.txt:
# mlx-whisper>=0.4.2
openai-whisper>=20231117

# Update voice_assistant.py transcription:
import whisper
model = whisper.load_model("base")  # or "small" for English-only speed
result = model.transcribe(audio_path)
```

**Faster Whisper (CPU optimized):**
```bash
pip install faster-whisper
# ~2-4x faster than OpenAI Whisper on CPU
# Use "small" or "base" models for English-only speed
```

**WhisperX (best accuracy + speed):**
```bash
pip install whisperx
# Includes speaker diarization, good for multi-speaker scenarios
```

## Speaker Recognition Only

If you only need speaker recognition without the voice assistant:

```python
from speaker_recognition import initialize_speaker_recognition, identify_speaker

# One-time setup
initialize_speaker_recognition()

# Identify speaker from audio file
speaker_id, confidence = identify_speaker("path/to/audio.wav")

if speaker_id:
    print(f"Recognized speaker: {speaker_id} (confidence: {confidence:.3f})")
else:
    print("New speaker - will be assigned ID on next interaction")
```

## Full Voice Assistant Usage

```python
from voice_assistant import VoiceAssistant

# Initialize
assistant = VoiceAssistant(use_speaker_recognition=True)

# Get AI response for text + speaker
response = assistant.get_ai_response("Hello!", speaker_id="#user123")
```

## Configuration

**Required API Keys:**
- `ANTHROPIC_API_KEY` - Get from [Anthropic Console](https://console.anthropic.com)
- `HF_TOKEN` - Get from [HuggingFace Settings](https://huggingface.co/settings/tokens) (needed for PyAnnote speaker recognition)

**Optional:**
- `HONCHO_API_KEY` - For conversation memory
- `SPEAKER_THRESHOLD=0.52` - Speaker matching sensitivity
- `HARSH_ENV_MODE=false` - Harsh environment mode

### Getting HuggingFace Token
1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click "New token" 
3. Name it (e.g., "speaker-recognition")
4. Select "Read" role
5. Copy token to `.env` as `HF_TOKEN=hf_your_token_here`

## API Reference

### Speaker Recognition Functions

```python
# Initialize the system (call once)
initialize_speaker_recognition(hf_token=None) -> bool

# Identify speaker from audio file  
identify_speaker(audio_path, threshold=0.52) -> Tuple[Optional[str], float]

# Get session statistics
get_session_stats() -> Dict[str, int]
```

### Audio Requirements

- **Formats**: WAV, MP3, M4A, FLAC
- **Sample Rate**: Any (auto-converted to 16kHz internally)
- **Channels**: Mono or Stereo (converted to mono)
- **Duration**: Minimum 1.5 seconds recommended
- **Quality**: Higher quality = better recognition

### Speaker IDs

- **Format**: 4-character hashtags (e.g., `#je7m`, `#3n6v`)
- **Generation**: Automatic on first detection
- **Persistence**: Saved to `speaker_hashtags.txt`
- **Confidence**: Lower values = higher confidence (cosine distance)

### Thresholds

- **Default**: 0.52 (balanced accuracy/new speaker detection)
- **Strict**: 0.35 (fewer false matches, more new speakers)
- **Lenient**: 0.70 (more matches, risk of false positives)

## Integration Examples

### Minimal Setup (Speaker Recognition Only)

```python
# Install minimal dependencies
pip install pyannote-audio numpy scipy

# Basic usage
from speaker_recognition import initialize_speaker_recognition, identify_speaker

initialize_speaker_recognition()
speaker_id, confidence = identify_speaker("audio.wav")
```

### Batch Processing

```python
results = []
for audio_file in audio_files:
    speaker_id, confidence = identify_speaker(audio_file)
    results.append({'file': audio_file, 'speaker': speaker_id, 'confidence': confidence})
```

### Pipeline Integration

```python
def add_speaker_recognition(audio_file, existing_results):
    speaker_id, confidence = identify_speaker(audio_file)
    existing_results['speaker'] = {'id': speaker_id, 'confidence': confidence}
    return existing_results
```

## Files

- `voice_assistant.py` - Full voice assistant with PTT
- `speaker_recognition.py` - Core speaker recognition module
- `examples/basic_usage.py` - Full system examples  
- `examples/speaker_only.py` - Minimal speaker recognition example
- `examples/integration_patterns.py` - Common integration patterns

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- [MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) - Fast transcription on Apple Silicon
- [PyAnnote Embedding](https://huggingface.co/pyannote/embedding) - Core speaker voice embeddings model
- [PyAnnote Audio](https://github.com/pyannote/pyannote-audio) - Speaker recognition toolkit
- [Anthropic Claude](https://anthropic.com) - AI conversation
- [Honcho](https://honcho.dev) - Memory management