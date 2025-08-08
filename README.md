# Voice Assistant with Speaker Recognition

A real-time voice assistant combining MLX Whisper transcription with speaker recognition for personalized conversations. Built for Apple Silicon but adaptable to other platforms.

## What It Does

- **Voice Input**: Push-to-talk interface (Option+Space) for hands-free interaction
- **Speaker ID**: Recognizes returning speakers using voice fingerprints 
- **Fast Transcription**: MLX Whisper optimized for Apple Silicon (M1/M2/M3)
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
   # Add your ANTHROPIC_API_KEY to .env
   ```

3. **Run**
   ```bash
   uv run python voice_assistant.py
   # Hold Option+Space, speak, release to get AI response
   ```

## Installation Options

**With uv (fastest, recommended):**
```bash
uv sync
uv run python voice_assistant.py
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
This project uses **MLX Whisper Large-v3-Turbo** for multilingual support and language auto-detection on Apple Silicon (M1/M2/M3). **MLX only runs on macOS**.

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

## Basic Usage

```python
from voice_assistant import VoiceAssistant

# Initialize
assistant = VoiceAssistant(use_speaker_recognition=True)

# Get AI response for text + speaker
response = assistant.get_ai_response("Hello!", speaker_id="#user123")
```

## Configuration

**Required:** `ANTHROPIC_API_KEY` in `.env`

**Optional:**
- `HONCHO_API_KEY` - For conversation memory
- `SPEAKER_THRESHOLD=0.52` - Speaker matching sensitivity
- `HARSH_ENV_MODE=false` - Harsh environment mode

## Files

- `voice_assistant.py` - Main application with PTT
- `speaker_recognition.py` - Voice fingerprinting  
- `examples/basic_usage.py` - Integration examples

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- [MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) - Fast transcription on Apple Silicon
- [PyAnnote Audio](https://github.com/pyannote/pyannote-audio) - Speaker voice embeddings
- [Anthropic Claude](https://anthropic.com) - AI conversation
- [Honcho](https://honcho.dev) - Memory management