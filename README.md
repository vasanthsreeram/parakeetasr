# Parakeet TDT ASR API Server

This server provides a REST API for speech-to-text transcription using NVIDIA's Parakeet TDT 0.6B model.

## Requirements

- Python 3.8+
- NVIDIA GPU with sufficient VRAM (at least 2GB)
- CUDA drivers installed

## Installation

1. Install the required packages:

```bash
pip install -r requirements.txt
```

2. Start the server:

```bash
python app.py
```

For production deployment, use a proper WSGI server:

```bash
waitress-serve --port=5000 app:app
```

or

```bash
gunicorn -w 1 -b 0.0.0.0:5000 app:app
```

## API Endpoints

### Load Model

```
POST /load_model
```

This endpoint loads the ASR model. Call this first before using the transcription endpoint.

Example:

```bash
curl -X POST http://localhost:5000/load_model
```

### Transcribe Audio

```
POST /transcribe
```

Parameters:
- `audio_file`: The audio file to transcribe (.wav or .flac format)
- `timestamps` (optional): Set to "true" to get word and segment timestamps

Example:

```bash
curl -X POST -F "audio_file=@path/to/your/audio.wav" -F "timestamps=true" http://localhost:5000/transcribe
```

Response:

```json
{
  "text": "Transcribed text with punctuation and capitalization.",
  "processing_time_seconds": 0.45,
  "word_timestamps": [...],
  "segment_timestamps": [...]
}
```

### Health Check

```
GET /health
```

Checks if the server is running and if the model is loaded.

Example:

```bash
curl http://localhost:5000/health
```

## Notes

- The model will be downloaded the first time you load it
- Supports audio files up to 24 minutes long
- Includes punctuation, capitalization, and timestamps
- Best performance on 16kHz mono audio files 