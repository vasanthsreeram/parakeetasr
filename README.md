# Parakeet ASR API

This API provides speech-to-text transcription capabilities using NVIDIA's Parakeet TDT 0.6B model. It automatically benchmarks different chunk sizes to find the optimal configuration for your specific audio files and GPU hardware.

## Features

- Automatic speech recognition with NVIDIA's Parakeet TDT 0.6B model
- Automatic chunking for processing long audio files
- Word and segment-level timestamp support
- RESTful API for integration with other applications
- Automatic benchmarking to find optimal performance settings

## Requirements

- Python 3.8+
- CUDA-compatible NVIDIA GPU (minimum 2GB VRAM, 6GB+ recommended)
- ffmpeg (for audio processing)

## Installation

1. Install required system packages:
   ```bash
   sudo apt update
   sudo apt install -y ffmpeg
   ```

2. Install Python dependencies:
   ```bash
   pip install flask nemo_toolkit[asr] torch pydub werkzeug
   pip install cuda-python>=12.3  # Optional: for potentially better performance
   ```

## Usage

### Command Line Options

The application supports two main modes of operation:

1. **Testing Mode** - Runs benchmarks to find the optimal chunk size:
   ```bash
   python asr_app.py --test
   ```
   This will test chunk sizes from 1 minute to 3.5 minutes in 30-second intervals.

2. **Server Mode** - Starts the API server:
   ```bash
   python asr_app.py --serve
   ```
   
   Optional parameters:
   - `--port PORT` - Set the server port (default: 5000)
   - `--chunk-size SIZE` - Specify a chunk size in minutes (overrides the default)

   Example with options:
   ```bash
   python asr_app.py --serve --port 8080 --chunk-size 2.5
   ```

### Recommended Workflow

For best performance, we recommend:

1. First run in test mode to find the optimal chunk size for your hardware:
   ```bash
   python asr_app.py --test
   ```

2. Then start the server with the optimal chunk size:
   ```bash
   python asr_app.py --serve --chunk-size OPTIMAL_SIZE
   ```
   Replace `OPTIMAL_SIZE` with the value determined from the benchmark test.

### API Endpoints

#### POST /transcribe

Transcribes audio from an uploaded file.

**Parameters:**

- `file`: The audio file to transcribe (required)
- `chunk_duration`: Maximum duration of each chunk in minutes (optional, default: server's configured optimal value)
- `timestamps`: Whether to include timestamps in the output (optional, default: true)

**Example curl request:**

```bash
curl -X POST http://localhost:5000/transcribe \
  -F "file=@/path/to/your/audio.mp3" \
  -F "chunk_duration=2.0" \
  -F "timestamps=true"
```

**Example response:**

```json
{
  "text": "This is the transcribed text of the audio file.",
  "processing_time": 12.45,
  "chunks": 3,
  "segments": [
    {
      "segment": "This is the transcribed text",
      "start": 0.0,
      "end": 2.34
    },
    {
      "segment": "of the audio file.",
      "start": 2.34,
      "end": 4.56
    }
  ]
}
```

### Python Client Example

```python
import requests

url = 'http://localhost:5000/transcribe'
files = {'file': open('/path/to/your/audio.mp3', 'rb')}
data = {'chunk_duration': 2.0, 'timestamps': 'true'}

response = requests.post(url, files=files, data=data)
result = response.json()

print(result['text'])
```

## Performance Considerations

- **Memory Usage**: Larger chunk sizes may provide faster overall processing but require more GPU memory.
- **Optimal Settings**: The benchmark will automatically find the best chunk size for your specific file and hardware.
- **VRAM Requirements**: For longer audio files, you need sufficient GPU VRAM. If you encounter out-of-memory errors:
  - Use smaller chunk sizes
  - Ensure other applications aren't using GPU memory
  - Consider upgrading your GPU

## License

This project utilizes NVIDIA's Parakeet TDT 0.6B model, which is subject to NVIDIA's license terms for pre-trained models. 