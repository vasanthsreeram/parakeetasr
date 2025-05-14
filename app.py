import os
import tempfile
import time
from flask import Flask, request, jsonify

import nemo.collections.asr as nemo_asr

app = Flask(__name__)

# Global variable to store our ASR model
asr_model = None

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({"status": "healthy", "model_loaded": asr_model is not None})

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """
    Endpoint to transcribe audio files.
    
    Expects a POST request with:
    - audio_file: The audio file to transcribe (.wav or .flac)
    - timestamps: (Optional) Boolean flag to return timestamps
    
    Returns JSON with the transcription and optional timestamps.
    """
    global asr_model
    
    # Check if model is loaded
    if asr_model is None:
        return jsonify({"error": "Model not loaded yet"}), 503
    
    # Check if request has the file
    if 'audio_file' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    file = request.files['audio_file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Check if timestamps are requested (default to False)
    include_timestamps = request.form.get('timestamps', 'false').lower() == 'true'
    
    # Save the uploaded file to a temporary location
    _, temp_path = tempfile.mkstemp(suffix=os.path.splitext(file.filename)[1])
    file.save(temp_path)
    
    try:
        # Time the transcription
        start_time = time.time()
        
        # Transcribe the audio
        result = asr_model.transcribe([temp_path], timestamps=include_timestamps)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Extract text and timestamps if requested
        output = {
            "text": result[0].text,
            "processing_time_seconds": processing_time
        }
        
        if include_timestamps:
            output["word_timestamps"] = result[0].timestamp['word']
            output["segment_timestamps"] = result[0].timestamp['segment']
        
        return jsonify(output)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/load_model', methods=['POST'])
def load_model():
    """
    Endpoint to load the ASR model.
    This can be called separately to initialize the model without transcribing.
    """
    global asr_model
    
    try:
        start_time = time.time()
        asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")
        load_time = time.time() - start_time
        
        return jsonify({
            "status": "Model loaded successfully",
            "load_time_seconds": load_time
        })
    
    except Exception as e:
        return jsonify({"error": f"Failed to load model: {str(e)}"}), 500

if __name__ == '__main__':
    # The model will be loaded on first request or by calling /load_model endpoint
    # For production, use a proper WSGI server like gunicorn or waitress
    app.run(host='0.0.0.0', port=5000) 