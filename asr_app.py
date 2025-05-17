import os
import time
import tempfile
import json
import requests
import nemo.collections.asr as nemo_asr
import torch
from pydub import AudioSegment
import math
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import argparse

# Global variables
MODEL = None
SAMPLE_RATE = 16000
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a'}
TEST_AUDIO_PATH = '/home/Ubuntu/parakeetasr/New Recording 22.m4a'

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB max upload

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    """Load the ASR model."""
    global MODEL
    if MODEL is None:
        print("Loading Parakeet TDT 0.6B model (this may take a while)...")
        start_time = time.time()
        MODEL = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")
        print(f"Model loaded in {time.time() - start_time:.2f} seconds")
    return MODEL

def transcribe_audio_chunk(model, audio_chunk_path, include_timestamps=True):
    """Helper function to transcribe a single audio chunk."""
    return model.transcribe([audio_chunk_path], timestamps=include_timestamps)

def transcribe_audio(audio_path, chunk_duration_minutes, include_timestamps=True):
    """Transcribe audio file, chunking if necessary."""
    model = load_model()
    if not audio_path:
        return {"error": "No audio file provided"}, None, 0

    current_chunk_duration_ms = chunk_duration_minutes * 60 * 1000
    if current_chunk_duration_ms <= 0:
        return {"error": "Chunk duration must be positive."}, None, 0

    overall_processing_time = 0
    combined_text = ""
    combined_word_timestamps = []
    combined_segment_timestamps = []
    temp_chunk_files = []

    try:
        print(f"Loading audio file: {audio_path}")
        audio = AudioSegment.from_file(audio_path)
        duration_ms = len(audio)
        print(f"Audio duration: {duration_ms / 1000.0:.2f} seconds")

        if duration_ms <= current_chunk_duration_ms:
            print(f"Audio is short enough ({duration_ms/1000.0:.2f}s), processing as a single chunk.")
            start_time = time.time()
            result = transcribe_audio_chunk(model, audio_path, include_timestamps)
            overall_processing_time = time.time() - start_time
            
            text = result[0].text
            segments_dict = None
            if include_timestamps and result[0].timestamp:
                word_timestamps = result[0].timestamp.get('word', [])
                segment_timestamps = result[0].timestamp.get('segment', [])
                segments_dict = segment_timestamps
                
            return {"text": text, "processing_time": overall_processing_time, "chunks": 1}, segments_dict, overall_processing_time
        else:
            num_chunks = math.ceil(duration_ms / current_chunk_duration_ms)
            print(f"Audio too long ({duration_ms/1000.0:.2f}s). Splitting into {num_chunks} chunks of max {current_chunk_duration_ms/1000.0:.2f}s each.")
            current_offset_s = 0

            for i in range(num_chunks):
                start_ms = i * current_chunk_duration_ms
                end_ms = min((i + 1) * current_chunk_duration_ms, duration_ms)
                print(f"Preparing chunk {i+1}/{num_chunks}: from {start_ms/1000.0:.2f}s to {end_ms/1000.0:.2f}s")
                chunk = audio[start_ms:end_ms]
                
                chunk_fd, chunk_path = tempfile.mkstemp(suffix=".wav")
                os.close(chunk_fd)
                chunk.export(chunk_path, format="wav")
                temp_chunk_files.append(chunk_path)
                
                print(f"Transcribing chunk {i+1}/{num_chunks} (file: {chunk_path})...")
                start_chunk_time = time.time()
                chunk_result = transcribe_audio_chunk(model, chunk_path, include_timestamps)
                chunk_processing_time = time.time() - start_chunk_time
                overall_processing_time += chunk_processing_time
                print(f"Chunk {i+1} transcribed in {chunk_processing_time:.2f}s.")

                combined_text += chunk_result[0].text + " " 

                if include_timestamps and chunk_result[0].timestamp:
                    if 'word' in chunk_result[0].timestamp:
                        for ts in chunk_result[0].timestamp['word']:
                            combined_word_timestamps.append({
                                'word': ts['word'],
                                'start': ts['start'] + current_offset_s,
                                'end': ts['end'] + current_offset_s
                            })
                    if 'segment' in chunk_result[0].timestamp:
                        for ts in chunk_result[0].timestamp['segment']:
                            combined_segment_timestamps.append({
                                'segment': ts['segment'],
                                'start': ts['start'] + current_offset_s,
                                'end': ts['end'] + current_offset_s
                            })
                current_offset_s += (end_ms - start_ms) / 1000.0
            
            return {"text": combined_text.strip(), "processing_time": overall_processing_time, "chunks": num_chunks}, combined_segment_timestamps, overall_processing_time

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"error": str(e)}, None, 0
    finally:
        print("Cleaning up temporary chunk files...")
        for temp_file in temp_chunk_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                print(f"Removed temp file: {temp_file}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA cache cleared after transcription attempt.")

def benchmark_chunk_sizes(audio_path, min_minutes=1.0, max_minutes=3.5, step_minutes=0.5):
    """Test different chunk sizes and find the optimal one."""
    print(f"Starting benchmark on {audio_path}")
    print(f"Testing chunk sizes from {min_minutes} to {max_minutes} minutes with {step_minutes} minute steps")
    
    results = []
    
    # Ensure the model is loaded before benchmarking to avoid counting model loading time
    load_model()
    
    # Use a floating-point range with the specified step
    chunk_size = min_minutes
    while chunk_size <= max_minutes:
        print(f"Testing {chunk_size:.1f} minute chunks...")
        
        # Run the transcription and measure time
        result, _, processing_time = transcribe_audio(
            audio_path, 
            chunk_duration_minutes=chunk_size,
            include_timestamps=True
        )
        
        if "error" in result:
            print(f"Error with chunk size {chunk_size:.1f} minutes: {result['error']}")
            chunk_size += step_minutes
            continue
            
        results.append({
            "chunk_size_minutes": chunk_size,
            "processing_time_seconds": processing_time,
            "chunks": result.get("chunks", 0)
        })
        
        print(f"Chunk size: {chunk_size:.1f} minutes, Processing time: {processing_time:.2f} seconds, Chunks: {result.get('chunks', 0)}")
        
        # Clear GPU memory between runs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        chunk_size += step_minutes
    
    # Find the fastest chunk size
    if results:
        fastest = min(results, key=lambda x: x["processing_time_seconds"])
        print("\n===== BENCHMARK RESULTS =====")
        for result in sorted(results, key=lambda x: x["processing_time_seconds"]):
            print(f"Chunk size: {result['chunk_size_minutes']:.1f} min, Time: {result['processing_time_seconds']:.2f}s, Chunks: {result['chunks']}")
        print(f"\nFastest configuration: {fastest['chunk_size_minutes']:.1f} minute chunks - {fastest['processing_time_seconds']:.2f} seconds")
        return fastest["chunk_size_minutes"]
    else:
        print("No valid benchmark results obtained")
        return None

# API Routes
@app.route('/transcribe', methods=['POST'])
def transcribe_endpoint():
    """API endpoint to transcribe an uploaded file."""
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if file and allowed_file(file.filename):
        # Parse form parameters
        chunk_duration = float(request.form.get('chunk_duration', app.config.get('OPTIMAL_CHUNK_SIZE', 2.0)))
        include_timestamps = request.form.get('timestamps', 'true').lower() == 'true'
        
        # Save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the file
        try:
            result, segments, _ = transcribe_audio(
                filepath, 
                chunk_duration_minutes=chunk_duration,
                include_timestamps=include_timestamps
            )
            
            # Add segments to the result if they exist
            if segments:
                result["segments"] = segments
                
            # Delete the file after processing
            os.remove(filepath)
            
            return jsonify(result)
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"error": str(e)}), 500
    
    return jsonify({"error": "File type not allowed"}), 400

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Parakeet ASR API Server')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--test', action='store_true', help='Run benchmark tests')
    group.add_argument('--serve', action='store_true', help='Start the API server')
    parser.add_argument('--port', type=int, default=5000, help='Port for the API server')
    parser.add_argument('--chunk-size', type=float, help='Set a specific chunk size (in minutes) for the API server')
    return parser.parse_args()

# Run the benchmark and API server
if __name__ == "__main__":
    try:
        args = parse_arguments()
        
        if args.test:
            # Run benchmark
            print("Running benchmark on test audio file...")
            optimal_chunk_size = benchmark_chunk_sizes(
                TEST_AUDIO_PATH,
                min_minutes=1.0,
                max_minutes=3.5,
                step_minutes=0.5
            )
            print(f"Benchmark complete. Optimal chunk size: {optimal_chunk_size:.1f} minutes")
            
        elif args.serve:
            # If a specific chunk size is provided, use it
            if args.chunk_size:
                app.config['OPTIMAL_CHUNK_SIZE'] = args.chunk_size
                print(f"Starting API server with specified chunk size of {args.chunk_size:.1f} minutes")
            else:
                # Use a default chunk size if no benchmark has been run
                app.config['OPTIMAL_CHUNK_SIZE'] = 2.5
                print(f"Starting API server with default chunk size of 2.5 minutes")
                print("For optimal performance, first run with --test flag to find the best chunk size")
            
            # Start Flask app
            app.run(host='0.0.0.0', port=args.port)
        
    except Exception as e:
        print(f"Error during startup: {e}") 