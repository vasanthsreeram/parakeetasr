import os
import time
import tempfile
import gradio as gr
import requests
import nemo.collections.asr as nemo_asr
import torch
from pydub import AudioSegment # Added for audio chunking
import math # For ceiling function

# Global variables
MODEL = None
SAMPLE_RATE = 16000
# CHUNK_DURATION_MS: Max duration of audio chunks in milliseconds.
# Smaller chunks use less peak GPU memory during transcription but might slightly increase overhead.
# For GPUs with limited VRAM (e.g., 6-8GB), 5-10 minutes is a good starting point.
# The Parakeet model can handle up to ~24 mins, but that may require more VRAM.
CHUNK_DURATION_MS = 1.5 * 60 * 1000  # 1.5 minutes in milliseconds (NOTE: previous comment said 10 minutes, which was incorrect for this value)

def download_sample_audio():
    """Download a sample audio file for testing."""
    url = "https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav"
    output_path = os.path.join(tempfile.gettempdir(), "sample_audio.wav")
    
    if os.path.exists(output_path):
        return output_path
    
    print(f"Downloading sample audio from {url}...")
    response = requests.get(url)
    
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded sample to {output_path}")
        return output_path
    else:
        print(f"Failed to download sample: {response.status_code}")
        return None

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
        return "No audio file provided", None, None

    current_chunk_duration_ms = chunk_duration_minutes * 60 * 1000
    if current_chunk_duration_ms <= 0:
        return "Chunk duration must be positive.", None, None

    overall_processing_time = 0
    combined_text = ""
    combined_word_timestamps = []
    combined_segment_timestamps = []
    temp_chunk_files = []

    try:
        print(f"Loading audio file: {audio_path}")
        audio = AudioSegment.from_file(audio_path) # Loaded into CPU RAM by pydub
        duration_ms = len(audio)
        print(f"Audio duration: {duration_ms / 1000.0:.2f} seconds")

        if duration_ms <= current_chunk_duration_ms:
            print(f"Audio is short enough ({duration_ms/1000.0:.2f}s), processing as a single chunk.")
            start_time = time.time()
            # NeMo loads this chunk file to GPU for processing
            result = transcribe_audio_chunk(model, audio_path, include_timestamps)
            overall_processing_time = time.time() - start_time
            
            text = result[0].text
            segments_text = None
            if include_timestamps and result[0].timestamp:
                word_timestamps = result[0].timestamp.get('word', [])
                segment_timestamps = result[0].timestamp.get('segment', [])
                segments_text = ""
                for segment in segment_timestamps:
                    segments_text += f"{segment['start']:.2f}s - {segment['end']:.2f}s : {segment['segment']}\n"
                return text, segments_text, f"Processing time: {overall_processing_time:.2f} seconds (single chunk)"
            return text, None, f"Processing time: {overall_processing_time:.2f} seconds (single chunk)"

        else:
            num_chunks = math.ceil(duration_ms / current_chunk_duration_ms)
            print(f"Audio too long ({duration_ms/1000.0:.2f}s). Splitting into {num_chunks} chunks of max {current_chunk_duration_ms/1000.0:.2f}s each.")
            current_offset_s = 0

            for i in range(num_chunks):
                start_ms = i * current_chunk_duration_ms
                end_ms = min((i + 1) * current_chunk_duration_ms, duration_ms)
                print(f"Preparing chunk {i+1}/{num_chunks}: from {start_ms/1000.0:.2f}s to {end_ms/1000.0:.2f}s")
                # Slicing and exporting happens with data in CPU RAM / on disk
                chunk = audio[start_ms:end_ms]
                
                chunk_fd, chunk_path = tempfile.mkstemp(suffix=".wav")
                os.close(chunk_fd)
                chunk.export(chunk_path, format="wav") # Chunk saved to disk
                temp_chunk_files.append(chunk_path)
                
                print(f"Transcribing chunk {i+1}/{num_chunks} (file: {chunk_path})...")
                start_chunk_time = time.time()
                # NeMo loads this specific chunk file to GPU for processing
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
            
            segments_text_output = None
            if include_timestamps:
                segments_text_output = ""
                for segment in combined_segment_timestamps:
                    segments_text_output += f"{segment['start']:.2f}s - {segment['end']:.2f}s : {segment['segment']}\n"
            
            return combined_text.strip(), segments_text_output, f"Total processing time: {overall_processing_time:.2f} seconds ({num_chunks} chunks)"

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return f"Error: {str(e)}", None, None
    finally:
        print("Cleaning up temporary chunk files...")
        for temp_file in temp_chunk_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                print(f"Removed temp file: {temp_file}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA cache cleared after transcription attempt.")

def process_upload(audio, chunk_duration_minutes_ui, timestamps):
    """Process uploaded audio file."""
    if audio is None:
        return "No audio uploaded", None, None
    return transcribe_audio(audio, chunk_duration_minutes_ui, timestamps)

def process_microphone(audio, chunk_duration_minutes_ui, timestamps):
    """Process microphone input."""
    if audio is None:
        return "No audio recorded", None, None
    return transcribe_audio(audio, chunk_duration_minutes_ui, timestamps)

def process_sample(chunk_duration_minutes_ui, timestamps):
    """Process sample audio."""
    sample_path = download_sample_audio()
    if sample_path:
        return transcribe_audio(sample_path, chunk_duration_minutes_ui, timestamps)
    return "Failed to download sample audio", None, None

# Create the Gradio interface
with gr.Blocks(title="Parakeet ASR Transcription") as demo:
    gr.Markdown("# Parakeet TDT 0.6B ASR Transcription")
    gr.Markdown("This demo uses NVIDIA's Parakeet TDT 0.6B model for speech recognition. It supports accurate word-level timestamps, automatic punctuation, and capitalization.")
    
    chunk_duration_input = gr.Number(
        label="Max Chunk Duration (minutes)", 
        value=1.5,  # Defaulting to the current value in the code (1.5 minutes)
        minimum=0.5, 
        maximum=10000, # Parakeet can handle up to ~24 mins with enough VRAM
        step=0.5,
        info="Set the maximum duration for audio chunks. Smaller values use less peak GPU memory."
    )

    with gr.Tab("Upload Audio"):
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(type="filepath", label="Upload Audio File (.wav or .flac)")
                timestamps_checkbox = gr.Checkbox(label="Include Timestamps", value=True)
                upload_button = gr.Button("Transcribe Uploaded Audio")
            
            with gr.Column():
                text_output = gr.Textbox(label="Transcription", lines=5, show_copy_button=True)
                segments_output = gr.Textbox(label="Segments with Timestamps", lines=10, show_copy_button=True)
                info_output = gr.Textbox(label="Info")
        
        upload_button.click(
            fn=process_upload,
            inputs=[audio_input, chunk_duration_input, timestamps_checkbox],
            outputs=[text_output, segments_output, info_output],
        )
    
    with gr.Tab("Microphone"):
        with gr.Row():
            with gr.Column():
                mic_input = gr.Audio(type="filepath", label="Record Audio") 
                mic_timestamps_checkbox = gr.Checkbox(label="Include Timestamps", value=True)
                mic_button = gr.Button("Transcribe Recording")
            
            with gr.Column():
                mic_text_output = gr.Textbox(label="Transcription", lines=5, show_copy_button=True)
                mic_segments_output = gr.Textbox(label="Segments with Timestamps", lines=10, show_copy_button=True)
                mic_info_output = gr.Textbox(label="Info")
        
        mic_button.click(
            fn=process_microphone,
            inputs=[mic_input, chunk_duration_input, mic_timestamps_checkbox],
            outputs=[mic_text_output, mic_segments_output, mic_info_output],
        )
    
    with gr.Tab("Sample Audio"):
        with gr.Row():
            with gr.Column():
                sample_timestamps_checkbox = gr.Checkbox(label="Include Timestamps", value=True)
                sample_button = gr.Button("Transcribe Sample Audio")
            
            with gr.Column():
                sample_text_output = gr.Textbox(label="Transcription", lines=5, show_copy_button=True)
                sample_segments_output = gr.Textbox(label="Segments with Timestamps", lines=10, show_copy_button=True)
                sample_info_output = gr.Textbox(label="Info")
        
        sample_button.click(
            fn=process_sample,
            inputs=[chunk_duration_input, sample_timestamps_checkbox],
            outputs=[sample_text_output, sample_segments_output, sample_info_output],
        )
    
    gr.Markdown("### Important Notes for GPU Usage & Large Files:")
    gr.Markdown("- **Chunking**: Long audio files are automatically split into smaller chunks for processing. You can set the maximum chunk duration above. This helps manage GPU memory.")
    gr.Markdown("- **`ffmpeg`**: Audio processing (like chunking) requires `ffmpeg`. If not installed, you might encounter errors. (e.g., `sudo apt install ffmpeg` on Debian/Ubuntu).")
    gr.Markdown("- **GPU VRAM**: Requires an NVIDIA GPU. While the model is 0.6B parameters, processing audio (especially longer chunks) needs significant VRAM (2GB is an absolute minimum, 6GB+ recommended). The default 10-minute chunk size is a balance; for very limited VRAM, you might need to edit `CHUNK_DURATION_MS` in the script to be even smaller (e.g., 5 minutes).")
    gr.Markdown("  2. **Reduce Chunk Size**: Use the input field above to set a smaller value (e.g., 5 minutes or less). ")
    gr.Markdown("  3. **Close Other GPU Apps**: Ensure other applications are not consuming VRAM.")
    gr.Markdown("- **`cuda-python`**: For potentially better performance with NeMo, install `cuda-python>=12.3` (`pip install cuda-python>=12.3`).")
    gr.Markdown("- **Model Download**: The model (~600MB) is downloaded on first use.")

if __name__ == "__main__":
    try:
        load_model()
    except Exception as e:
        print(f"Warning: Could not pre-load model: {e}")
        print("The model will be loaded on first transcription request.")
    
    demo.launch(share=False, server_name="0.0.0.0") 