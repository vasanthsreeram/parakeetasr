import argparse
import time
import nemo.collections.asr as nemo_asr
import torch
import os
import gradio as gr # Added Gradio

# Global variable for the model
MODEL = None
MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"

def load_asr_model():
    """Loads the ASR model."""
    global MODEL
    if MODEL is None:
        print(f"Loading {MODEL_NAME} model (this may take a while)...")
        start_time = time.time()
        # Check for CUDA availability
        if torch.cuda.is_available():
            print("CUDA is available. Model will run on GPU.")
            # You can also specify the device explicitly if needed, e.g., map_location=torch.device('cuda')
            # However, NeMo models usually handle this automatically.
        else:
            print("CUDA not available. Model will run on CPU (this might be very slow).")
        
        MODEL = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME)
        
        if torch.cuda.is_available() and hasattr(MODEL, 'cuda'):
             MODEL.cuda() # Ensure model is on GPU if available

        print(f"Model loaded in {time.time() - start_time:.2f} seconds.")
    return MODEL

def transcribe_raw_audio_for_gradio(audio_file_path: str, include_timestamps: bool = True):
    """
    Transcribes the given audio file using the pre-loaded ASR model.
    This function relies on NeMo's internal handling of the audio file.
    Modified to return strings suitable for Gradio output.
    """
    model = load_asr_model()
    if not audio_file_path or not os.path.exists(audio_file_path):
        return "Error: Audio file not provided or not found.", "", None

    print(f"Starting transcription for: {audio_file_path}")
    transcription_start_time = time.time()
    full_transcription_output = ""
    processing_info = ""
    # The third return (raw timestamps) is not directly used by this Gradio simple UI, but kept for consistency
    raw_timestamps_obj = None 

    try:
        # NeMo ASR model's transcribe method can take a list of audio file paths
        result = model.transcribe([audio_file_path], timestamps=include_timestamps)
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        full_transcription_output = f"Error during transcription: {str(e)}"
        processing_info = "Transcription failed."
        return full_transcription_output, processing_info, None
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA cache cleared after transcription attempt.")

    processing_time = time.time() - transcription_start_time
    processing_info = f"Processing time: {processing_time:.2f} seconds."
    print(f"Transcription completed in {processing_time:.2f} seconds.")

    if result and len(result) > 0:
        text = result[0].text
        raw_timestamps_obj = result[0].timestamp # Store for potential future use or detailed inspection
        
        full_transcription_output = text

        if include_timestamps and raw_timestamps_obj:
            word_ts = raw_timestamps_obj.get('word')
            segment_ts = raw_timestamps_obj.get('segment')

            if word_ts:
                full_transcription_output += "\n\n--- Word Timestamps ---\n" + "\n".join([
                    f"{ts['start']:.2f}s - {ts['end']:.2f}s: {ts['word']}" for ts in word_ts
                ])
            
            if segment_ts:
                full_transcription_output += "\n\n--- Segment Timestamps ---\n" + "\n".join([
                    f"{seg['start']:.2f}s - {seg['end']:.2f}s: {seg['segment']}" for seg in segment_ts
                ])
            
        return full_transcription_output, processing_info, raw_timestamps_obj # Return raw_timestamps_obj for completeness
    else:
        full_transcription_output = "Transcription failed or produced no output."
        return full_transcription_output, processing_info, None

# Create the Gradio interface
with gr.Blocks(title="Raw Parakeet ASR Transcription") as demo:
    gr.Markdown("# Raw Parakeet TDT 0.6B ASR Transcription")
    gr.Markdown("This demo uses NVIDIA's Parakeet TDT 0.6B model for speech recognition. It processes the entire file at once (no chunking).")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(type="filepath", label="Upload Audio File (.wav, .flac, etc.)")
            timestamps_checkbox = gr.Checkbox(label="Include Timestamps", value=True)
            transcribe_button = gr.Button("Transcribe Audio")
        
        with gr.Column():
            text_output = gr.Textbox(label="Transcription (with Timestamps if enabled)", lines=15, show_copy_button=True)
            info_output = gr.Textbox(label="Info")
    
    transcribe_button.click(
        fn=transcribe_raw_audio_for_gradio,
        inputs=[audio_input, timestamps_checkbox],
        # Outputting full transcription to text_output, and processing_info to info_output
        # The third output (raw_timestamps_obj) is not directly displayed in a separate component here.
        outputs=[text_output, info_output, gr.State()], 
    )
    
    gr.Markdown("### Important Notes:")
    gr.Markdown("- **No Chunking**: This version processes the entire audio file at once. Long files may lead to high memory usage or errors if they exceed GPU VRAM capacity.")
    gr.Markdown("- **`ffmpeg`**: While this script doesn't use `pydub` for chunking, `torchaudio` (used by NeMo for loading audio) might still rely on `ffmpeg` or `sox` being installed for handling various audio formats. Ensure you have `ffmpeg` for broader format support.")
    gr.Markdown("- **GPU VRAM**: Requires an NVIDIA GPU for reasonable speed. Processing very long files directly can consume significant VRAM.")
    gr.Markdown("- **Model Download**: The model (~600MB) is downloaded on first use if not already cached by NeMo.")

if __name__ == "__main__":
    try:
        print("Attempting to pre-load model...")
        load_asr_model() 
        print("Model pre-loading attempt complete.")
    except Exception as e:
        print(f"Warning: Could not pre-load model on startup: {e}")
        print("The model will attempt to load on the first transcription call if needed.")

    demo.launch(share=False, server_name="0.0.0.0") 