import argparse
import time
import nemo.collections.asr as nemo_asr
import torch
import os

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

def transcribe_raw_audio(audio_file_path: str, include_timestamps: bool = True):
    """
    Transcribes the given audio file using the pre-loaded ASR model.
    This function relies on NeMo's internal handling of the audio file.
    """
    model = load_asr_model()
    if not os.path.exists(audio_file_path):
        return f"Error: Audio file not found at {audio_file_path}", None, None

    print(f"Starting transcription for: {audio_file_path}")
    transcription_start_time = time.time()

    try:
        # NeMo ASR model's transcribe method can take a list of audio file paths
        # It handles loading and processing.
        # For Parakeet-TDT, timestamps are usually enabled by default if the model supports it.
        # The `timestamps` parameter in `transcribe` can be 'word', 'segment', or True (for both if available)
        # or False.
        result = model.transcribe([audio_file_path], timestamps=include_timestamps)
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return f"Error during transcription: {str(e)}", None, None
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA cache cleared after transcription attempt.")


    processing_time = time.time() - transcription_start_time
    print(f"Transcription completed in {processing_time:.2f} seconds.")

    if result and len(result) > 0:
        text = result[0].text
        word_timestamps_str = None
        segment_timestamps_str = None

        if include_timestamps and result[0].timestamp:
            word_ts = result[0].timestamp.get('word')
            segment_ts = result[0].timestamp.get('segment')

            if word_ts:
                word_timestamps_str = "\nWord Timestamps:\n" + "\n".join([f"{ts['start']:.2f}s - {ts['end']:.2f}s: {ts['word']}" for ts in word_ts])
            
            if segment_ts:
                segment_timestamps_str = "\nSegment Timestamps:\n" + "\n".join([f"{seg['start']:.2f}s - {seg['end']:.2f}s: {seg['segment']}" for seg in segment_ts])
        
        full_transcript_info = text
        if word_timestamps_str:
            full_transcript_info += word_timestamps_str
        if segment_timestamps_str:
            full_transcript_info += segment_timestamps_str
            
        return full_transcript_info, f"Processing time: {processing_time:.2f} seconds", result[0].timestamp
    else:
        return "Transcription failed or produced no output.", f"Processing time: {processing_time:.2f} seconds", None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe an audio file using NVIDIA Parakeet ASR model (raw, no chunking).")
    parser.add_argument("audio_file", help="Path to the audio file to transcribe (e.g., .wav, .flac).")
    parser.add_argument("--no-timestamps", action="store_false", dest="include_timestamps", help="Disable word and segment timestamps.")
    parser.set_defaults(include_timestamps=True)

    args = parser.parse_args()

    try:
        # Attempt to pre-load the model
        load_asr_model() 
    except Exception as e:
        print(f"Warning: Could not pre-load model on startup: {e}")
        print("The model will attempt to load on the first transcription call if needed.")

    transcription, info, _ = transcribe_raw_audio(args.audio_file, args.include_timestamps)
    
    print("\n--- Transcription Result ---")
    if transcription:
        print(transcription)
    if info:
        print(f"Info: {info}")
    
    print("--------------------------") 