import requests
import time
import sys
import os

def load_model(server_url):
    """Send a request to load the model."""
    url = f"{server_url}/load_model"
    print("Loading model (this may take a while)...")
    response = requests.post(url)
    if response.status_code == 200:
        result = response.json()
        print(f"Model loaded in {result.get('load_time_seconds', 'N/A')} seconds")
        return True
    else:
        print(f"Failed to load model: {response.json().get('error', response.text)}")
        return False

def transcribe_audio(server_url, audio_path, include_timestamps=False):
    """Send an audio file for transcription."""
    if not os.path.exists(audio_path):
        print(f"Error: File not found: {audio_path}")
        return None
    
    url = f"{server_url}/transcribe"
    files = {'audio_file': open(audio_path, 'rb')}
    data = {'timestamps': str(include_timestamps).lower()}
    
    print(f"Sending file {audio_path} for transcription...")
    response = requests.post(url, files=files, data=data)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error transcribing: {response.json().get('error', response.text)}")
        return None

def check_health(server_url):
    """Check if the server is running and model is loaded."""
    url = f"{server_url}/health"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            result = response.json()
            print(f"Server status: {result.get('status')}")
            print(f"Model loaded: {result.get('model_loaded')}")
            return result.get('model_loaded', False)
        else:
            print(f"Error checking health: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("Connection error: Server not running")
        return False

def download_sample_audio():
    """Download a sample audio file for testing."""
    url = "https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav"
    output_path = "sample_audio.wav"
    
    if os.path.exists(output_path):
        print(f"Sample file already exists at {output_path}")
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

def main():
    server_url = "http://localhost:5000"
    
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
    else:
        print("No audio file provided, downloading sample...")
        audio_path = download_sample_audio()
        if not audio_path:
            return
    
    # Check if server is running
    if not check_health(server_url):
        print("Server is running but model is not loaded.")
        if not load_model(server_url):
            return
    
    # Transcribe with timestamps
    result = transcribe_audio(server_url, audio_path, include_timestamps=True)
    
    if result:
        print("\nTranscription:")
        print(result['text'])
        print(f"Processing time: {result['processing_time_seconds']:.2f} seconds")
        
        if 'segment_timestamps' in result:
            print("\nSegments with timestamps:")
            for segment in result['segment_timestamps']:
                print(f"{segment['start']:.2f}s - {segment['end']:.2f}s: {segment['segment']}")

if __name__ == "__main__":
    main() 