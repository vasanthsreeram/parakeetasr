# Setting Up a VM for Parakeet ASR API Server

This guide will help you set up a virtual machine (VM) with the correct environment to run the Parakeet ASR API server.

## Option 1: Setting Up on an Existing Linux VM

If you already have a Linux VM with an NVIDIA GPU:

1. Make sure NVIDIA drivers are installed:
   ```bash
   nvidia-smi
   ```
   If this command fails, install NVIDIA drivers for your specific GPU.

2. Clone this repository:
   ```bash
   git clone <your-repo-url>
   cd parakeetasr
   ```

3. Run the setup script:
   ```bash
   ./setup.sh
   ```
   This will:
   - Check for Python 3.8 or 3.9
   - Install it if not present (with your permission)
   - Create a virtual environment
   - Install the necessary dependencies

4. Run the server:
   ```bash
   ./run.sh
   ```

## Option 2: Creating a New VM with Google Cloud Platform

If you need to create a new VM with an NVIDIA GPU:

1. **Create a Google Cloud Platform (GCP) account** if you don't have one.

2. **Open the GCP Console** and go to Compute Engine > VM instances.

3. **Create a new VM instance**:
   - Click "Create Instance"
   - Choose a name for your VM
   - Select a region and zone close to you
   - Under "Machine Configuration":
     - Choose "GPU" tab
     - Select a GPU type (e.g., NVIDIA T4)
     - Select a machine type with at least 8 vCPUs and 16GB memory
   - Under "Boot disk":
     - Click "Change"
     - Select "Deep Learning on Linux" as the operating system
     - Choose "Deep Learning VM" as the version
     - Set disk size to at least 50GB
   - Expand "Advanced options" > "Networking":
     - Check "Enable HTTP traffic" and "Enable HTTPS traffic"
   - Click "Create"

4. **SSH into your VM**:
   - Click the SSH button next to your VM in the GCP Console

5. **Clone this repository**:
   ```bash
   git clone <your-repo-url>
   cd parakeetasr
   ```

6. **Run the setup script**:
   ```bash
   ./setup.sh
   ```

7. **Run the server**:
   ```bash
   ./run.sh
   ```

## Option 3: Using Docker

If you prefer using Docker:

1. **Install Docker and NVIDIA Docker runtime**:
   ```bash
   # Install Docker
   curl -fsSL https://get.docker.com -o get-docker.sh
   sh get-docker.sh
   
   # Install NVIDIA Docker runtime
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

2. **Build and run the Docker container**:
   ```bash
   docker build -t parakeet-asr-api .
   docker run --gpus all -p 5000:5000 parakeet-asr-api
   ```

## Testing the Server

Once your server is running, you can test it with the provided `test_client.py`:

```bash
# Activate the virtual environment if not already active
source venv/bin/activate

# Run the test client
python test_client.py
```

This will download a sample audio file and send it to the server for transcription.

## Accessing from Other Applications

Other applications can access your ASR service by sending HTTP requests to:

```
http://<your-vm-ip>:5000/transcribe
```

For example, using curl:

```bash
curl -X POST -F "audio_file=@path/to/your/audio.wav" -F "timestamps=true" http://<your-vm-ip>:5000/transcribe
```

Remember to open port 5000 in your firewall if accessing from outside the VM. 