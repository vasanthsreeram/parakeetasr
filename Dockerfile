FROM nvcr.io/nvidia/pytorch:22.12-py3

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application files
COPY app.py .
COPY test_client.py .
COPY README.md .

# Create directories for model cache
RUN mkdir -p /root/.cache/huggingface

# Make port 5000 available
EXPOSE 5000

# Run the application with Waitress WSGI server
CMD ["waitress-serve", "--host=0.0.0.0", "--port=5000", "app:app"] 