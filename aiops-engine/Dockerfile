FROM python:3.11-slim

WORKDIR /aiops-engine

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pipeline.py .
COPY model_monitor.py .
COPY detect_and_trigger_retrain.py .
COPY auto_retrain_pipeline.py .
COPY mlops_status_exporter.py .

# Create data and models directories
RUN mkdir -p /data /aiops-engine/models /shared

# Copy pre-trained model into image (populated at build time)
COPY models/ /aiops-engine/models/

# Ensure kubectl is available for job triggering
RUN apt-get update && apt-get install -y kubectl && rm -rf /var/lib/apt/lists/*

EXPOSE 8000

# Default to pipeline, but can run other scripts
CMD ["python", "pipeline.py"]
