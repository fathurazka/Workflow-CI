# Use official Python image as base
FROM python:3.12.7-slim

# Set working directory
WORKDIR /opt/ml/model

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy conda.yaml to get dependencies
COPY conda.yaml /opt/ml/model/

# Install Python dependencies
RUN pip install --no-cache-dir \
    mlflow==2.19.0 \
    cloudpickle==3.1.2 \
    numpy==2.4.0 \
    pandas==2.3.3 \
    psutil==7.2.0 \
    pyarrow==18.1.0 \
    scikit-learn==1.8.0 \
    scipy==1.16.3

# Copy the MLflow model artifacts
COPY . /opt/ml/model/

# Set environment variables
ENV MLFLOW_TRACKING_URI=""
ENV MLFLOW_MODEL_DIR=/opt/ml/model

# Expose port
EXPOSE 8080

# Command to run the MLflow model server
ENTRYPOINT ["mlflow", "models", "serve", "-m", "/opt/ml/model", "-h", "0.0.0.0", "-p", "8080", "--no-conda"]
