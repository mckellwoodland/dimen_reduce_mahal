# Base image
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Set up workspace
WORKDIR /workspace
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt
