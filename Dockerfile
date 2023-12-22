# Base image
FROM python:3

# Set up workspace
WORKDIR /workspace

# Install dependencies
RUN pip install -r requirements.txt
