# Base image
FROM pytorch/pytorch

# Set up workspace
WORKDIR /workspace
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt
