FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

# Set non-interactive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-venv git && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install torch with CUDA 12.6 support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install ultralytics
RUN pip install ultralytics

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# install ffmpeg with h.265
RUN apt-get update && \
    apt-get install -y ffmpeg libavcodec-extra && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app