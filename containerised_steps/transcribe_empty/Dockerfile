FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    tree \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y git ffmpeg


RUN pip install --no-cache-dir torch torchvision torchaudio


RUN pip install git+https://github.com/m-bain/whisperx.git

WORKDIR /app


CMD ["ls", "-R", "/app"] 