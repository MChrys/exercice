FROM python:3.9-slim

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y git ffmpeg

# Installer PyTorch pour Mac M1/M2
RUN pip install --no-cache-dir torch torchvision torchaudio

# Installer WhisperX
RUN pip install git+https://github.com/m-bain/whisperx.git

WORKDIR /app
COPY transcribe.py .

ENTRYPOINT ["python", "transcribe.py"]