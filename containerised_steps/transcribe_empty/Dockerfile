FROM python:3.12-slim


RUN apt-get update && apt-get install -y \
    tree \
    git \
    ffmpeg \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*


RUN pip install --upgrade pip && pip install poetry


RUN poetry config virtualenvs.create false


COPY pyproject.toml poetry.lock* poetry.toml ./


RUN pip install --no-cache-dir torch torchvision torchaudio
RUN pip install git+https://github.com/m-bain/whisperx.git
RUN pip install varname


RUN poetry install --no-dev --no-root

WORKDIR /app

RUN pip install git+https://github.com/m-bain/whisperx.git



#RUN poetry config virtualenvs.create false \
#    && poetry install --no-interaction --no-ansi
#COPY entrypoint.sh /entrypoint.sh
#RUN chmod +x /entrypoint.sh


CMD ["tree"]