FROM python:3.9-slim-bookworm

WORKDIR /usr/src/

COPY src/backend/requirements.txt ./

RUN mkdir /usr/src/logs
RUN mkdir -p chroma_db && chmod a+rwx -R chroma_db

RUN apt-get update && apt-get install -y \
    build-essential cmake pkg-config \
    portaudio19-dev libportaudio2 sox libsox-fmt-all \
    alsa-utils \
    libvoikko1 voikko-fi libvoikko-dev \
    && rm -rf /var/lib/apt/lists/*

ENV CMAKE_BUILD_PARALLEL_LEVEL=1
ENV CMAKE_ARGS="-DGGML_NATIVE=OFF"
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && pip check

COPY src/backend/ ./
COPY download_models.py ./

CMD ["python", "app.py", "--cli"]
