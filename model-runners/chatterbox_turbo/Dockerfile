FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
ENV HOME=/home/app
ENV XDG_CACHE_HOME=/home/app/.cache
ENV HF_HOME=/home/app/.cache/huggingface
ENV TORCH_HOME=/home/app/.cache/torch

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/app/.cache /app/data/inputs \
    && chmod -R 0777 /home/app /app/data

RUN pip install --no-cache-dir \
    chatterbox-tts==0.1.6 \
    datasets==3.6.0 \
    soundfile==0.12.1 \
    torchaudio==2.6.0
