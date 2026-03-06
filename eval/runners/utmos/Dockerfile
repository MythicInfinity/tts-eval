FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
ENV HOME=/home/app
ENV XDG_CACHE_HOME=/home/app/.cache
ENV HF_HOME=/home/app/.cache/huggingface
ENV TORCH_HOME=/home/app/.cache/torch
ENV UTMOSV2_CHACHE=/home/app/.cache/utmosv2

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git libsndfile1 wget \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/app/.cache /app/data \
    && chmod -R 0777 /home/app /app/data

RUN pip install --no-cache-dir \
    pandas==2.2.3 \
    torchaudio==2.5.1 \
    soundfile==0.13.1 \
    tqdm==4.67.1 \
    transformers==4.46.3 \
    git+https://github.com/sarulab-speech/UTMOSv2.git@e53a6762948b908105d48d6cfd453f1b58156ed0
