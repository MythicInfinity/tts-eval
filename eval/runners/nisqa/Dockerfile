FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
ENV HOME=/home/app
ENV XDG_CACHE_HOME=/home/app/.cache
ENV HF_HOME=/home/app/.cache/huggingface
ENV TORCH_HOME=/home/app/.cache/torch
ENV TORCHMETRICS_CACHE=/home/app/.torchmetrics

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/app/.cache /home/app/.torchmetrics /app/data \
    && chmod -R 0777 /home/app /app/data

RUN pip install --no-cache-dir \
    torchmetrics==1.8.2 \
    torchaudio==2.5.1 \
    librosa==0.11.0 \
    requests==2.32.5 \
    soundfile==0.13.1 \
    tqdm==4.67.1
