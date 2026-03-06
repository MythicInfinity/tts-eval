FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
ENV HOME=/home/app
ENV XDG_CACHE_HOME=/home/app/.cache
ENV HF_HOME=/home/app/.cache/huggingface
ENV TORCH_HOME=/home/app/.cache/torch

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        autoconf \
        automake \
        ffmpeg \
        gfortran \
        libsndfile1 \
        libtool \
        sox \
        subversion \
        unzip \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/app/.cache /app/data \
    && chmod -R 0777 /home/app /app/data

RUN pip install --no-cache-dir setuptools "numpy<2" "ttsds==2.1.1"
