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
        git \
        git-lfs \
        g++ \
        gfortran \
        libsndfile1 \
        libtool \
        sox \
        subversion \
        unzip \
    && git lfs install --system \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/app/.cache /app/data \
    && chmod -R 0777 /home/app /app/data

RUN printf "setuptools<81\n" > /tmp/pip-constraints.txt \
    && python -m pip install --no-cache-dir "setuptools<81" wheel "numpy<2" \
    && PIP_CONSTRAINT=/tmp/pip-constraints.txt python -m pip install --no-cache-dir "ttsds==2.1.1" \
    && rm -f /tmp/pip-constraints.txt
