FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torchaudio==2.5.1 soundfile==0.12.1
