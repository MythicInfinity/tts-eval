FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torchmetrics==1.9.0 \
    torchaudio==2.5.1 \
    librosa==0.11.0 \
    onnxruntime==1.22.1 \
    requests==2.32.5 \
    soundfile==0.13.1
