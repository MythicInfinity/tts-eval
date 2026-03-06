FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
ENV HOME=/home/app
ENV XDG_CACHE_HOME=/home/app/.cache
ENV MPLCONFIGDIR=/home/app/.cache/matplotlib

WORKDIR /app

RUN mkdir -p /home/app/.cache /app/data \
    && chmod -R 0777 /home/app /app/data

RUN pip install --no-cache-dir \
    matplotlib==3.10.6 \
    numpy==2.3.3
