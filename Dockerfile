#--- Sem usar model/
# FROM python:3.11.9-slim
# WORKDIR /app
# COPY . .
# RUN pip install -r requirements.txt

#--- Usando model/
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1 

WORKDIR /app

# Instala dependências do sistema que bibliotecas de imagem (como OpenCV) costumam exigir
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

