FROM python:3.10-slim

WORKDIR /mochi

RUN apt-get update && apt-get install -y git wget ffmpeg && rm -rf /var/lib/apt/lists/*

# Instala PyTorch + CUDA 12.1
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Dependências básicas
RUN pip install --no-cache-dir diffusers transformers accelerate

# Cria pastas
COPY src/ /mochi/src/
COPY handler.py /mochi/
COPY requirements.txt /mochi/

RUN pip install --no-cache-dir -r /mochi/requirements.txt

CMD ["python", "/mochi/handler.py"]
