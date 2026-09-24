FROM vllm/vllm-openai:v0.21.0

ENV DEBIAN_FRONTEND=noninteractive \
  PYTHONUNBUFFERED=1 \
  MINERU_MODEL_SMALL_BACKEND=torch \
  MINERU_MODEL_VLM_ENGINE=vllm \
  MINERU_MODEL_SOURCE=local \
  MINERU_API_MAX_CONCURRENT_REQUESTS=3 \
  MINERU_PROCESSING_WINDOW_SIZE=64 \
  MINERU_CONCURRENCY_PER_GPU=3 \
  MINERU_TIER=standard \
  MINERU_ROUTER_PORT=8002 \
  MINERU_ROUTER_STARTUP_TIMEOUT=1800

WORKDIR /app

RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  fonts-noto-core \
  fonts-noto-cjk \
  fontconfig \
  libgl1 \
  ca-certificates \
  curl && \
  fc-cache -fv && \
  rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir \
  "mineru[torch]==4.0.7" \
  "runpod>=1.7" \
  requests \
  httpx \
  boto3 \
  --break-system-packages

RUN mineru-kit models download \
  --tier standard \
  --small-backend torch \
  --vlm-engine vllm \
  --source huggingface && \
  mineru-kit models verify \
  --tier standard \
  --small-backend torch \
  --vlm-engine vllm

COPY handler.py /app/handler.py

ENTRYPOINT ["python3", "-u", "/app/handler.py"]
