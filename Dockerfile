FROM vllm/vllm-openai:v0.21.0

ENV DEBIAN_FRONTEND=noninteractive

ENV MINERU_MODEL_SMALL_BACKEND=torch
ENV MINERU_MODEL_VLM_ENGINE=vllm
ENV MINERU_MODEL_SOURCE=local

WORKDIR /app

RUN apt-get update && \
  apt-get install -y \
  fonts-noto-core \
  fonts-noto-cjk \
  fontconfig \
  libgl1 \
  curl \
  ca-certificates && \
  fc-cache -fv && \
  rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir \
  "mineru[torch]>=4.0,<5" \
  runpod \
  requests \
  --break-system-packages

# Modelos usados por standard/advanced.
RUN mineru-kit models download \
  --tier standard \
  --small-backend torch \
  --vlm-engine vllm \
  --source huggingface

RUN mineru-kit models verify \
  --tier standard \
  --small-backend torch \
  --vlm-engine vllm

COPY handler.py /app/handler.py

ENTRYPOINT ["python3", "-u", "/app/handler.py"]
