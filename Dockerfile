# 单镜像同时承载 API 与 UI 两个服务 —— 依赖完全一致，
# 由 docker-compose 用不同的 command 启动，避免维护两份几乎相同的 Dockerfile。
FROM python:3.12-slim

# PyMuPDF / RapidOCR / onnxruntime 需要的运行时库
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    # 模型缓存挂到卷上，重建镜像不必重新下载 bge-m3
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# 依赖单独一层：只有 requirements 变化时才重装
COPY requirements.txt constraints.txt ./
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

COPY src/ ./src/
COPY configs/ ./configs/
COPY eval/reports/ ./eval/reports/
COPY docs/ ./docs/

# data/ 与 .env 不进镜像：前者 292M 且随抓取更新，后者含密钥。
# 二者都在 compose 里以挂载/env_file 的方式在运行时注入。

EXPOSE 8000 8501

CMD ["uvicorn", "derivrag.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
