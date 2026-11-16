# The API. Models are downloaded on first start into the volume mounted at /cache (not into the image).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface

RUN useradd --create-home app \
    && mkdir -p /cache/huggingface /app/data/uploads \
    && chown -R app:app /cache /app

WORKDIR /app

COPY requirements.txt .
# The CPU-only build of PyTorch is much smaller than the default one.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY src ./src

USER app
EXPOSE 8000
# 0.0.0.0 is the container's own network; docker-compose publishes the port on 127.0.0.1 only.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
