FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/model_cache

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/model_cache && chmod 777 /app/model_cache
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('all-MiniLM-L6-v2', device='cpu')"

COPY . .

RUN mkdir -p uploads && chmod 777 uploads
RUN mkdir -p /app/chroma_db && chmod 777 /app/chroma_db

EXPOSE 7860

CMD gunicorn app:app --bind 0.0.0.0:${PORT:-7860} --workers 1 --threads 2 --timeout 300
