FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 RAG_DATA_DIR=/app/data FASTEMBED_CACHE_PATH=/app/fastembed_cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Pre-cache the embedding model during build so runtime never blocks on downloads
RUN python -c "from fastembed import TextEmbedding; list(TextEmbedding(model_name='sentence-transformers/all-MiniLM-L6-v2', cache_dir='/app/fastembed_cache').embed(['warmup']))"
COPY app ./app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
