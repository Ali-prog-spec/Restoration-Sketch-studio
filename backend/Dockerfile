# Backend: FastAPI + ONNX Runtime (CPU). Build context = repository root (see docker-compose.yml).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MODEL_DIR=/app/models \
    SAMPLES_DIR=/app/backend/samples \
    MAX_UPLOAD_MB=10

WORKDIR /app
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Only the torch-free shared code is needed: corruption definitions + image preprocessing.
COPY src/__init__.py /app/src/__init__.py
COPY src/corruptions /app/src/corruptions
COPY src/utils/__init__.py src/utils/image_io.py /app/src/utils/
COPY backend /app/backend

# Model files are NOT baked into the image; they are mounted from ./models (see models/README.md).
RUN mkdir -p /app/models && useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
