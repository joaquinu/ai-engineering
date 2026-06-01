FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install minimal dependencies for postgres + sentence-transformers + fastapi
# (build context is llm-engineering/rag)
COPY requirements-pgvector.txt .
RUN pip install --no-cache-dir -r requirements-pgvector.txt

# Copy rag package source (install without dependencies; we've already installed what we need)
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir -e . --no-deps

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
