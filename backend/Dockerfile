# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# Improve python runtime
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency file
COPY pyproject.toml ./

# Export dependencies and install
RUN uv export --no-dev --no-hashes > requirements.txt || true

RUN uv pip install --system --no-cache -r requirements.txt || \
    uv pip install --system --no-cache \
    fastapi uvicorn[standard] openai python-dotenv sentence-transformers \
    faiss-cpu numpy langgraph langchain langchain-community pydantic \
    wikipedia ddgs beautifulsoup4 requests pypdf python-multipart \
    datasets ragas slowapi

# Copy application
COPY app/ ./app/

# Create data directory (for FAISS index persistence)
RUN mkdir -p data

# Expose port
EXPOSE 8000

# Start FastAPI
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]