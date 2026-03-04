# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy dependency files
COPY pyproject.toml ./

# Install dependencies using uv
RUN uv pip install --system --no-cache -r <(uv export --no-dev --no-hashes 2>/dev/null || pip install . 2>/dev/null) || \
    uv pip install --system --no-cache \
    fastapi uvicorn[standard] openai python-dotenv sentence-transformers \
    faiss-cpu numpy langgraph langchain langchain-community pydantic \
    wikipedia ddgs beautifulsoup4 requests pypdf python-multipart \
    datasets ragas slowapi

# Copy app source
COPY app/ ./app/

# Create data directory for FAISS persistence
RUN mkdir -p data

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
