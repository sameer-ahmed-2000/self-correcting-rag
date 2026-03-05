FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

# copy only dependency file first
COPY pyproject.toml ./

RUN uv pip install --system --no-cache \
fastapi \
uvicorn[standard] \
openai \
python-dotenv \
sentence-transformers \
faiss-cpu \
numpy \
langgraph \
langchain \
langchain-community \
pydantic \
wikipedia \
ddgs \
beautifulsoup4 \
requests \
pypdf \
python-multipart \
datasets \
ragas \
slowapi

# copy app later
COPY app/ ./app/

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]