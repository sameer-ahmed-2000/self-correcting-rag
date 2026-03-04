# app/api/ingest.py
"""
Ingest Wikipedia articles into the FAISS vector store.
Chunks articles into ~300-word paragraphs with overlap.
"""
import re
import io
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
import wikipedia
from app.core.vector_store import add_documents

router = APIRouter()

DEFAULT_TOPICS = [
    "Artificial intelligence",
    "Machine learning",
    "Large language model",
    "Retrieval-augmented generation",
    "Natural language processing",
    "Vector database",
    "Transformer (deep learning architecture)",
    "BERT (language model)",
    "GPT-4",
    "Information retrieval",
]

CHUNK_SIZE = 300        # words per chunk
CHUNK_OVERLAP = 50      # word overlap between chunks


def chunk_text(text: str, title: str, source: str) -> list[dict]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append({"text": chunk_text, "title": title, "source": source})
        if end == len(words):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


class IngestRequest(BaseModel):
    topics: list[str] | None = None   # defaults to DEFAULT_TOPICS

class UrlIngestRequest(BaseModel):
    urls: list[str]


class IngestResponse(BaseModel):
    status: str
    topics_ingested: list[str]
    chunks_added: int
    errors: list[str]


@router.post("/wikipedia", response_model=IngestResponse)
def ingest_wikipedia(req: IngestRequest = IngestRequest()):
    topics = req.topics or DEFAULT_TOPICS
    all_chunks = []
    errors = []
    ingested = []

    for topic in topics:
        try:
            page = wikipedia.page(topic, auto_suggest=False)
            chunks = chunk_text(page.content, title=page.title, source=page.url)
            all_chunks.extend(chunks)
            ingested.append(topic)
            print(f"  ✓ {topic} → {len(chunks)} chunks")
        except wikipedia.exceptions.DisambiguationError as e:
            # Try the first option
            try:
                page = wikipedia.page(e.options[0], auto_suggest=False)
                chunks = chunk_text(page.content, title=page.title, source=page.url)
                all_chunks.extend(chunks)
                ingested.append(topic)
            except Exception as inner:
                errors.append(f"{topic}: {str(inner)}")
        except Exception as e:
            errors.append(f"{topic}: {str(e)}")

    if all_chunks:
        add_documents(all_chunks)

    return {
        "status": "success" if not errors else "partial",
        "topics_ingested": ingested,
        "chunks_added": len(all_chunks),
        "errors": errors,
    }


@router.post("/url", response_model=IngestResponse)
def ingest_url(req: UrlIngestRequest):
    all_chunks = []
    errors = []
    ingested = []

    for url in req.urls:
        try:
            response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            # Remove nav, footer, script, style noise
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            text = soup.get_text(separator=' ', strip=True)
            # Collapse whitespace and cap to 50k chars (≈ ~7,000 words)
            import re as _re
            text = _re.sub(r'\s+', ' ', text).strip()[:50_000]
            title = soup.title.string.strip() if soup.title and soup.title.string else url
            chunks = chunk_text(text, title=title, source=url)
            all_chunks.extend(chunks)
            ingested.append(url)
            print(f"  ✓ {url} → {len(chunks)} chunks")
        except Exception as e:
            errors.append(f"{url}: {str(e)}")

    if all_chunks:
        add_documents(all_chunks)

    return {
        "status": "success" if not errors else "partial",
        "topics_ingested": ingested,
        "chunks_added": len(all_chunks),
        "errors": errors,
    }


@router.post("/pdf", response_model=IngestResponse)
async def ingest_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    all_chunks = []
    errors = []
    ingested = []

    try:
        content = await file.read()
        pdf = PdfReader(io.BytesIO(content))
        text = ""
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        
        title = file.filename
        chunks = chunk_text(text, title=title, source=f"local-pdf:{title}")
        all_chunks.extend(chunks)
        ingested.append(title)
        print(f"  ✓ {title} → {len(chunks)} chunks")
    except Exception as e:
        errors.append(f"{file.filename}: {str(e)}")

    if all_chunks:
        add_documents(all_chunks)

    return {
        "status": "success" if not errors else "partial",
        "topics_ingested": ingested,
        "chunks_added": len(all_chunks),
        "errors": errors,
    }
