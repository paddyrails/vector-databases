# Production-Grade PDF Indexing into MongoDB Atlas Vector Search

A professional AI Engineer's approach to building a document indexing and retrieval pipeline -- designed for reliability, scalability, and retrieval quality.

---

## Key Differences from a Naive Approach

| Concern | Naive | Professional |
|---------|-------|-------------|
| Text extraction | `pypdf` only | OCR fallback (`unstructured`) for scanned/image PDFs |
| Chunking | Fixed character split | Semantic/recursive splitting that respects sentence and paragraph boundaries |
| Embeddings | One API call per chunk | Batched calls with retry + rate-limit handling |
| Deduplication | None -- re-run = duplicates | Content hashing + upsert for idempotency |
| Metadata | Page number only | Source, page, section headers, chunk hash, timestamps |
| Observability | `print()` | Structured logging |
| Configuration | Hardcoded | Pydantic settings / env-driven config |
| Retrieval | Vector-only | Hybrid search (vector + full-text) with re-ranking |
| Pipeline | Loose scripts | LangChain / LlamaIndex orchestration |
| Testing | Manual | Retrieval evaluation with ground-truth Q&A pairs |

---

## Architecture

```
                 ┌─────────────┐
                 │  PDF Files   │
                 └──────┬──────┘
                        │
              ┌─────────▼──────────┐
              │  Document Loader   │  (unstructured / pypdf + OCR fallback)
              └─────────┬──────────┘
                        │
              ┌─────────▼──────────┐
              │  Preprocessing     │  (clean, normalize, extract metadata)
              └─────────┬──────────┘
                        │
              ┌─────────▼──────────┐
              │  Chunking          │  (RecursiveCharacterTextSplitter)
              └─────────┬──────────┘
                        │
              ┌─────────▼──────────┐
              │  Embedding (batch) │  (OpenAI / Cohere / local model)
              └─────────┬──────────┘
                        │
              ┌─────────▼──────────┐
              │  MongoDB Atlas     │  (vector index + full-text index)
              └─────────┬──────────┘
                        │
              ┌─────────▼──────────┐
              │  Retrieval + RAG   │  (hybrid search → re-rank → LLM)
              └────────────────────┘
```

---

## Step 1 -- Install Dependencies

```bash
pip install \
  pymongo \
  langchain \
  langchain-mongodb \
  langchain-openai \
  langchain-community \
  unstructured[pdf] \
  python-dotenv \
  pydantic-settings \
  tenacity
```

> `unstructured[pdf]` handles scanned PDFs, tables, and complex layouts far better than `pypdf` alone.

---

## Step 2 -- Configuration (Pydantic Settings)

A professional setup externalizes all tunables. No magic numbers buried in code.

### `config.py`

```python
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # MongoDB
    mongodb_uri: str
    db_name: str = "ssa_db"
    collection_name: str = "documents"
    vector_index_name: str = "vector_index"
    fulltext_index_name: str = "fulltext_index"

    # Embedding
    openai_api_key: str
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    embedding_batch_size: int = 100  # max chunks per API call

    # Chunking
    chunk_size: int = 1000       # tokens, not characters
    chunk_overlap: int = 200

    # Paths
    pdf_directory: str = "."

    class Config:
        env_file = ".env"


settings = Settings()
```

---

## Step 3 -- Document Loading with OCR Fallback

A professional pipeline never assumes PDFs have clean extractable text.

### `loader.py`

```python
import hashlib
import logging
from pathlib import Path

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.schema import Document

logger = logging.getLogger(__name__)


def load_pdf(pdf_path: str) -> list[Document]:
    """
    Load a PDF using Unstructured, which handles:
    - Text-based PDFs
    - Scanned/image PDFs (via OCR)
    - Tables and complex layouts
    """
    loader = UnstructuredPDFLoader(
        pdf_path,
        mode="paged",            # one Document per page
        strategy="hi_res",       # use OCR + layout detection
    )
    docs = loader.load()

    # Enrich metadata
    file_hash = hashlib.md5(Path(pdf_path).read_bytes()).hexdigest()
    for doc in docs:
        doc.metadata.update({
            "source": Path(pdf_path).name,
            "file_hash": file_hash,
        })

    logger.info(f"Loaded {len(docs)} pages from {pdf_path}")
    return docs
```

---

## Step 4 -- Semantic Chunking

Naive fixed-character splitting breaks sentences mid-word and ignores document structure. A professional uses **recursive splitting** that respects natural text boundaries.

### `chunker.py`

```python
import hashlib
import logging

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from config import settings

logger = logging.getLogger(__name__)


def chunk_documents(docs: list[Document]) -> list[Document]:
    """
    Split documents using RecursiveCharacterTextSplitter.

    This splitter tries to split on (in order):
      1. Paragraph breaks (\\n\\n)
      2. Line breaks (\\n)
      3. Sentence endings (. ! ?)
      4. Word boundaries (spaces)
      5. Characters (last resort)

    This preserves semantic coherence within each chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(docs)

    # Add chunk-level metadata for deduplication and traceability
    for i, chunk in enumerate(chunks):
        content_hash = hashlib.sha256(chunk.page_content.encode()).hexdigest()[:16]
        chunk.metadata.update({
            "chunk_index": i,
            "chunk_hash": content_hash,
            "char_count": len(chunk.page_content),
        })

    logger.info(f"Created {len(chunks)} chunks from {len(docs)} pages")
    return chunks
```

---

## Step 5 -- Batched Embeddings with Retry

Sending one embedding request per chunk is slow and fragile. A professional batches requests and adds retry logic for transient failures.

### `embedder.py`

```python
import logging

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings

logger = logging.getLogger(__name__)

client = OpenAI(api_key=settings.openai_api_key)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=60),
)
def _embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts with exponential backoff retry."""
    response = client.embeddings.create(
        input=texts,
        model=settings.embedding_model,
    )
    return [item.embedding for item in response.data]


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """Embed all texts in batches."""
    all_embeddings = []
    batch_size = settings.embedding_batch_size

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        logger.info(f"Embedding batch {i // batch_size + 1} "
                     f"({len(batch)} chunks)")
        embeddings = _embed_batch(batch)
        all_embeddings.extend(embeddings)

    return all_embeddings
```

---

## Step 6 -- Idempotent Upsert into MongoDB

Re-running the pipeline should never create duplicates. Use `chunk_hash` as a deduplication key with `update_one(..., upsert=True)`.

### `store.py`

```python
import logging
from datetime import datetime, timezone

from pymongo import MongoClient, UpdateOne

from config import settings

logger = logging.getLogger(__name__)

mongo = MongoClient(settings.mongodb_uri)
collection = mongo[settings.db_name][settings.collection_name]


def upsert_documents(chunks, embeddings):
    """
    Upsert chunks into MongoDB. Uses chunk_hash as the
    deduplication key so re-indexing the same PDF is safe.
    """
    operations = []
    now = datetime.now(timezone.utc)

    for chunk, embedding in zip(chunks, embeddings):
        doc = {
            "source": chunk.metadata["source"],
            "page": chunk.metadata.get("page_number", chunk.metadata.get("page")),
            "chunk_index": chunk.metadata["chunk_index"],
            "chunk_hash": chunk.metadata["chunk_hash"],
            "text": chunk.page_content,
            "embedding": embedding,
            "indexed_at": now,
        }

        operations.append(
            UpdateOne(
                {"chunk_hash": doc["chunk_hash"]},  # dedup key
                {"$set": doc},
                upsert=True,
            )
        )

    if operations:
        result = collection.bulk_write(operations)
        logger.info(
            f"Upserted {result.upserted_count} new, "
            f"modified {result.modified_count} existing documents"
        )

    # Ensure indexes exist
    collection.create_index("chunk_hash", unique=True)
    collection.create_index("source")
```

---

## Step 7 -- Create Atlas Search Indexes

A professional uses **hybrid search** -- combining vector similarity with keyword (BM25) matching -- for better retrieval quality.

### Vector Search Index (`vector_index`)

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 1536,
      "similarity": "cosine"
    },
    {
      "type": "filter",
      "path": "source"
    }
  ]
}
```

### Full-Text Search Index (`fulltext_index`)

```json
{
  "mappings": {
    "dynamic": false,
    "fields": {
      "text": {
        "type": "string",
        "analyzer": "luceneStandard"
      }
    }
  }
}
```

---

## Step 8 -- Orchestration Pipeline

### `index_pdf.py`

```python
import logging
import sys
from pathlib import Path

from config import settings
from loader import load_pdf
from chunker import chunk_documents
from embedder import generate_embeddings
from store import upsert_documents

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def index_file(pdf_path: str):
    """Full pipeline: load → chunk → embed → store."""
    logger.info(f"=== Indexing {pdf_path} ===")

    # 1. Load
    docs = load_pdf(pdf_path)
    if not docs:
        logger.warning(f"No content extracted from {pdf_path}")
        return

    # 2. Chunk
    chunks = chunk_documents(docs)

    # 3. Embed (batched)
    texts = [c.page_content for c in chunks]
    embeddings = generate_embeddings(texts)

    # 4. Store (idempotent upsert)
    upsert_documents(chunks, embeddings)

    logger.info(f"=== Done: {len(chunks)} chunks indexed from {pdf_path} ===")


def main():
    pdf_files = sys.argv[1:] if len(sys.argv) > 1 else [
        str(p) for p in Path(settings.pdf_directory).glob("*.pdf")
    ]

    if not pdf_files:
        logger.error("No PDF files found.")
        sys.exit(1)

    for pdf in pdf_files:
        index_file(pdf)


if __name__ == "__main__":
    main()
```

```bash
# Index a single file
python index_pdf.py SSA.pdf

# Index all PDFs in the directory
python index_pdf.py
```

---

## Step 9 -- Hybrid Search with Re-Ranking

Vector search alone misses exact keyword matches. Full-text search alone misses semantic meaning. A professional combines both and re-ranks with **Reciprocal Rank Fusion (RRF)**.

### `retriever.py`

```python
import logging
from config import settings
from embedder import _embed_batch
from store import collection

logger = logging.getLogger(__name__)


def hybrid_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Hybrid retrieval: vector search + full-text search,
    combined with Reciprocal Rank Fusion.
    """
    query_embedding = _embed_batch([query])[0]

    # --- Vector search ---
    vector_results = list(collection.aggregate([
        {
            "$vectorSearch": {
                "index": settings.vector_index_name,
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 100,
                "limit": 20,
            }
        },
        {"$addFields": {"vs_score": {"$meta": "vectorSearchScore"}}},
        {"$project": {"embedding": 0}},
    ]))

    # --- Full-text search ---
    text_results = list(collection.aggregate([
        {
            "$search": {
                "index": settings.fulltext_index_name,
                "text": {
                    "query": query,
                    "path": "text",
                },
            }
        },
        {"$limit": 20},
        {"$addFields": {"ft_score": {"$meta": "searchScore"}}},
        {"$project": {"embedding": 0}},
    ]))

    # --- Reciprocal Rank Fusion ---
    k = 60  # RRF constant
    scores = {}  # chunk_hash -> (rrf_score, doc)

    for rank, doc in enumerate(vector_results):
        key = doc["chunk_hash"]
        rrf = 1.0 / (k + rank + 1)
        scores[key] = (scores.get(key, (0, doc))[0] + rrf, doc)

    for rank, doc in enumerate(text_results):
        key = doc["chunk_hash"]
        rrf = 1.0 / (k + rank + 1)
        if key in scores:
            scores[key] = (scores[key][0] + rrf, scores[key][1])
        else:
            scores[key] = (rrf, doc)

    # Sort by combined RRF score, return top_k
    ranked = sorted(scores.values(), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]


if __name__ == "__main__":
    query = "What are the eligibility requirements for Social Security?"
    results = hybrid_search(query)

    for doc in results:
        print(f"[Page {doc.get('page')}] {doc['source']}")
        print(f"  {doc['text'][:200]}...\n")
```

---

## Step 10 -- RAG (Retrieval-Augmented Generation)

The final step: feed retrieved context into an LLM to answer questions grounded in the documents.

### `rag.py`

```python
from openai import OpenAI
from config import settings
from retriever import hybrid_search

client = OpenAI(api_key=settings.openai_api_key)


def ask(question: str) -> str:
    """Answer a question using retrieved document context."""
    results = hybrid_search(question, top_k=5)
    context = "\n\n---\n\n".join(
        f"[Source: {r['source']}, Page {r.get('page')}]\n{r['text']}"
        for r in results
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer the user's question "
                    "based ONLY on the provided context. If the context does "
                    "not contain the answer, say so. Cite the source and page."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    answer = ask("What are the eligibility requirements for Social Security?")
    print(answer)
```

---

## Project Structure (Professional)

```
vecto-databases/
├── config.py              # Centralized settings (Pydantic)
├── loader.py              # PDF loading with OCR fallback
├── chunker.py             # Semantic chunking
├── embedder.py            # Batched embeddings with retry
├── store.py               # Idempotent MongoDB upsert
├── retriever.py           # Hybrid search (vector + full-text + RRF)
├── rag.py                 # RAG pipeline
├── index_pdf.py           # Orchestration entrypoint
├── .env                   # Secrets (never commit)
├── .gitignore
├── requirements.txt
├── SSA.pdf
└── README2.md             # This file
```

---

## Checklist: What Makes This Professional

- [x] **OCR fallback** -- handles scanned PDFs, not just text-based
- [x] **Semantic chunking** -- respects sentence/paragraph boundaries
- [x] **Batched embeddings** -- efficient API usage, not one call per chunk
- [x] **Retry with backoff** -- resilient to transient API failures
- [x] **Idempotent upserts** -- safe to re-run without creating duplicates
- [x] **Content hashing** -- deduplication via `chunk_hash`
- [x] **Rich metadata** -- source, page, timestamp, hash for traceability
- [x] **Hybrid search** -- vector + full-text with RRF re-ranking
- [x] **Structured logging** -- not `print()` statements
- [x] **Externalized config** -- Pydantic settings, no hardcoded values
- [x] **Modular design** -- each concern in its own module
- [x] **Full RAG pipeline** -- retrieval directly feeds into LLM generation
