# Indexing Strategies for MongoDB Atlas Vector Database

A consolidated strategy guide synthesized from hands-on implementation (README.md, README2.md) and 40 arxiv papers (README3.md). Each strategy is presented with its rationale, when to use it, MongoDB-specific implementation, and complete working code.

---

## Table of Contents

1. [Strategy Overview: Which Strategy to Pick](#strategy-overview)
2. [Strategy 1: Naive Indexing (Baseline)](#strategy-1-naive-indexing)
3. [Strategy 2: Structure-Aware Chunking](#strategy-2-structure-aware-chunking)
4. [Strategy 3: Hierarchical Multi-Granularity Indexing](#strategy-3-hierarchical-multi-granularity-indexing)
5. [Strategy 4: Late Chunking (Embed-Then-Chunk)](#strategy-4-late-chunking)
6. [Strategy 5: Hybrid Dense+Sparse Indexing](#strategy-5-hybrid-densesparse-indexing)
7. [Strategy 6: Parent-Child Document Indexing](#strategy-6-parent-child-document-indexing)
8. [Strategy 7: Vision-Based Indexing (ColPali)](#strategy-7-vision-based-indexing)
9. [Strategy 8: Multi-Representation Indexing (Multi-Vector)](#strategy-8-multi-representation-indexing)
10. [Strategy 9: Hypothetical Question Indexing](#strategy-9-hypothetical-question-indexing)
11. [Strategy 10: Agentic Indexing with Metadata Enrichment](#strategy-10-agentic-indexing-with-metadata-enrichment)
12. [Strategy Comparison Matrix](#strategy-comparison-matrix)
13. [MongoDB Atlas Index Configurations](#mongodb-atlas-index-configurations)
14. [Choosing the Right Strategy for SSA.pdf](#choosing-the-right-strategy-for-ssapdf)

---

## Strategy Overview

```
                           RETRIEVAL QUALITY
                                 ▲
                                 │
          Strategy 10            │            Strategy 7
       (Agentic + Metadata)     │         (Vision/ColPali)
                                 │
          Strategy 8             │            Strategy 5
       (Multi-Vector)           │         (Hybrid Dense+Sparse)
                                 │
          Strategy 6             │            Strategy 4
       (Parent-Child)           │         (Late Chunking)
                                 │
          Strategy 3             │            Strategy 9
       (Hierarchical)           │         (Hypothetical Q)
                                 │
          Strategy 2             │
       (Structure-Aware)        │
                                 │
          Strategy 1             │
       (Naive Baseline)         │
                                 └──────────────────────────▶
                                      IMPLEMENTATION EFFORT
```

### Quick Decision Guide

| Your Situation | Recommended Strategy |
|---------------|---------------------|
| Proof of concept, demo, or learning | **Strategy 1** (Naive) |
| Text-heavy PDFs, production quality needed | **Strategy 2** (Structure-Aware) + **Strategy 5** (Hybrid) |
| Long documents where context spans multiple pages | **Strategy 4** (Late Chunking) or **Strategy 6** (Parent-Child) |
| Government/legal documents with sections and subsections | **Strategy 3** (Hierarchical) |
| Scanned PDFs, forms, tables, images | **Strategy 7** (Vision/ColPali) |
| Highest retrieval quality, budget available | **Strategy 8** (Multi-Vector) + **Strategy 10** (Agentic) |
| FAQ-style or compliance documents | **Strategy 9** (Hypothetical Questions) |

---

## Strategy 1: Naive Indexing

**The Baseline.** Split text into fixed-size chunks, embed each, store in MongoDB.

### When to Use
- Prototyping, learning, or quick demos
- Small document collections (< 50 pages)
- When speed of implementation matters more than retrieval quality

### How It Works

```
PDF ──▶ Extract text per page ──▶ Fixed-size split (500 chars) ──▶ Embed ──▶ MongoDB
```

### MongoDB Document Schema

```json
{
  "source": "SSA.pdf",
  "page": 5,
  "chunk_index": 2,
  "text": "The Social Security Administration provides...",
  "embedding": [0.012, -0.034, ...],  // 1536 dims
}
```

### Implementation

```python
from pypdf import PdfReader
from openai import OpenAI
from pymongo import MongoClient

client = OpenAI()
collection = MongoClient(MONGO_URI)["ssa_db"]["documents"]

def index_naive(pdf_path: str):
    reader = PdfReader(pdf_path)
    docs = []
    for page_num, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        # Fixed-size chunking -- simple but breaks mid-sentence
        for i in range(0, len(text), 400):
            chunk = text[i:i+500]
            if not chunk.strip():
                continue
            emb = client.embeddings.create(
                input=chunk, model="text-embedding-3-small"
            ).data[0].embedding
            docs.append({
                "source": pdf_path,
                "page": page_num,
                "chunk_index": i // 400,
                "text": chunk,
                "embedding": emb,
            })
    collection.insert_many(docs)
```

### Limitations
- Breaks sentences mid-word at chunk boundaries
- No deduplication -- re-running creates duplicates
- One API call per chunk (slow, expensive)
- Misses semantic context across chunk boundaries

**Research evidence:** Structure-unaware methods produce 2x more chunks for equivalent retrieval quality ([arXiv:2603.06976](https://arxiv.org/abs/2603.06976)).

---

## Strategy 2: Structure-Aware Chunking

**The practical default.** Chunk along natural document boundaries (paragraphs, sections, sentences) instead of arbitrary character counts.

### When to Use
- Any production system (this should be your minimum)
- Text-heavy documents with clear paragraph/section structure
- Government documents, reports, manuals

### How It Works

```
PDF ──▶ Layout-aware extraction ──▶ Recursive splitting on ¶/sentence boundaries
    ──▶ Metadata enrichment ──▶ Batch embed ──▶ Upsert to MongoDB
```

### Why It Works
Research shows Paragraph Group Chunking achieves the highest nDCG@5 (~0.459) across six domains with **half the chunks** of naive methods ([arXiv:2602.16974](https://arxiv.org/abs/2602.16974)). Recursive token-based chunking at 100 tokens consistently outperforms other approaches ([arXiv:2505.21700](https://arxiv.org/abs/2505.21700)).

### MongoDB Document Schema

```json
{
  "source": "SSA.pdf",
  "page": 5,
  "chunk_index": 12,
  "chunk_hash": "a3f8b2c1e9d04567",
  "text": "To be eligible for Social Security retirement benefits...",
  "embedding": [0.012, -0.034, ...],
  "indexed_at": "2026-04-18T10:30:00Z"
}
```

### Implementation

```python
import hashlib
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader
from openai import OpenAI
from pymongo import MongoClient, UpdateOne
from tenacity import retry, stop_after_attempt, wait_exponential

client = OpenAI()
collection = MongoClient(MONGO_URI)["ssa_db"]["documents"]

# --- Load with layout awareness ---
def load_pdf(path: str):
    loader = UnstructuredPDFLoader(path, mode="paged", strategy="hi_res")
    return loader.load()

# --- Chunk respecting natural boundaries ---
splitter = RecursiveCharacterTextSplitter(
    chunk_size=256,           # ~256 tokens
    chunk_overlap=50,         # ~20% overlap
    separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
)

# --- Batch embed with retry ---
@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=60))
def embed_batch(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(input=texts, model="text-embedding-3-small")
    return [d.embedding for d in resp.data]

# --- Idempotent upsert ---
def upsert(chunks, embeddings):
    ops = []
    for chunk, emb in zip(chunks, embeddings):
        content_hash = hashlib.sha256(chunk.page_content.encode()).hexdigest()[:16]
        ops.append(UpdateOne(
            {"chunk_hash": content_hash},
            {"$set": {
                "source": chunk.metadata.get("source", ""),
                "page": chunk.metadata.get("page_number", 0),
                "chunk_hash": content_hash,
                "text": chunk.page_content,
                "embedding": emb,
            }},
            upsert=True,
        ))
    collection.bulk_write(ops)

# --- Pipeline ---
def index_structured(pdf_path: str):
    docs = load_pdf(pdf_path)
    chunks = splitter.split_documents(docs)
    texts = [c.page_content for c in chunks]

    # Batch embed in groups of 100
    embeddings = []
    for i in range(0, len(texts), 100):
        embeddings.extend(embed_batch(texts[i:i+100]))

    upsert(chunks, embeddings)
```

---

## Strategy 3: Hierarchical Multi-Granularity Indexing

**Index the same content at multiple levels of detail.** Store document-level summaries, section-level chunks, and paragraph-level chunks in separate collections -- query the right granularity based on the question.

### When to Use
- Long documents (50+ pages) with clear section hierarchy
- Questions range from broad ("What is this document about?") to specific ("What is the income limit for SSI?")
- Government regulations, legal documents, technical manuals

### How It Works

```
                    ┌──────────────────────┐
                    │   Document Summary   │  Collection: summaries
                    │   (1 per document)   │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │   Section Chunks     │  Collection: sections
                    │   (1 per section)    │  ~500-1000 tokens each
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │   Paragraph Chunks   │  Collection: paragraphs
                    │   (fine-grained)     │  ~100-256 tokens each
                    └──────────────────────┘
```

### Why It Works
MultiDocFusion ([arXiv:2604.12352](https://arxiv.org/abs/2604.12352)) showed that hierarchical chunking outperforms flat chunking for complex documents. Broad questions are answered by document/section-level embeddings; specific questions by paragraph-level.

### MongoDB Document Schemas

**Level 1: Document summaries**
```json
{
  "source": "SSA.pdf",
  "level": "document",
  "summary": "This document covers Social Security Administration programs...",
  "embedding": [...]
}
```

**Level 2: Section chunks**
```json
{
  "source": "SSA.pdf",
  "level": "section",
  "section_title": "Chapter 3: Retirement Benefits",
  "pages": [15, 16, 17, 18],
  "text": "Full section text...",
  "embedding": [...]
}
```

**Level 3: Paragraph chunks**
```json
{
  "source": "SSA.pdf",
  "level": "paragraph",
  "section_title": "Chapter 3: Retirement Benefits",
  "page": 16,
  "text": "To qualify for retirement benefits, you must...",
  "embedding": [...]
}
```

### Implementation

```python
from openai import OpenAI

client = OpenAI()
db = MongoClient(MONGO_URI)["ssa_db"]

def index_hierarchical(pdf_path: str):
    docs = load_pdf(pdf_path)  # from Strategy 2

    # --- Level 3: Paragraph chunks (fine-grained) ---
    para_splitter = RecursiveCharacterTextSplitter(
        chunk_size=256, chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    paragraphs = para_splitter.split_documents(docs)
    para_texts = [p.page_content for p in paragraphs]
    para_embeddings = embed_batch_all(para_texts)

    for chunk, emb in zip(paragraphs, para_embeddings):
        db["paragraphs"].update_one(
            {"chunk_hash": hash_text(chunk.page_content)},
            {"$set": {
                "level": "paragraph",
                "source": pdf_path,
                "page": chunk.metadata.get("page_number"),
                "text": chunk.page_content,
                "embedding": emb,
            }},
            upsert=True,
        )

    # --- Level 2: Section chunks (medium-grained) ---
    sec_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100,
        separators=["\n\n\n", "\n\n", "\n"],
    )
    sections = sec_splitter.split_documents(docs)
    sec_texts = [s.page_content for s in sections]
    sec_embeddings = embed_batch_all(sec_texts)

    for chunk, emb in zip(sections, sec_embeddings):
        db["sections"].update_one(
            {"chunk_hash": hash_text(chunk.page_content)},
            {"$set": {
                "level": "section",
                "source": pdf_path,
                "text": chunk.page_content,
                "embedding": emb,
            }},
            upsert=True,
        )

    # --- Level 1: Document summary ---
    full_text = " ".join([d.page_content for d in docs])[:10000]
    summary = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": f"Summarize this document in 500 words:\n\n{full_text}",
        }],
    ).choices[0].message.content
    summary_emb = embed_batch([summary])[0]

    db["summaries"].update_one(
        {"source": pdf_path},
        {"$set": {
            "level": "document",
            "source": pdf_path,
            "summary": summary,
            "embedding": summary_emb,
        }},
        upsert=True,
    )


def query_hierarchical(question: str, level: str = "paragraph"):
    """Query the appropriate granularity."""
    q_emb = embed_batch([question])[0]
    return list(db[f"{level}s"].aggregate([
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": q_emb,
                "numCandidates": 100,
                "limit": 5,
            }
        },
        {"$project": {"embedding": 0, "score": {"$meta": "vectorSearchScore"}}},
    ]))
```

### Query Routing

```python
def route_query(question: str):
    """Pick the right granularity based on question type."""
    broad_signals = ["what is", "overview", "summarize", "about", "explain"]
    if any(s in question.lower() for s in broad_signals):
        return query_hierarchical(question, level="summary")

    medium_signals = ["section", "chapter", "compare", "differences"]
    if any(s in question.lower() for s in medium_signals):
        return query_hierarchical(question, level="section")

    return query_hierarchical(question, level="paragraph")
```

---

## Strategy 4: Late Chunking (Embed-Then-Chunk)

**Flip the pipeline.** Instead of chunk-then-embed, embed the full document through the transformer first, then chunk the token embeddings. Each chunk's embedding retains contextual information from the entire document.

### When to Use
- Context frequently spans chunk boundaries
- Pronouns, references ("this program", "as mentioned above") lose meaning when isolated
- Using a long-context embedding model (Jina, Nomic)

### How It Works

```
Traditional:  PDF ──▶ Chunk ──▶ Embed each chunk independently
Late Chunk:   PDF ──▶ Embed full doc tokens ──▶ Chunk token embeddings ──▶ Pool
```

### Why It Works
Late Chunking ([arXiv:2409.04701](https://arxiv.org/abs/2409.04701)) preserves cross-chunk context without additional training. A chunk that says "This program requires..." retains knowledge of *which* program from earlier in the document.

### Implementation

```python
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np

# Use a long-context embedding model
model_name = "jinaai/jina-embeddings-v2-base-en"  # 8192 token context
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def late_chunking_index(pdf_path: str, chunk_size: int = 256):
    docs = load_pdf(pdf_path)

    for doc in docs:
        text = doc.page_content

        # 1. Tokenize the full page
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=8192)

        # 2. Get all token embeddings from the transformer
        with torch.no_grad():
            outputs = model(**inputs)
        token_embeddings = outputs.last_hidden_state[0]  # (seq_len, hidden_dim)

        # 3. Split tokens into chunks
        tokens = tokenizer.tokenize(text)
        num_tokens = len(tokens)

        for start in range(0, num_tokens, chunk_size):
            end = min(start + chunk_size, num_tokens)

            # 4. Mean pool the token embeddings for this chunk
            chunk_emb = token_embeddings[start+1:end+1].mean(dim=0)  # +1 for [CLS]
            chunk_emb = chunk_emb.numpy().tolist()

            # 5. Reconstruct chunk text
            chunk_text = tokenizer.convert_tokens_to_string(tokens[start:end])

            collection.update_one(
                {"chunk_hash": hash_text(chunk_text)},
                {"$set": {
                    "source": pdf_path,
                    "page": doc.metadata.get("page_number"),
                    "text": chunk_text,
                    "embedding": chunk_emb,
                    "strategy": "late_chunking",
                }},
                upsert=True,
            )
```

### MongoDB Index for Late Chunking
```json
{
  "fields": [{
    "type": "vector",
    "path": "embedding",
    "numDimensions": 768,
    "similarity": "cosine"
  }]
}
```

> Note: Jina v2 produces 768-dim embeddings, not 1536. Update `numDimensions` accordingly.

---

## Strategy 5: Hybrid Dense+Sparse Indexing

**Index both vector embeddings and full-text content.** Query using both vector similarity *and* BM25 keyword matching, then merge results with Reciprocal Rank Fusion. This is the single highest-impact improvement over vector-only search.

### When to Use
- **Always.** Every production system should use hybrid search.
- Especially important when documents contain domain-specific terms, acronyms, proper nouns, or exact figures

### How It Works

```
                         ┌─────────────────┐
                    ┌───▶│  Vector Search   │───┐
                    │    │  ($vectorSearch) │   │
    Query ──▶ Embed │    └─────────────────┘   ├──▶ RRF Merge ──▶ Top K
                    │    ┌─────────────────┐   │
                    └───▶│  Full-Text Search│───┘
                         │  ($search BM25) │
                         └─────────────────┘
```

### Why It Works
"Blended RAG" ([arXiv:2404.07220](https://arxiv.org/abs/2404.07220)) showed hybrid retrieval consistently outperforms single-method. BGE-M3 ([arXiv:2410.20381](https://arxiv.org/abs/2410.20381)) produces both dense and sparse representations in one pass.

### MongoDB Setup: Two Indexes on One Collection

**Vector Search Index** (`vector_index`):
```json
{
  "fields": [
    {"type": "vector", "path": "embedding", "numDimensions": 1536, "similarity": "cosine"},
    {"type": "filter", "path": "source"}
  ]
}
```

**Atlas Full-Text Search Index** (`fulltext_index`):
```json
{
  "mappings": {
    "dynamic": false,
    "fields": {
      "text": {"type": "string", "analyzer": "luceneStandard"}
    }
  }
}
```

### Implementation

```python
from pymongo import MongoClient
from openai import OpenAI

client = OpenAI()
collection = MongoClient(MONGO_URI)["ssa_db"]["documents"]


def hybrid_search(query: str, top_k: int = 5) -> list[dict]:
    # 1. Embed the query
    q_emb = client.embeddings.create(
        input=query, model="text-embedding-3-small"
    ).data[0].embedding

    # 2. Vector search -- semantic similarity
    vector_results = list(collection.aggregate([
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": q_emb,
                "numCandidates": 150,  # 15x the final limit
                "limit": 20,
            }
        },
        {"$addFields": {"vs_score": {"$meta": "vectorSearchScore"}}},
        {"$project": {"embedding": 0}},
    ]))

    # 3. Full-text search -- keyword/BM25 matching
    text_results = list(collection.aggregate([
        {
            "$search": {
                "index": "fulltext_index",
                "text": {"query": query, "path": "text"},
            }
        },
        {"$limit": 20},
        {"$addFields": {"ft_score": {"$meta": "searchScore"}}},
        {"$project": {"embedding": 0}},
    ]))

    # 4. Reciprocal Rank Fusion (k=60)
    k = 60
    fused = {}
    for rank, doc in enumerate(vector_results):
        key = str(doc["_id"])
        fused[key] = {"score": 1.0 / (k + rank + 1), "doc": doc}

    for rank, doc in enumerate(text_results):
        key = str(doc["_id"])
        rrf = 1.0 / (k + rank + 1)
        if key in fused:
            fused[key]["score"] += rrf
        else:
            fused[key] = {"score": rrf, "doc": doc}

    # 5. Sort and return top_k
    ranked = sorted(fused.values(), key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in ranked[:top_k]]
```

### When Vector Wins vs When BM25 Wins

| Query Type | Winner | Example |
|-----------|--------|---------|
| Conceptual / semantic | Vector | "How do disability benefits work?" |
| Exact terms / acronyms | BM25 | "SSI income limit 2024" |
| Named entities | BM25 | "Title XVI Section 1611" |
| Paraphrased questions | Vector | "Can I get money if I can't work?" |
| Mixed | Hybrid (RRF) | "What is the SSI eligibility age requirement?" |

---

## Strategy 6: Parent-Child Document Indexing

**Embed small chunks for precision, but retrieve their parent (larger context) for the LLM.** This solves the chunk-size dilemma: small chunks match better, but large chunks provide better context for generation.

### When to Use
- Answers require surrounding context to make sense
- Small chunks match well but lose meaning in isolation
- Legal/regulatory documents where a single sentence references surrounding clauses

### How It Works

```
                    ┌──────────────────────┐
                    │    Parent Chunk      │  Stored in: parents collection
                    │    (1000 tokens)     │  Retrieved for LLM context
                    │                      │
                    │  ┌────┐ ┌────┐ ┌────┐│
                    │  │ C1 │ │ C2 │ │ C3 ││  Stored in: children collection
                    │  └────┘ └────┘ └────┘│  Embedded for search
                    └──────────────────────┘

    Search hits child C2 ──▶ Retrieve parent ──▶ Feed parent to LLM
```

### MongoDB Document Schemas

**Parent document:**
```json
{
  "_id": "parent_abc123",
  "source": "SSA.pdf",
  "pages": [5, 6],
  "text": "Full section text spanning 1000 tokens...",
}
```

**Child document (embedded for search):**
```json
{
  "_id": "child_xyz789",
  "parent_id": "parent_abc123",
  "source": "SSA.pdf",
  "page": 5,
  "text": "Small precise chunk of 100 tokens...",
  "embedding": [0.012, -0.034, ...]
}
```

### Implementation

```python
from bson import ObjectId

db = MongoClient(MONGO_URI)["ssa_db"]
parents = db["parents"]
children = db["children"]

def index_parent_child(pdf_path: str):
    docs = load_pdf(pdf_path)

    # Create large parent chunks
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100,
        separators=["\n\n\n", "\n\n"],
    )
    parent_chunks = parent_splitter.split_documents(docs)

    # For each parent, create small child chunks
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200, chunk_overlap=40,
        separators=["\n\n", "\n", ". ", " "],
    )

    for parent_chunk in parent_chunks:
        # Store parent (no embedding needed)
        parent_id = parents.insert_one({
            "source": pdf_path,
            "page": parent_chunk.metadata.get("page_number"),
            "text": parent_chunk.page_content,
        }).inserted_id

        # Create and embed children
        child_docs = child_splitter.create_documents(
            [parent_chunk.page_content],
            metadatas=[{"parent_id": str(parent_id)}],
        )

        child_texts = [c.page_content for c in child_docs]
        child_embeddings = embed_batch_all(child_texts)

        for child_doc, emb in zip(child_docs, child_embeddings):
            children.insert_one({
                "parent_id": str(parent_id),
                "source": pdf_path,
                "text": child_doc.page_content,
                "embedding": emb,
            })


def search_with_parent_context(query: str, top_k: int = 3):
    q_emb = embed_batch([query])[0]

    # Search children (precise matching)
    child_hits = list(children.aggregate([
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": q_emb,
                "numCandidates": 100,
                "limit": top_k,
            }
        },
        {"$project": {"embedding": 0, "score": {"$meta": "vectorSearchScore"}}},
    ]))

    # Retrieve parents (rich context for LLM)
    results = []
    for child in child_hits:
        parent = parents.find_one({"_id": ObjectId(child["parent_id"])})
        results.append({
            "matched_chunk": child["text"],
            "full_context": parent["text"],
            "page": parent.get("page"),
            "score": child["score"],
        })

    return results
```

---

## Strategy 7: Vision-Based Indexing (ColPali)

**Skip OCR entirely.** Embed document page images directly using a Vision Language Model. Captures tables, figures, layout, and text simultaneously.

### When to Use
- Scanned PDFs, poor-quality scans
- Documents with heavy table/figure content
- Forms, applications, infographics
- When OCR produces garbage

### How It Works

```
PDF ──▶ Render each page as image ──▶ VLM encodes image patches
    ──▶ Multi-vector embeddings per page ──▶ Store in MongoDB
```

### Why It Works
ColPali ([arXiv:2407.01449](https://arxiv.org/abs/2407.01449), ICLR 2025) uses PaliGemma to directly embed image patches, producing ColBERT-style multi-vector representations that capture both textual and visual information without any OCR step.

### Implementation

```python
import torch
from PIL import Image
from pdf2image import convert_from_path
from colpali_engine.models import ColPali, ColPaliProcessor

# Load ColPali model
model = ColPali.from_pretrained("vidore/colpali-v1.2").eval()
processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.2")

collection = MongoClient(MONGO_URI)["ssa_db"]["page_images"]

def index_vision(pdf_path: str):
    # Convert PDF pages to images
    images = convert_from_path(pdf_path, dpi=144)

    for page_num, image in enumerate(images, 1):
        # Encode page image directly -- no OCR
        inputs = processor(images=[image], return_tensors="pt")
        with torch.no_grad():
            page_embedding = model(**inputs).last_hidden_state.mean(dim=1)

        embedding = page_embedding[0].numpy().tolist()

        collection.update_one(
            {"source": pdf_path, "page": page_num},
            {"$set": {
                "source": pdf_path,
                "page": page_num,
                "embedding": embedding,
                "strategy": "colpali_vision",
            }},
            upsert=True,
        )

def search_vision(query: str, top_k: int = 3):
    # Encode query text
    inputs = processor(text=[query], return_tensors="pt")
    with torch.no_grad():
        q_emb = model(**inputs).last_hidden_state.mean(dim=1)

    return list(collection.aggregate([
        {
            "$vectorSearch": {
                "index": "vision_index",
                "path": "embedding",
                "queryVector": q_emb[0].numpy().tolist(),
                "numCandidates": 50,
                "limit": top_k,
            }
        },
        {"$project": {"embedding": 0, "score": {"$meta": "vectorSearchScore"}}},
    ]))
```

### Limitation
- Returns page-level results (not paragraph-level) -- less precise
- Requires GPU for reasonable speed
- ColPali embedding dimensions differ from text models -- needs its own vector index

---

## Strategy 8: Multi-Representation Indexing (Multi-Vector)

**Store multiple embeddings per chunk** -- each from a different model or representation. Query all representations and merge for maximum recall.

### When to Use
- Highest retrieval quality is non-negotiable
- Budget allows multiple embedding API calls
- Different query styles (keyword, semantic, question-format) target the same corpus

### How It Works

```
                                ┌── Dense embedding (OpenAI) ──────────┐
                                │                                       │
    Chunk ──▶ Generate ─────────┼── Dense embedding (BGE-M3) ──────────┼──▶ MongoDB
                                │                                       │    (one doc,
                                ├── Sparse embedding (BM25/SPLADE) ────┤     multiple
                                │                                       │     embedding
                                └── Summary embedding (LLM summary) ───┘     fields)
```

### MongoDB Document Schema

```json
{
  "source": "SSA.pdf",
  "page": 5,
  "text": "Original chunk text...",
  "summary": "LLM-generated summary of this chunk",
  "embedding_dense": [0.012, ...],       // OpenAI text-embedding-3-small (1536d)
  "embedding_bge": [0.034, ...],         // BGE-M3 dense (1024d)
  "embedding_summary": [0.056, ...]      // Embedding of the LLM summary (1536d)
}
```

### MongoDB Index Configuration (Multiple Vector Indexes)

```json
// Index 1: Dense embeddings
{
  "fields": [
    {"type": "vector", "path": "embedding_dense", "numDimensions": 1536, "similarity": "cosine"}
  ]
}

// Index 2: BGE-M3 embeddings
{
  "fields": [
    {"type": "vector", "path": "embedding_bge", "numDimensions": 1024, "similarity": "cosine"}
  ]
}

// Index 3: Summary embeddings
{
  "fields": [
    {"type": "vector", "path": "embedding_summary", "numDimensions": 1536, "similarity": "cosine"}
  ]
}
```

### Implementation

```python
from sentence_transformers import SentenceTransformer

openai_client = OpenAI()
bge_model = SentenceTransformer("BAAI/bge-m3")

def index_multi_vector(pdf_path: str):
    docs = load_pdf(pdf_path)
    chunks = splitter.split_documents(docs)

    for chunk in chunks:
        text = chunk.page_content

        # Representation 1: OpenAI dense embedding
        emb_dense = openai_client.embeddings.create(
            input=text, model="text-embedding-3-small"
        ).data[0].embedding

        # Representation 2: BGE-M3 dense embedding (local)
        emb_bge = bge_model.encode(text).tolist()

        # Representation 3: LLM summary + its embedding
        summary = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user",
                       "content": f"Summarize in one sentence:\n{text}"}],
        ).choices[0].message.content

        emb_summary = openai_client.embeddings.create(
            input=summary, model="text-embedding-3-small"
        ).data[0].embedding

        collection.update_one(
            {"chunk_hash": hash_text(text)},
            {"$set": {
                "source": pdf_path,
                "text": text,
                "summary": summary,
                "embedding_dense": emb_dense,
                "embedding_bge": emb_bge,
                "embedding_summary": emb_summary,
            }},
            upsert=True,
        )


def search_multi_vector(query: str, top_k: int = 5):
    q_dense = embed_openai(query)
    q_bge = bge_model.encode(query).tolist()

    # Search each index
    results_dense = vector_search(q_dense, "vector_index_dense", "embedding_dense")
    results_bge = vector_search(q_bge, "vector_index_bge", "embedding_bge")
    results_summary = vector_search(q_dense, "vector_index_summary", "embedding_summary")

    # RRF merge across all three
    return rrf_merge([results_dense, results_bge, results_summary], top_k=top_k)
```

---

## Strategy 9: Hypothetical Question Indexing

**Generate questions that each chunk answers, embed the questions, and store them alongside the chunk.** At query time, match user questions against the hypothetical questions -- question-to-question similarity is often stronger than question-to-passage.

### When to Use
- FAQ-style retrieval
- Compliance documents ("Does X policy allow Y?")
- User queries are always in question form
- Passages are dense/factual and hard to match semantically to questions

### How It Works

```
    Chunk: "SSI recipients must have income below $943/month..."
      │
      ├──▶ Q1: "What is the income limit for SSI?"
      ├──▶ Q2: "How much can I earn and still get SSI?"
      └──▶ Q3: "What is the maximum monthly income for SSI eligibility?"

    Each Q is embedded and stored. At query time, user question matches
    against generated questions (question↔question similarity).
```

### Why It Works
HyDE ([arXiv:2501.07391](https://arxiv.org/abs/2501.07391)) showed that matching questions to hypothetical answers improves retrieval. This strategy flips it: instead of generating hypothetical answers at query time, pre-generate hypothetical questions at index time. This is faster at query time (no LLM call needed).

### MongoDB Document Schema

```json
{
  "source": "SSA.pdf",
  "page": 12,
  "text": "Original passage text...",
  "hypothetical_questions": [
    "What is the income limit for SSI?",
    "How much can I earn and still get SSI?",
    "What is the maximum monthly income for SSI eligibility?"
  ],
  "embedding_q1": [0.012, ...],
  "embedding_q2": [0.034, ...],
  "embedding_q3": [0.056, ...],
  "embedding_passage": [0.078, ...]
}
```

### Implementation

```python
def generate_questions(text: str, n: int = 3) -> list[str]:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": (
                f"Generate exactly {n} diverse questions that this passage answers. "
                f"Return only the questions, one per line.\n\n{text}"
            ),
        }],
    )
    return [q.strip() for q in resp.choices[0].message.content.strip().split("\n") if q.strip()]


def index_with_questions(pdf_path: str):
    docs = load_pdf(pdf_path)
    chunks = splitter.split_documents(docs)

    for chunk in chunks:
        text = chunk.page_content
        questions = generate_questions(text, n=3)

        # Embed passage + each question
        all_texts = [text] + questions
        all_embeddings = embed_batch(all_texts)

        doc = {
            "source": pdf_path,
            "page": chunk.metadata.get("page_number"),
            "text": text,
            "hypothetical_questions": questions,
            "embedding_passage": all_embeddings[0],
        }
        # Store each question embedding separately
        for i, (q, emb) in enumerate(zip(questions, all_embeddings[1:])):
            doc[f"embedding_q{i}"] = emb

        collection.update_one(
            {"chunk_hash": hash_text(text)},
            {"$set": doc},
            upsert=True,
        )


def search_by_question(query: str, top_k: int = 5):
    q_emb = embed_batch([query])[0]

    # Search across question embeddings (not passage embeddings)
    # Create a vector index on each embedding_q* field
    results_q0 = vector_search(q_emb, "idx_q0", "embedding_q0", limit=10)
    results_q1 = vector_search(q_emb, "idx_q1", "embedding_q1", limit=10)
    results_q2 = vector_search(q_emb, "idx_q2", "embedding_q2", limit=10)
    results_passage = vector_search(q_emb, "idx_passage", "embedding_passage", limit=10)

    return rrf_merge([results_q0, results_q1, results_q2, results_passage], top_k=top_k)
```

---

## Strategy 10: Agentic Indexing with Metadata Enrichment

**Use an LLM to analyze each chunk during indexing** -- extracting structured metadata, classifying topics, identifying entities, and generating summaries. This enriched metadata powers filtered search and faceted retrieval at query time.

### When to Use
- Enterprise-grade systems with diverse query patterns
- Users filter by topic, date, entity, document type
- Need to support both search and analytics on the same data
- Documents have implicit structure not captured by text extraction

### How It Works

```
    Chunk ──▶ LLM Analysis ──▶ Extract:
                                  ├── topic: "retirement_benefits"
                                  ├── entities: ["Social Security", "Medicare"]
                                  ├── doc_type: "regulation"
                                  ├── key_figures: {"income_limit": "$943"}
                                  ├── summary: "Describes eligibility for..."
                                  └── keywords: ["SSI", "income", "eligibility"]

                            ──▶ Embed chunk + summary
                            ──▶ Store with rich metadata ──▶ MongoDB
```

### MongoDB Document Schema

```json
{
  "source": "SSA.pdf",
  "page": 12,
  "text": "Original passage...",
  "summary": "LLM-generated summary",
  "metadata": {
    "topic": "supplemental_security_income",
    "subtopic": "eligibility_requirements",
    "entities": ["SSI", "Social Security Administration"],
    "doc_type": "government_regulation",
    "key_figures": {"monthly_income_limit": "$943"},
    "keywords": ["SSI", "income", "eligibility", "benefits"],
    "audience": "applicants",
    "complexity": "medium"
  },
  "embedding": [0.012, ...],
  "embedding_summary": [0.034, ...],
  "chunk_hash": "a3f8b2c1",
  "indexed_at": "2026-04-18T10:30:00Z"
}
```

### MongoDB Index with Metadata Filters

```json
{
  "fields": [
    {"type": "vector", "path": "embedding", "numDimensions": 1536, "similarity": "cosine"},
    {"type": "filter", "path": "metadata.topic"},
    {"type": "filter", "path": "metadata.doc_type"},
    {"type": "filter", "path": "metadata.entities"},
    {"type": "filter", "path": "source"}
  ]
}
```

### Implementation

```python
import json
from openai import OpenAI

client = OpenAI()

EXTRACTION_PROMPT = """Analyze this text passage and extract structured metadata.
Return a JSON object with these fields:
- topic: primary topic (snake_case, e.g., "retirement_benefits")
- subtopic: specific subtopic
- entities: list of named entities (organizations, programs, acts)
- doc_type: one of [regulation, guidance, faq, overview, procedure]
- key_figures: dict of any specific numbers/limits/dates mentioned
- keywords: list of 5-10 search keywords
- audience: who this is written for
- summary: one-sentence summary

Text:
{text}

Return only valid JSON, no markdown."""


def enrich_chunk(text: str) -> dict:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": EXTRACTION_PROMPT.format(text=text)}],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def index_agentic(pdf_path: str):
    docs = load_pdf(pdf_path)
    chunks = splitter.split_documents(docs)

    for chunk in chunks:
        text = chunk.page_content

        # LLM-powered metadata extraction
        metadata = enrich_chunk(text)

        # Embed both original text and summary
        embs = embed_batch([text, metadata.get("summary", text)])

        collection.update_one(
            {"chunk_hash": hash_text(text)},
            {"$set": {
                "source": pdf_path,
                "page": chunk.metadata.get("page_number"),
                "text": text,
                "summary": metadata.get("summary", ""),
                "metadata": metadata,
                "embedding": embs[0],
                "embedding_summary": embs[1],
                "indexed_at": datetime.now(timezone.utc),
            }},
            upsert=True,
        )


def filtered_search(query: str, topic: str = None, doc_type: str = None, top_k: int = 5):
    """Vector search with metadata pre-filtering."""
    q_emb = embed_batch([query])[0]

    # Build filter
    filter_conditions = {}
    if topic:
        filter_conditions["metadata.topic"] = topic
    if doc_type:
        filter_conditions["metadata.doc_type"] = doc_type

    pipeline = [{
        "$vectorSearch": {
            "index": "vector_index",
            "path": "embedding",
            "queryVector": q_emb,
            "numCandidates": 150,
            "limit": top_k,
            **({"filter": filter_conditions} if filter_conditions else {}),
        }
    }]

    return list(collection.aggregate(pipeline))
```

---

## Strategy Comparison Matrix

| Strategy | Retrieval Quality | Index Cost | Query Speed | Storage | Best For |
|----------|:-:|:-:|:-:|:-:|---------|
| **1. Naive** | Low | Low | Fast | Small | Demos |
| **2. Structure-Aware** | Good | Low | Fast | Small | General production |
| **3. Hierarchical** | Very Good | Medium | Medium | 3x | Long docs, mixed queries |
| **4. Late Chunking** | Very Good | Medium | Fast | Small | Context-dependent text |
| **5. Hybrid Dense+Sparse** | Very Good | Low | Medium | Small | All production systems |
| **6. Parent-Child** | Very Good | Medium | Medium | 2x | Legal/regulatory docs |
| **7. Vision (ColPali)** | Good* | High (GPU) | Medium | Large | Scanned PDFs, forms |
| **8. Multi-Vector** | Excellent | High | Slow | 3-4x | Max quality, any budget |
| **9. Hypothetical Q** | Very Good | High | Fast | 2x | FAQ, compliance |
| **10. Agentic + Metadata** | Excellent | Very High | Fast (filtered) | 2x | Enterprise, faceted |

*\* Strategy 7 is page-level, not paragraph-level, so precision differs.*

---

## MongoDB Atlas Index Configurations

All strategies need at least one vector search index. Here are the configs for each:

### Strategies 1, 2, 4 (Single Vector)
```json
{
  "fields": [
    {"type": "vector", "path": "embedding", "numDimensions": 1536, "similarity": "cosine"},
    {"type": "filter", "path": "source"}
  ]
}
```

### Strategy 3 (One Index Per Collection)
Create `vector_index` on each of: `summaries`, `sections`, `paragraphs` -- all with the same schema as above.

### Strategy 5 (Vector + Full-Text)
Two indexes on the same collection:
- Vector index (as above)
- Full-text index on `text` field

### Strategy 6 (Parent-Child)
Vector index only on the `children` collection. No index needed on `parents`.

### Strategies 8, 9 (Multiple Vector Fields)
One vector index per embedding field:
```json
// Repeat for each embedding_* field, adjusting path and numDimensions
{"fields": [{"type": "vector", "path": "embedding_dense", "numDimensions": 1536, "similarity": "cosine"}]}
{"fields": [{"type": "vector", "path": "embedding_bge", "numDimensions": 1024, "similarity": "cosine"}]}
```

### Strategy 10 (Vector + Filters)
```json
{
  "fields": [
    {"type": "vector", "path": "embedding", "numDimensions": 1536, "similarity": "cosine"},
    {"type": "filter", "path": "metadata.topic"},
    {"type": "filter", "path": "metadata.doc_type"},
    {"type": "filter", "path": "metadata.entities"},
    {"type": "filter", "path": "source"}
  ]
}
```

---

## Choosing the Right Strategy for SSA.pdf

SSA.pdf is a government document -- text-heavy, structured with sections, containing specific figures and regulations. Here is the recommended combination:

### Recommended Stack

```
┌────────────────────────────────────────────────────────┐
│              RECOMMENDED FOR SSA.pdf                    │
├────────────────────────────────────────────────────────┤
│                                                        │
│  Parsing:     Strategy 2 (Unstructured.io, hi_res)     │
│  Chunking:    Strategy 2 (Recursive, 200 tokens, 20%)  │
│  Indexing:    Strategy 5 (Hybrid dense + full-text)     │
│  + Bonus:     Strategy 10 (Metadata enrichment)         │
│  Retrieval:   Hybrid search + RRF + cross-encoder       │
│                                                        │
│  Why this combination:                                  │
│  - Text-heavy = no need for Vision (Strategy 7)        │
│  - Has sections = structure-aware chunking helps        │
│  - Has specific terms (SSI, SSDI) = BM25 catches them  │
│  - Has regulations = metadata filtering is valuable    │
│                                                        │
│  If answers need more context:                          │
│    Add Strategy 6 (Parent-Child) on top                 │
│                                                        │
│  If users ask broad + specific questions:               │
│    Add Strategy 3 (Hierarchical) on top                 │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### Quick Start (Minimum Viable)

```bash
# Strategy 2 + 5 combined: structure-aware chunking + hybrid search
pip install pymongo langchain langchain-openai langchain-community \
  unstructured[pdf] python-dotenv tenacity

python index_pdf.py SSA.pdf   # Uses Strategy 2
python query.py               # Uses Strategy 5 hybrid search
```

### Full Power (All Enrichments)

```bash
# Strategy 2 + 5 + 10: structure-aware + hybrid + metadata enrichment
python index_pdf.py SSA.pdf           # Strategies 2 + 10
python query.py --hybrid --rerank     # Strategy 5 with reranking
```
