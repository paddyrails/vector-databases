# Research-Backed PDF Indexing into MongoDB Atlas Vector Search

Findings from 40 arxiv papers covering chunking, embeddings, vector indexing, RAG, evaluation, and document processing -- applied to our PDF indexing pipeline.

---

## Table of Contents

1. [Chunking Strategies](#1-chunking-strategies)
2. [Embedding Models and Retrieval Techniques](#2-embedding-models-and-retrieval-techniques)
3. [Vector Indexing Algorithms](#3-vector-indexing-algorithms)
4. [RAG Best Practices](#4-rag-best-practices)
5. [Evaluation Methods](#5-evaluation-methods)
6. [PDF/Document Processing](#6-pdfdocument-processing)
7. [Recommended Architecture](#7-recommended-architecture)
8. [Full Paper Reference Table](#8-full-paper-reference-table)

---

## 1. Chunking Strategies

### The Problem with Naive Chunking

Fixed-character splitting (e.g., split every 500 chars) breaks sentences mid-word, ignores document structure, and destroys semantic coherence. Research shows this directly hurts retrieval quality.

### What the Research Says

#### Structure-Aware Chunking Wins

**"Beyond Chunk-Then-Embed"** ([arXiv:2602.16974](https://arxiv.org/abs/2602.16974)) evaluated 36 segmentation methods across six domains with five embedding models. **Paragraph Group Chunking** achieved the highest overall accuracy (nDCG@5 ~0.459). The key insight: chunking that respects document structure (headers, paragraphs, sections) achieves the highest retrieval scores with **half the chunks** compared to structure-unaware methods, reducing both indexing cost and query latency ([arXiv:2603.06976](https://arxiv.org/abs/2603.06976)).

#### Optimal Chunk Size: 100-256 Tokens

**"Rethinking Chunk Size for Long-Document Retrieval"** ([arXiv:2505.21700](https://arxiv.org/abs/2505.21700)) tested chunk sizes from 64-512 tokens with overlap ratios 0%-80%. **Recursive token-based chunking at ~100 tokens with no overlap** (R100-0) consistently outperformed other approaches.

#### Late Chunking: Embed First, Chunk Second

**"Late Chunking"** ([arXiv:2409.04701](https://arxiv.org/abs/2409.04701)) from Jina AI flips the standard approach: instead of chunk-then-embed, you embed the full document through the transformer, *then* split into chunks before mean pooling. This preserves cross-chunk contextual information without any additional training. Superior results across multiple benchmarks.

#### Domain-Specific Chunking Matters

**"Breaking It Down"** ([arXiv:2512.00367](https://arxiv.org/abs/2512.00367)) introduced Projected Similarity Chunking (PSC), achieving a **24x improvement in MRR** on PubMedQA over fixed-length chunking. For cross-document scenarios, topic-aligned chunking ([arXiv:2601.05265](https://arxiv.org/abs/2601.05265)) reached context precision of 0.89, outperforming semantic chunking by 19% and fixed-size by 45%.

### Actionable Takeaway for Our Pipeline

```
Use RecursiveCharacterTextSplitter at 100-256 tokens with 20% overlap.
Respect paragraph/section boundaries.
Consider Late Chunking if using Jina embedding models.
Test on your specific document domain before committing to a strategy.
```

---

## 2. Embedding Models and Retrieval Techniques

### The Landscape Has Shifted

The MTEB benchmark review ([arXiv:2406.01607](https://arxiv.org/abs/2406.01607)) shows that **LLM-based embedding models** (BGE, E5-Mistral, GTE-Qwen2) now substantially outperform older BERT/T5-based encoders. The gap is significant.

### Top Models (Ranked)

| Model | Dimensions | Type | Best For |
|-------|-----------|------|----------|
| **BGE-M3** (BAAI) | 1024 | Dense + Sparse in single pass | Hybrid search (ideal for MongoDB) |
| **text-embedding-3-large** (OpenAI) | 3072 (flexible) | Dense | High accuracy, easy API |
| **E5-Mistral-7B** (Microsoft) | 4096 | Dense | High accuracy, self-hosted |
| **GTE-Qwen2** (Alibaba) | 768-1536 | Dense | Open-source, multilingual |
| **NV-Retriever-v1** (NVIDIA) | 4096 | Dense | SOTA on BEIR benchmark |
| **Cohere embed-v3** | 1024 | Dense | Production-ready API |

**NV-Retriever** ([arXiv:2407.15831](https://arxiv.org/abs/2407.15831)) achieved #1 on MTEB BEIR with NDCG@10 of 60.9 through superior hard-negative mining during training.

### Hybrid Search is Non-Negotiable

Multiple papers confirm that combining dense (vector) + sparse (BM25) retrieval outperforms either alone:

- **"Blended RAG"** ([arXiv:2404.07220](https://arxiv.org/abs/2404.07220)): Linear combination of cosine similarity + BM25 with tunable weights consistently wins.
- **"DAT"** ([arXiv:2503.23013](https://arxiv.org/abs/2503.23013)): Dynamically balances dense vs BM25 per query using LLM-based scoring.
- **BGE-M3** ([arXiv:2410.20381](https://arxiv.org/abs/2410.20381)): Produces both dense and sparse representations in one forward pass -- perfect for MongoDB Atlas (vector search + Atlas Search).

### Late Interaction Models (ColBERT)

**ColBERT** ([arXiv:2004.12832](https://arxiv.org/abs/2004.12832)) and **Jina-ColBERT-v2** ([arXiv:2408.16672](https://arxiv.org/abs/2408.16672)) use token-level multi-vector representations with late interaction for fine-grained similarity. These are excellent as **rerankers** after initial retrieval.

### Actionable Takeaway for Our Pipeline

```
Embedding:  BGE-M3 for hybrid search, or OpenAI text-embedding-3-large for simplicity.
Retrieval:  Always combine vector search + Atlas full-text search.
Reranking:  Add a cross-encoder or ColBERT reranker (top 20 → top 5).
Dimensions: 768-1536 is the sweet spot; use Matryoshka embeddings for flexibility.
```

---

## 3. Vector Indexing Algorithms

### How MongoDB Atlas Works Under the Hood

MongoDB Atlas Vector Search uses **HNSW (Hierarchical Navigable Small World)** internally. The Faiss library paper ([arXiv:2401.08281](https://arxiv.org/abs/2401.08281)) from Meta/FAIR provides the definitive reference for understanding the trade-off space:

| Algorithm | How It Works | Trade-off |
|-----------|-------------|-----------|
| **HNSW** | Multi-layer graph with navigable small-world properties | Best recall-speed for in-memory. O(log N) search. |
| **IVF** | Partitions vectors into Voronoi cells via k-means | Good for large-scale; search only nprobe nearest cells |
| **PQ** (Product Quantization) | Splits vectors into sub-vectors, quantizes each | 10-100x compression, moderate accuracy loss |
| **IVF-PQ** | IVF + PQ combined | Billion-scale search with compression |

### Filtered ANN Search is Critical

**"Filtered ANN Search"** ([arXiv:2509.07789](https://arxiv.org/abs/2509.07789)) shows that combining metadata filters with vector similarity is essential for production systems. MongoDB supports this via `$vectorSearch` with filter predicates. **Pre-filtering** works best for high-selectivity filters.

### Tuning MongoDB Atlas Vector Search

The key parameter is `numCandidates` (analogous to HNSW's `ef_search`):

```
numCandidates = 10x to 20x your desired limit (k)

Example: If you want top 5 results, set numCandidates to 50-100.
Higher = better recall, slower latency.
```

MongoDB Atlas also supports **scalar and binary quantization** for reducing storage on large collections.

---

## 4. RAG Best Practices

### Evolution: Naive → Advanced → Modular → Agentic

The comprehensive RAG survey ([arXiv:2312.10997](https://arxiv.org/abs/2312.10997)) documents four generations:

```
Naive RAG:    Retrieve → Generate (basic, often hallucinated)
Advanced RAG: Query expansion + reranking + context filtering
Modular RAG:  Pluggable components (retriever, reranker, generator)
Agentic RAG:  LLM decides when/how much to retrieve dynamically
```

### Six Research-Backed Practices

1. **Query Expansion** ([arXiv:2501.07391](https://arxiv.org/abs/2501.07391))
   - **HyDE**: Generate a hypothetical answer, embed it, use that embedding to search. Captures intent better than raw query.
   - **Multi-Query**: Rephrase the query into 3-5 variants, retrieve for each, merge results.

2. **Hybrid Retrieval** ([arXiv:2404.07220](https://arxiv.org/abs/2404.07220))
   - Combine vector search + BM25 full-text search.
   - Use Reciprocal Rank Fusion (RRF) or learned weighting to merge results.

3. **Reranking** ([arXiv:2506.00054](https://arxiv.org/abs/2506.00054))
   - Initial retrieval returns 20 candidates.
   - Cross-encoder reranker scores each (query, passage) pair.
   - Return top 5 with highest reranker scores.

4. **Context Filtering** ([arXiv:2506.00054](https://arxiv.org/abs/2506.00054))
   - Filter irrelevant spans within retrieved chunks before feeding to the LLM.
   - Reduces noise, prevents hallucination.

5. **Corrective RAG** ([arXiv:2604.01733](https://arxiv.org/abs/2604.01733))
   - Validate retrieved context before generation.
   - If retrieved context is irrelevant, trigger web search or alternative retrieval.

6. **Iterative/Agentic Retrieval** ([arXiv:2501.09136](https://arxiv.org/abs/2501.09136))
   - For complex queries, let the LLM request additional context in multiple rounds.
   - SAM-RAG dynamically filters documents and verifies evidence.

### Actionable Takeaway for Our Pipeline

```
Pre-retrieval:   HyDE or multi-query expansion
Retrieval:       Hybrid search (vector + BM25), 20 candidates
Post-retrieval:  Cross-encoder rerank → top 5, filter irrelevant spans
Generation:      Structured prompt with citations, source attribution
Chunk overlap:   20-25% to prevent boundary information loss
```

---

## 5. Evaluation Methods

### Classical IR Metrics Are Not Enough

**"Redefining Retrieval Evaluation"** ([arXiv:2510.21440](https://arxiv.org/abs/2510.21440)) shows that classical metrics (nDCG, MAP, MRR) fail to adequately predict downstream RAG performance. You need both retrieval metrics *and* end-to-end generation metrics.

### Retrieval Metrics

| Metric | What It Measures |
|--------|-----------------|
| **Recall@k** | Fraction of relevant documents in top-k |
| **MRR** | Average reciprocal rank of first correct result |
| **nDCG@10** | Relevance-weighted ranking quality |
| **MAP** | Average precision across all queries |

### End-to-End RAG Metrics (RAGAS Framework)

The RAG evaluation survey ([arXiv:2504.14891](https://arxiv.org/abs/2504.14891)) recommends the **RAGAS** framework:

| Metric | What It Measures |
|--------|-----------------|
| **Faithfulness** | Is the answer supported by retrieved context? |
| **Answer Relevancy** | Does the answer address the question? |
| **Context Precision** | Are the retrieved chunks actually relevant? |
| **Context Recall** | Did retrieval capture all needed information? |

### Evaluation Frameworks

- **RAGAS** ([ragas.io](https://docs.ragas.io)): Automated evaluation, most widely adopted.
- **ARES** ([arXiv:2405.07437](https://arxiv.org/abs/2405.07437)): Automated RAG Evaluation System.
- **TruLens**: Context relevance, groundedness, answer relevance.
- **Rarity-Aware Metrics** ([arXiv:2511.09545](https://arxiv.org/abs/2511.09545)): Weight rare/specific information higher -- important for domain documents like SSA.pdf.

### Actionable Takeaway

```
1. Build a ground-truth Q&A dataset from your PDFs (20-50 pairs).
2. Measure retrieval: Recall@5, MRR, nDCG@10.
3. Measure end-to-end: RAGAS (faithfulness + answer relevancy + context precision).
4. A/B test chunking strategies, embedding models, and retrieval params.
```

---

## 6. PDF/Document Processing

### The OCR Problem

Traditional pipelines (pypdf → text → chunks) fail on scanned PDFs, complex layouts, tables, and figures. Research offers two paradigms:

### Paradigm A: Advanced OCR + Layout Detection

**"Document Parsing Unveiled"** ([arXiv:2410.21169](https://arxiv.org/abs/2410.21169)) covers end-to-end pipelines combining OCR, table recognition, and formula recognition. Key tools:

| Tool | Strengths |
|------|-----------|
| **Docling** (IBM) | Native MongoDB Atlas integration, tables, multi-column layouts |
| **Unstructured.io** | General-purpose, layout-aware, hi-res mode with OCR |
| **Qianfan-OCR** ([arXiv:2603.13398](https://arxiv.org/abs/2603.13398)) | Unified end-to-end model for layout + content + relations |
| **dots.ocr** ([arXiv:2512.02498](https://arxiv.org/abs/2512.02498)) | Single VLM for multilingual layout parsing |

### Paradigm B: Vision-Based (Skip OCR Entirely)

**ColPali** ([arXiv:2407.01449](https://arxiv.org/abs/2407.01449), ICLR 2025) is a breakthrough. It:

- Uses a Vision Language Model (PaliGemma) to directly embed document page images
- Produces ColBERT-style multi-vector embeddings per page
- Captures both textual *and* visual information (tables, figures, layout)
- Eliminates OCR errors entirely

**"Lost in OCR Translation"** ([arXiv:2505.05666](https://arxiv.org/abs/2505.05666)) confirms vision-based approaches are more robust for complex layouts, poor scans, and non-standard fonts.

### Handling Tables

Tables are a known failure point for text-based extraction. Research recommends:

1. Detect tables separately during layout analysis
2. Extract as structured data (Markdown or JSON)
3. Embed as separate chunks with `type: "table"` metadata
4. Use table-specific prompts during RAG generation

### Actionable Takeaway

```
For text-heavy PDFs (like SSA.pdf):
  → Docling or Unstructured.io with hi_res strategy

For image-heavy / scanned PDFs:
  → ColPali (vision-based, no OCR needed)

Always:
  → Extract tables separately as structured data
  → Preserve section headers as metadata
  → Store page numbers for citation
```

---

## 7. Recommended Architecture

Based on the research, here is the optimal end-to-end architecture for indexing SSA.pdf (and similar government documents) into MongoDB Atlas:

### Ingestion Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE                       │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1. DOCUMENT PARSING                                         │
│     Tool: Docling (IBM) or Unstructured.io                   │
│     Why:  Layout-aware, handles tables, native MongoDB       │
│           integration                                        │
│     Paper: arXiv:2410.21169                                  │
│                                                              │
│  2. CHUNKING                                                 │
│     Method: Recursive structure-aware splitting              │
│     Size:   100-256 tokens, 20% overlap                      │
│     Why:    Paragraph Group Chunking had highest nDCG@5      │
│     Papers: arXiv:2602.16974, arXiv:2505.21700               │
│                                                              │
│  3. METADATA ENRICHMENT                                      │
│     Fields: source, page, section_header, chunk_hash,        │
│             char_count, indexed_at                            │
│     Why:    Enables filtered ANN search                      │
│     Paper:  arXiv:2509.07789                                 │
│                                                              │
│  4. EMBEDDING                                                │
│     Model:  BGE-M3 (dense+sparse) or OpenAI                  │
│             text-embedding-3-large                            │
│     Batch:  100 chunks/request, exponential backoff retry     │
│     Why:    BGE-M3 enables hybrid search in single pass       │
│     Papers: arXiv:2406.01607, arXiv:2410.20381               │
│                                                              │
│  5. STORAGE                                                  │
│     Target: MongoDB Atlas                                    │
│     Method: Idempotent upsert via chunk_hash                 │
│     Index:  HNSW vector index + Atlas full-text index         │
│     Config: numCandidates = 10-20x desired k                  │
│     Paper:  arXiv:2401.08281                                 │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Retrieval + RAG Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│                    RETRIEVAL + RAG PIPELINE                   │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1. QUERY EXPANSION                                          │
│     Method: HyDE (generate hypothetical answer, embed it)    │
│     Alt:    Multi-query (3-5 rephrasings)                    │
│     Paper:  arXiv:2501.07391                                 │
│                                                              │
│  2. HYBRID RETRIEVAL                                         │
│     Vector: $vectorSearch (numCandidates=100, limit=20)      │
│     Text:   $search (Atlas full-text, limit=20)              │
│     Merge:  Reciprocal Rank Fusion (RRF, k=60)               │
│     Paper:  arXiv:2404.07220                                 │
│                                                              │
│  3. RERANKING                                                │
│     Model:  Cross-encoder (ms-marco-MiniLM-L-12-v2)         │
│             or Cohere Rerank API                              │
│     Input:  Top 20 from hybrid search                        │
│     Output: Top 5 reranked results                           │
│     Paper:  arXiv:2506.00054                                 │
│                                                              │
│  4. CONTEXT ASSEMBLY                                         │
│     - Merge overlapping chunks                               │
│     - Add source/page metadata for citations                 │
│     - Filter irrelevant spans (FILCO approach)               │
│                                                              │
│  5. GENERATION                                               │
│     Model:  GPT-4o / Claude                                  │
│     Prompt: System prompt enforcing source-grounded answers  │
│     Output: Answer with [Source, Page] citations              │
│     Paper:  arXiv:2312.10997                                 │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Evaluation Framework

```
┌──────────────────────────────────────────────────────────────┐
│                    EVALUATION FRAMEWORK                      │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Ground Truth:                                               │
│    - 20-50 Q&A pairs from SSA.pdf with source page refs      │
│                                                              │
│  Retrieval Metrics:                                          │
│    - Recall@5, Recall@10, MRR, nDCG@10                       │
│    Paper: arXiv:2510.21440                                   │
│                                                              │
│  End-to-End Metrics (RAGAS):                                 │
│    - Faithfulness:      Is the answer grounded in context?   │
│    - Answer Relevancy:  Does it address the question?        │
│    - Context Precision: Are retrieved chunks relevant?       │
│    - Context Recall:    Is all needed info retrieved?         │
│    Paper: arXiv:2504.14891                                   │
│                                                              │
│  A/B Testing:                                                │
│    - Compare chunking strategies on your eval set            │
│    - Compare embedding models                                │
│    - Tune numCandidates and hybrid search weights            │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 8. Full Paper Reference Table

40 papers organized by topic:

### Chunking (7 papers)

| Paper | ArXiv ID | Key Contribution |
|-------|----------|-----------------|
| Beyond Chunk-Then-Embed | [2602.16974](https://arxiv.org/abs/2602.16974) | Taxonomy of 36 chunking methods; Paragraph Group Chunking wins |
| Systematic Investigation of Chunking | [2603.06976](https://arxiv.org/abs/2603.06976) | Structure-aware = half the chunks, better retrieval |
| Late Chunking | [2409.04701](https://arxiv.org/abs/2409.04701) | Embed first, chunk second; preserves cross-chunk context |
| Domain-Aware Semantic Segmentation | [2512.00367](https://arxiv.org/abs/2512.00367) | 24x MRR improvement with PSC on PubMedQA |
| Rethinking Chunk Size | [2505.21700](https://arxiv.org/abs/2505.21700) | R100-0 (recursive, 100 tokens, no overlap) wins |
| Cross-Document Topic-Aligned Chunking | [2601.05265](https://arxiv.org/abs/2601.05265) | Context precision 0.89, +45% over fixed-size |
| MultiDocFusion | [2604.12352](https://arxiv.org/abs/2604.12352) | Hierarchical chunking for complex industrial docs |

### Embeddings & Retrieval (10 papers)

| Paper | ArXiv ID | Key Contribution |
|-------|----------|-----------------|
| Universal Text Embeddings (MTEB Review) | [2406.01607](https://arxiv.org/abs/2406.01607) | LLM-based embeddings dominate BERT-based |
| NV-Retriever | [2407.15831](https://arxiv.org/abs/2407.15831) | #1 MTEB BEIR via hard-negative mining |
| Large Reasoning Embedding Models | [2510.14321](https://arxiv.org/abs/2510.14321) | LLMs adapted as embedding models |
| Model Architectures in IR Survey | [2502.14822](https://arxiv.org/abs/2502.14822) | Evolution from BERT to LLM embeddings |
| ColBERT | [2004.12832](https://arxiv.org/abs/2004.12832) | Token-level late interaction retrieval |
| ColBERTv2 | [2112.01488](https://arxiv.org/abs/2112.01488) | Efficient ColBERT with residual compression |
| Jina-ColBERT-v2 | [2408.16672](https://arxiv.org/abs/2408.16672) | Multilingual late interaction |
| DAT: Dynamic Alpha Tuning | [2503.23013](https://arxiv.org/abs/2503.23013) | Dynamic dense/BM25 weighting per query |
| Blended RAG | [2404.07220](https://arxiv.org/abs/2404.07220) | Hybrid retrieval consistently outperforms single-method |
| Dense-Sparse Hybrid Vectors | [2410.20381](https://arxiv.org/abs/2410.20381) | BGE-M3 for single-pass dense+sparse |

### Vector Indexing (5 papers)

| Paper | ArXiv ID | Key Contribution |
|-------|----------|-----------------|
| The Faiss Library | [2401.08281](https://arxiv.org/abs/2401.08281) | Definitive reference for HNSW, IVF, PQ |
| Comprehensive Vector DB Survey | [2310.11703](https://arxiv.org/abs/2310.11703) | Algorithm comparison across methods |
| Vector DBMS Survey | [2310.14021](https://arxiv.org/abs/2310.14021) | Milvus, Pinecone, Weaviate, Qdrant comparison |
| Filtered ANN Benchmark | [2509.07789](https://arxiv.org/abs/2509.07789) | Pre-filtering vs post-filtering for metadata |
| Attribute Filtering in ANN | [2508.16263](https://arxiv.org/abs/2508.16263) | In-depth filtered search study |

### RAG (6 papers)

| Paper | ArXiv ID | Key Contribution |
|-------|----------|-----------------|
| RAG for LLMs Survey | [2312.10997](https://arxiv.org/abs/2312.10997) | Naive → Advanced → Modular RAG taxonomy |
| RAG Architectures Survey | [2506.00054](https://arxiv.org/abs/2506.00054) | Retriever-centric vs generator-centric |
| RAG Best Practices Study | [2501.07391](https://arxiv.org/abs/2501.07391) | Chunk size, query expansion, prompt design |
| Agentic RAG | [2501.09136](https://arxiv.org/abs/2501.09136) | LLM decides when/how much to retrieve |
| Systematic RAG Review | [2507.18910](https://arxiv.org/abs/2507.18910) | Granularity-aware retrieval |
| BM25 to Corrective RAG | [2604.01733](https://arxiv.org/abs/2604.01733) | Validate context before generation |

### Evaluation (5 papers)

| Paper | ArXiv ID | Key Contribution |
|-------|----------|-----------------|
| Redefining Retrieval Evaluation | [2510.21440](https://arxiv.org/abs/2510.21440) | Classical metrics fail for RAG |
| RAG Evaluation Survey | [2504.14891](https://arxiv.org/abs/2504.14891) | RAGAS framework metrics |
| Evaluation of RAG Survey | [2405.07437](https://arxiv.org/abs/2405.07437) | RAGAS, ARES, TruLens comparison |
| Rarity-Aware RAG Metric | [2511.09545](https://arxiv.org/abs/2511.09545) | Weight rare information higher |
| How Important is Recall | [2512.20854](https://arxiv.org/abs/2512.20854) | Recall vs precision trade-offs |

### Document Processing (6 papers)

| Paper | ArXiv ID | Key Contribution |
|-------|----------|-----------------|
| Document Parsing Unveiled | [2410.21169](https://arxiv.org/abs/2410.21169) | End-to-end OCR + layout + tables |
| ColPali (ICLR 2025) | [2407.01449](https://arxiv.org/abs/2407.01449) | Vision-based retrieval, skip OCR entirely |
| Lost in OCR Translation | [2505.05666](https://arxiv.org/abs/2505.05666) | Vision > OCR for complex layouts |
| Qianfan-OCR | [2603.13398](https://arxiv.org/abs/2603.13398) | Unified document intelligence model |
| dots.ocr | [2512.02498](https://arxiv.org/abs/2512.02498) | Single VLM for multilingual layout |
| Logics-Parsing | [2509.19760](https://arxiv.org/abs/2509.19760) | RL-based layout-centric parsing |

### Industry Tools (not arxiv)

| Tool | Reference | Key Feature |
|------|-----------|-------------|
| Docling (IBM) | [docs](https://docling-project.github.io/docling/examples/rag_mongodb/) | Native MongoDB Atlas integration |

---

## TL;DR -- What to Change in Our Pipeline

Compared to the basic approach in `README.md` and the professional approach in `README2.md`, research suggests these high-impact changes:

| Change | Impact | Effort |
|--------|--------|--------|
| Switch to recursive structure-aware chunking at 100-256 tokens | High | Low |
| Add hybrid search (vector + Atlas full-text) | High | Medium |
| Use BGE-M3 for single-pass dense+sparse embeddings | High | Medium |
| Add cross-encoder reranking after retrieval | High | Medium |
| Use Docling instead of pypdf/Unstructured for PDF parsing | Medium | Low |
| Add HyDE query expansion | Medium | Medium |
| Build ground-truth Q&A pairs and evaluate with RAGAS | High | High |
| Consider ColPali for scanned/image-heavy documents | Medium | High |
