# Indexing PDFs into MongoDB Atlas Vector Database

This project demonstrates how to extract text from PDF documents (e.g., `SSA.pdf`), generate vector embeddings, and store them in MongoDB Atlas Vector Search for semantic querying.

---

## Prerequisites

1. **MongoDB Atlas account** with a cluster (M0 free tier works).
2. **Python 3.9+** installed.
3. **OpenAI API key** (for generating embeddings). You can also use a local model; see alternatives below.

---

## Step 1 -- Install Dependencies

```bash
pip install pymongo pypdf openai python-dotenv
```

| Package | Purpose |
|---------|---------|
| `pymongo` | MongoDB driver |
| `pypdf` | Extract text from PDF files |
| `openai` | Generate text embeddings via `text-embedding-3-small` |
| `python-dotenv` | Load secrets from `.env` |

---

## Step 2 -- Set Up Environment Variables

Create a `.env` file in the project root:

```env
MONGODB_URI=mongodb+srv://<username>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority
OPENAI_API_KEY=sk-...
```

> **Never commit `.env` to version control.** Add it to `.gitignore`.

---

## Step 3 -- Create the Vector Search Index in MongoDB Atlas

1. Go to your Atlas cluster -> **Atlas Search** -> **Create Search Index**.
2. Choose **JSON Editor** and select the target database/collection (e.g., `ssa_db.documents`).
3. Paste the following index definition (index name: `vector_index`):

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

4. Click **Create Search Index** and wait for it to become active.

---

## Step 4 -- Run the Indexing Script

```bash
python index_pdf.py
```

This script does the following:

1. Extracts text from `SSA.pdf` page by page.
2. Splits each page into chunks (~500 characters with overlap).
3. Generates an embedding for each chunk using OpenAI's `text-embedding-3-small`.
4. Inserts each chunk + embedding into MongoDB.

### `index_pdf.py`

```python
import os
from dotenv import load_dotenv
from pypdf import PdfReader
from openai import OpenAI
from pymongo import MongoClient

load_dotenv()

# --- Configuration ---
PDF_PATH = "SSA.pdf"
CHUNK_SIZE = 500        # characters per chunk
CHUNK_OVERLAP = 100     # overlap between consecutive chunks
DB_NAME = "ssa_db"
COLLECTION_NAME = "documents"
EMBEDDING_MODEL = "text-embedding-3-small"  # 1536 dimensions

# --- Clients ---
mongo = MongoClient(os.environ["MONGODB_URI"])
collection = mongo[DB_NAME][COLLECTION_NAME]
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def extract_text(pdf_path: str) -> list[dict]:
    """Return a list of {page, text} dicts."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text and text.strip():
            pages.append({"page": i, "text": text.strip()})
    return pages


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def get_embedding(text: str) -> list[float]:
    """Get embedding vector from OpenAI."""
    response = openai_client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL,
    )
    return response.data[0].embedding


def main():
    pages = extract_text(PDF_PATH)
    print(f"Extracted {len(pages)} pages from {PDF_PATH}")

    documents = []
    for page_info in pages:
        chunks = chunk_text(page_info["text"])
        for idx, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            documents.append({
                "source": PDF_PATH,
                "page": page_info["page"],
                "chunk_index": idx,
                "text": chunk,
                "embedding": embedding,
            })
            print(f"  Page {page_info['page']}, chunk {idx} embedded.")

    if documents:
        collection.insert_many(documents)
        print(f"\nInserted {len(documents)} chunks into "
              f"{DB_NAME}.{COLLECTION_NAME}")
    else:
        print("No text extracted from the PDF.")


if __name__ == "__main__":
    main()
```

---

## Step 5 -- Query the Vector Database

Use MongoDB's `$vectorSearch` aggregation stage to find semantically similar chunks:

### `query.py`

```python
import os
from dotenv import load_dotenv
from openai import OpenAI
from pymongo import MongoClient

load_dotenv()

mongo = MongoClient(os.environ["MONGODB_URI"])
collection = mongo["ssa_db"]["documents"]
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

QUERY = "What are the eligibility requirements for Social Security?"

# Generate query embedding
query_embedding = openai_client.embeddings.create(
    input=QUERY,
    model="text-embedding-3-small",
).data[0].embedding

# Vector search
results = collection.aggregate([
    {
        "$vectorSearch": {
            "index": "vector_index",
            "path": "embedding",
            "queryVector": query_embedding,
            "numCandidates": 100,
            "limit": 5,
        }
    },
    {
        "$project": {
            "text": 1,
            "page": 1,
            "score": {"$meta": "vectorSearchScore"},
            "_id": 0,
        }
    },
])

for doc in results:
    print(f"[Page {doc['page']}] (score: {doc['score']:.4f})")
    print(f"  {doc['text'][:200]}...\n")
```

```bash
python query.py
```

---

## Alternatives

### Use a free/local embedding model instead of OpenAI

Replace OpenAI embeddings with `sentence-transformers` (runs locally, no API key needed):

```bash
pip install sentence-transformers
```

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")  # 384 dimensions

def get_embedding(text: str) -> list[float]:
    return model.encode(text).tolist()
```

> If you switch models, update `numDimensions` in the Atlas vector index to match (e.g., `384` for MiniLM).

### Index all PDFs in the directory

Change `main()` to loop over all PDF files:

```python
import glob

for pdf_path in glob.glob("*.pdf"):
    pages = extract_text(pdf_path)
    # ... rest of the processing with source=pdf_path
```

---

## Project Structure

```
vecto-databases/
  SSA.pdf                          # Source document
  Excess_Unstated_Income.pdf       # Additional PDFs
  Retirement_benefits.pdf
  SSI.pdf
  SSI for Groups and Organizations.pdf
  index_pdf.py                     # Indexing script
  query.py                         # Query script
  .env                             # Secrets (do not commit)
  README.md                        # This file
```
