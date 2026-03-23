# 🧠 Multi-Model RAG System

> Fully **offline** Retrieval-Augmented Generation pipeline using FAISS HNSW, bge-m3 embeddings, and your local GLM-OCR model.

---

## What This System Does

This notebook builds a complete RAG (Retrieval-Augmented Generation) pipeline that:

1. **Ingests** documents from multiple formats (PDF, images, CSV, code, text)
2. **Chunks** them into searchable pieces
3. **Embeds** each chunk into a 1024-dimensional vector using `bge-m3` (runs fully locally)
4. **Indexes** all vectors in a FAISS HNSW database on your disk
5. **Retrieves** the most relevant chunks via hybrid search (dense + BM25 + re-ranking)
6. **Generates** answers using your local GLM-OCR model — no internet needed

---

## Project Structure

```
multimodel_rag/
├── multimodel_rag.ipynb      ← Main notebook (run this)
├── requirements.txt          ← All dependencies
├── .env                      ← Your config (create from .env.example)
├── README.md                 ← This file
│
├── data/                     ← ✏️  PUT YOUR FILES HERE
│   ├── document.pdf
│   ├── image.png
│   ├── table.csv
│   └── script.py
│
├── faiss_index/              ← Auto-created: saved HNSW index
│   ├── hnsw.index
│   └── chunks.pkl
│
├── metadata/                 ← Auto-created: TinyDB metadata
│   └── metadata.json
│
├── models_cache/             ← Auto-created: downloaded HuggingFace models
│   ├── BAAI/bge-m3/          ← ~570MB, downloaded once
│   └── cross-encoder/...     ← ~85MB, downloaded once
│
└── logs/                     ← Auto-created: log files
```

---

## Notebook Sections Explained

### Section 0 — Setup & Configuration
Checks all packages are installed and loads configuration from `.env` or defaults.
Defines all paths, model names, FAISS parameters, and GLM-OCR connection settings.

**Key configs you can change in `.env`:**
| Variable | Default | Description |
|---|---|---|
| `EMBED_MODEL` | `BAAI/bge-m3` | Local embedding model |
| `EMBED_DEVICE` | `cpu` | `cpu` or `cuda` |
| `CHUNK_SIZE` | `512` | Tokens per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `TOP_K_FINAL` | `5` | Chunks passed to LLM |
| `GLM_HOST` | `http://localhost` | Your GLM-OCR server host |
| `GLM_PORT` | `8000` | Your GLM-OCR server port |
| `GLM_MODEL` | `glm-ocr` | Model name your server expects |

---

### Section 1 — Data Ingestion
Scans the `data/` folder and loads every supported file.

| File type | How it's loaded |
|---|---|
| `.pdf` | `pdfplumber` extracts text + tables; flags scanned PDFs for OCR |
| `.png`, `.jpg`, `.jpeg` | PIL loads image; GLM-OCR extracts text (Section 6) |
| `.csv`, `.xlsx` | Pandas loads; schema + sample + full data serialized as text |
| `.py`, `.js`, `.java` etc | Read as code blocks with language tag |
| `.txt`, `.md`, `.docx`, `.html` | Read as plain text |

Each loaded file becomes a `RawDocument` object with:
- `doc_id` — MD5 hash of file (for deduplication)
- `content` — extracted text
- `metadata` — filename, page count, type, timestamp
- `needs_ocr` — flag for scanned/image files

---

### Section 2 — Chunking
Splits each document into overlapping chunks using `RecursiveCharacterTextSplitter`.

- **Chunk size**: 512 tokens (configurable)
- **Overlap**: 64 tokens (preserves context across boundaries)
- **Separators**: tries `\n\n` → `\n` → `. ` → ` ` in order
- **Images**: kept as single chunks; text filled by OCR

Each chunk stores its `chunk_id`, parent `doc_id`, position (`chunk_index / total_chunks`), and full metadata.

---

### Section 3 — Embedding
Converts each chunk's text into a 1024-dimensional float32 vector using `BAAI/bge-m3`.

**Why bge-m3?**
- Best-in-class multilingual model (supports 100+ languages)
- Runs entirely on CPU — no GPU required
- Single model handles text from any domain
- Downloads once (~570MB) to `models_cache/`, then locked offline

Produces an `(N, 1024)` numpy array, one row per chunk.

---

### Section 4 — Vector Store (FAISS HNSW)
Builds and saves the FAISS HNSW index to disk.

**Why HNSW?**
- Hierarchical Navigable Small World graph
- Sub-linear search time — fast even with millions of vectors
- No index training needed (unlike IVF)
- Loads back from disk in milliseconds

**Files saved:**
- `faiss_index/hnsw.index` — the FAISS index
- `faiss_index/chunks.pkl` — list of all Chunk objects (for lookup by FAISS integer ID)
- `metadata/metadata.json` — TinyDB store for provenance tracking

To reload without rebuilding: call `load_vector_store()` in Section 4.

---

### Section 5 — Retrieval
Full hybrid retrieval pipeline:

```
Query
  ↓
Dense search (FAISS)      Sparse search (BM25)
  ↓                           ↓
     Reciprocal Rank Fusion (RRF)
                ↓
        Cross-encoder re-ranking
                ↓
          Top-5 chunks
```

1. **Dense search** — FAISS HNSW finds semantically similar chunks via cosine similarity
2. **Sparse search** — BM25 finds keyword-matching chunks
3. **RRF fusion** — Combines both ranked lists without needing score normalization
4. **Re-ranking** — `cross-encoder/ms-marco-MiniLM-L-6-v2` scores every (query, chunk) pair directly for precise final ordering

---

### Section 6 — GLM-OCR Connector
HTTP client for your locally running GLM-OCR model.

**Supports:**
- Text-only completions
- Vision/OCR requests (image bytes → base64 → API)
- Streaming token-by-token output
- Auto-retry (3 attempts) on transient failures

**OCR workflow:**
- Images and scanned PDFs flagged with `needs_ocr=True` during ingestion
- `run_ocr_pass()` sends each to GLM-OCR and updates `doc.content`
- Re-run Sections 2–4 after OCR to re-chunk and re-index with the extracted text

---

### Section 7 — Full RAG Pipeline
Puts it all together:

```python
result = rag_query("What does the contract say about payment terms?")
```

Flow:
1. Retrieve top-5 chunks for the query
2. Print source filenames + preview (optional)
3. Build prompt: `CONTEXT: [chunks] + QUESTION: [query]`
4. If any retrieved chunk is an image → include image bytes in the API call
5. Stream the answer from GLM-OCR token by token

**Customize the system prompt** in the `RAG_SYSTEM_PROMPT` variable at the top of Section 7.

the output by rank wise

![Alt text](/home/shamanth/Pictures/Screenshots/Screenshot from 2026-03-23 20-37-12.png)

---

### Section 8 — Evaluation
Two offline evaluation functions:

- `eval_retrieval(query, expected_file)` — checks if the right document appears in top-k results
- `eval_answer(generated, reference)` — computes ROUGE-1/2/L + semantic cosine similarity

---

## How to Add Your Files

```
Just copy files into the  data/  folder and re-run Section 1.
```

**Step by step:**
1. Copy your PDF / image / CSV / code file into the `data/` folder
2. Open the notebook
3. Re-run **Section 1** (loads new files)
4. Re-run **Section 2** (re-chunks)
5. Re-run **Section 3** (re-embeds)
6. Re-run **Section 4** (rebuilds index)
7. Now query in **Section 7**

**Supported formats:**
```
PDFs          →  .pdf
Images        →  .png  .jpg  .jpeg  .webp  .bmp  .tiff
Spreadsheets  →  .csv  .xlsx  .xls
Documents     →  .txt  .md  .docx  .html  .rst
Code          →  .py  .js  .ts  .java  .cpp  .c  .go  .rs  .rb  .sh
```

---

## First-Time Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Create your `.env` file
```bash
cp .env.example .env
# Edit .env to match your GLM-OCR port and any custom settings
```

### 3. Start your GLM-OCR model
Make sure GLM-OCR is running locally before executing Section 6+.  
Default expected endpoint: `http://localhost:8000/v1/chat/completions`

### 4. Run the notebook
```bash
jupyter notebook multimodel_rag.ipynb
```

Run sections **0 → 1 → 2 → 3 → 4 → 5 → 6 → 7** in order on first setup.  
After that, models and index are cached — skip straight to Section 7 for queries.

---

## Offline Operation

After the first run (which downloads models):

| Component | Where it lives | Internet needed? |
|---|---|---|
| bge-m3 embedding model | `models_cache/BAAI/bge-m3/` | ❌ No |
| Cross-encoder re-ranker | `models_cache/cross-encoder/...` | ❌ No |
| FAISS index | `faiss_index/hnsw.index` | ❌ No |
| Chunk store | `faiss_index/chunks.pkl` | ❌ No |
| GLM-OCR model | Your local server | ❌ No |

Set `TRANSFORMERS_OFFLINE=1` in `.env` to enforce offline mode and prevent any accidental download attempts.

---

## Environment Variables (`.env`)

```env
# Embedding
EMBED_MODEL=BAAI/bge-m3
EMBED_DEVICE=cpu
EMBED_BATCH=32

# Chunking
CHUNK_SIZE=512
CHUNK_OVERLAP=64

# Retrieval
TOP_K_DENSE=10
TOP_K_SPARSE=10
TOP_K_FINAL=5
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# GLM-OCR local server
GLM_HOST=http://localhost
GLM_PORT=8000
GLM_MODEL=glm-ocr
GLM_TIMEOUT=120
GLM_MAX_TOKENS=2048
GLM_TEMPERATURE=0.1

# Offline mode (set to 1 after first model download)
TRANSFORMERS_OFFLINE=1
```

---

## Troubleshooting

**`faiss` not found**
```bash
pip install faiss-cpu
```

**GLM-OCR not responding**
- Check your model server is running: `curl http://localhost:8000/health`
- Update `GLM_PORT` in `.env` to match your actual port
- Check `GLM_MODEL` matches the model name your server expects

**Out of memory during embedding**
- Reduce `EMBED_BATCH` in `.env` to `8` or `16`
- Or set `EMBED_DEVICE=cpu` (uses less memory than GPU batching)

**`bge-m3` download fails**
- Run once with internet on, then set `TRANSFORMERS_OFFLINE=1`
- Or manually download and place in `models_cache/`

**Chunks not finding relevant content**
- Increase `TOP_K_DENSE` and `TOP_K_SPARSE` to cast a wider net
- Reduce `CHUNK_SIZE` to `256` for more precise retrieval on short facts
- Check the source document was actually loaded (Section 1 output)
