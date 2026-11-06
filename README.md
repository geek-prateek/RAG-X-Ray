# 🔬 RAG X-Ray — Inspect Every Step of the Retrieval-Augmented Generation Pipeline

[![Angular](https://img.shields.io/badge/Angular-20-DD0031?logo=angular&logoColor=white)](https://angular.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://python.org/)
[![ChromaDB](https://img.shields.io/badge/Vector_DB-ChromaDB-FF6600)](https://www.trychroma.com/)
[![FastEmbed](https://img.shields.io/badge/Embeddings-FastEmbed_ONNX_(384D)-yellow)](https://github.com/qdrant/fastembed)
[![MCP Server](https://img.shields.io/badge/MCP_Server-Live_SSE-blueviolet)](https://rag-xray-backend.onrender.com/mcp/sse)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**RAG X-Ray** is an educational, interactive, open-source tool designed to demystify how Retrieval-Augmented Generation (RAG) actually works under the hood. Instead of treating RAG as a black box with opaque developer traces, RAG X-Ray opens up the pipeline into a visual, step-by-step **Before → Operation → After → Next** data journey with real intermediate values.

---

## ✨ Key Features

### 1. 📂 Document Ingestion Pipeline (6 Stages)
- **Upload:** Supports PDF, Markdown, and TXT files (up to 10 MB).
- **Text Extraction & Cleaning:** Extracts page-by-page raw text, strips noise and excessive whitespace.
- **Word Chunking:** Splits documents into overlapping word chunks (default: 90 words with 18-word overlap).
- **Embedding Generation:** Generates real 384-dimensional normalized dense vectors using `sentence-transformers/all-MiniLM-L6-v2`.
- **Vector Storage:** Persists embeddings and chunk metadata into ChromaDB.

### 2. ❓ Question Answering Pipeline (5 Stages)
- **Question Vectorization:** Encodes user queries into normalized 384D query vectors.
- **Vector Dot Product & Cosine Similarity:** Computes semantic similarity against all stored document chunks.
- **Ranking & Selection:** Ranks passage candidates from highest to lowest similarity score with visual score bars.
- **Answer Synthesis:** Synthesizes answers using a local **Ollama LLM** (e.g. `llama3.2:3b`), or falls back transparently to a clearly-labeled **Extractive Fallback** when Ollama is offline.
- **Source Grounding & Citation Inspection:** Click any citation pill to view the exact passage, chunk index, and similarity score.

### 3. ⚡ Interactive Top-K What-If Experimentation
- Dynamically adjust the **Top-K Context Chunks** slider (from 1 to N).
- Instant preview of which document passages are included or excluded based on the cutoff threshold.
- Click **"⚡ Experiment: Change Top-K"** to smoothly open and auto-scroll directly to the experiment workbench on desktop and mobile.
- **Re-run with Top-K:** Re-execute the query with the new context limit and watch the generated answer adapt live.

### 4. 🌳 Interactive Collapsible JSON Viewer
- **Tree View vs. Raw View:** Switch between an interactive collapsible tree and formatted raw JSON.
- **Expand / Collapse All:** Instantly inspect high-level structures or drill deep into chunk matrices.
- **Copy JSON:** One-click clipboard copy with visual confirmation.
- **Clarified Variable Names:** Self-documenting keys (e.g., `candidate_matches`, `similarity_scores`, `document_chunks`) designed for learners and beginners.

### 5. 🗂️ Workspaces & Local Session Isolation
- Create, rename, switch, and delete isolated workspaces.
- Per-browser session isolation (`data/sessions/<session_id>/`) guarantees queries and documents don't collide.
- Document selection checkboxes allow users to toggle specific documents in/out of scope for new chats.

### 6. 📱 Responsive & Mobile-Optimized
- Full mobile-adaptive layout with dedicated bottom tab navigation (`Library`, `Chat`, `Data Journey`).
- Seamless transitions between document exploration, query tracing, and LLM output.

### 7. 🔌 Model Context Protocol (MCP) Server
- **Remote SSE Endpoint:** Live on `https://rag-xray-backend.onrender.com/mcp/sse`.
- **AI Agent Integration:** Connect Cursor IDE, Claude Desktop, or Windsurf directly without installing Python.
- **RAG Inspector & Evaluator:** 6 specialized diagnostic tools to audit cosine scores, score dropoff, chunk boundaries, and hallucination risks.

---

## 🏛️ Architecture & Data Flow

```text
┌─────────────────────────────────────────────────────────────┐
│                    Angular 20 Frontend                      │
│  - Document Library & Workspaces                            │
│  - 6-Stage Document Journey Drawer                          │
│  - 5-Stage Question Journey Drawer                          │
│  - Interactive Collapsible JSON Inspector                   │
│  - Top-K Slider & Live Re-run Engine                        │
└──────────────────────────────┬──────────────────────────────┘
                               │ REST / JSON (FastAPI)
┌──────────────────────────────▼──────────────────────────────┐
│                    FastAPI Backend Core                     │
│  ┌───────────────────────┐      ┌────────────────────────┐  │
│  │   Document Pipeline   │      │   Question Pipeline    │  │
│  │ - PDF/TXT Extraction  │      │ - Query Embedding      │  │
│  │ - Text Cleaning       │      │ - Cosine Similarity    │  │
│  │ - Word Chunking       │      │ - Top-K Candidate Rank │  │
│  └───────────┬───────────┘      └───────────┬────────────┘  │
│              │                              │               │
│              ▼                              ▼               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │       Sentence Transformers (all-MiniLM-L6-v2)        │  │
│  └───────────────────────────┬───────────────────────────┘  │
│                              │                              │
│              ┌───────────────┴───────────────┐              │
│              ▼                               ▼              │
│  ┌──────────────────────┐        ┌───────────────────────┐  │
│  │ ChromaDB Vector DB   │        │ Local Ollama (LLM)    │  │
│  │ - Isolated Sessions  │        │ - llama3.2:3b         │  │
│  │ - Stored Embeddings  │        │ - Extractive Fallback │  │
│  └──────────────────────┘        └───────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 📋 Prerequisites

- **Python:** `3.11` or `3.12` recommended.
- **Node.js:** `20.19+` or `22.12+` with `npm`.
- **(Optional) Ollama:** For local LLM answer generation ([Download Ollama](https://ollama.com/)).

---

## 🔌 Model Context Protocol (MCP) Server

RAG X-Ray includes a built-in **Inspector & Evaluator MCP Server** hosted live via Server-Sent Events (SSE). It transforms RAG X-Ray from a standalone playground into a remote diagnostic engine for AI assistants such as **Cursor IDE**, **Claude Desktop**, and **Antigravity / Windsurf**.

### Live SSE Endpoint (No Local Setup Required)
```http
https://rag-xray-backend.onrender.com/mcp/sse
```

### 🛠️ Connecting to Cursor IDE

#### Option A: Settings UI (1-Click)
1. Open Cursor ➔ **Cursor Settings** (Gear icon or `Ctrl+,` / `Cmd+,`) ➔ **Features** ➔ **MCP**.
2. Click **+ Add New MCP Server**:
   - **Name:** `rag-xray`
   - **Type:** `sse`
   - **URL:** `https://rag-xray-backend.onrender.com/mcp/sse`
3. Save. Once connected (green status indicator), all 6 diagnostic tools will be accessible to Cursor's agent.

#### Option B: Workspace Config (`.cursor/mcp.json`)
If you open this repository in Cursor, it is already pre-configured in `.cursor/mcp.json`:
```json
{
  "mcpServers": {
    "rag-xray": {
      "url": "https://rag-xray-backend.onrender.com/mcp/sse"
    }
  }
}
```

---

### 🧰 Available MCP Diagnostic & Evaluation Tools

| MCP Tool | Purpose & Diagnostic Value |
| :--- | :--- |
| **`evaluate_retrieval(query, top_k)`** | Evaluates query against knowledge base. Returns **Confidence Verdict** (`EXCELLENT`, `MODERATE`, `POOR_MATCH_HIGH_HALLUCINATION_RISK`), cosine score, and score dropoff to prevent LLM hallucinations. |
| **`compare_semantic_similarity(text_a, text_b)`** | Computes exact cosine similarity (-1.0 to 1.0) and semantic overlap percentage between two texts to diagnose query-chunk alignment. |
| **`inspect_vector_embedding(text)`** | Inspects the 384-dimensional dense ONNX vector, L2 norm, and first 8 float preview components for any text. |
| **`diagnose_document_chunking(document_id)`** | Audits sliding-window chunks (90 words, 18 overlap), word counts, and boundary cuts for anomalies. |
| **`list_indexed_documents()`** | Lists all indexed documents, chunk counts, page counts, and indexing readiness. |
| **`index_document_text(filename, content)`** | Ingests raw text, code, or policies directly from Cursor/Claude into the vector store. |

---

### 💬 Real-World Demo Prompts in Cursor

Once connected, ask Cursor:

- **Retrieval Evaluation:**
  > `@rag-xray evaluate_retrieval for question 'What is the policy for annual leave?' and report the confidence score and hallucination risk.`

- **Semantic Comparison:**
  > `@rag-xray compare_semantic_similarity between 'Employees request leave on portal' and 'I want to take 3 days off'.`

- **Vector Math Inspection:**
  > `@rag-xray inspect_vector_embedding for the phrase 'Retrieval-Augmented Generation pipeline'.`

---

## 🚀 Quick Start Guide

### 1. Clone Repository
```bash
git clone https://github.com/your-username/rag-xray.git
cd rag-xray
```

### 2. Backend Setup

#### On Windows (PowerShell):
```powershell
cd backend
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

#### On Linux / macOS:
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

* Backend API Docs will be available at: **`http://127.0.0.1:8000/docs`**

---

### 3. Frontend Setup

In a new terminal:
```bash
cd frontend
npm install
npm start
```

* Open your browser at: **`http://localhost:4200`**

---

### 4. (Optional) Enable Local LLM Answers with Ollama

If you want actual generative answers instead of the extractive fallback:

```bash
# Pull lightweight llama model
ollama pull llama3.2:3b

# Start the Ollama server
ollama serve
```

*Default environment variables:*
- `OLLAMA_URL=http://localhost:11434`
- `OLLAMA_MODEL=llama3.2:3b`

If Ollama is not installed or unreachable, RAG X-Ray automatically uses high-accuracy **Extractive Fallback** so you can continue learning without any external setup.

---

## 🐳 Docker Deployment

You can also run the backend inside Docker:

```bash
docker compose up --build
```

Then run `npm install && npm start` inside the `frontend/` directory.

---

## 🧪 Testing

Run backend unit and API tests:

```bash
cd backend
python -m pytest -q
```

All 6 test suites verify text extraction, chunk boundary preservation, vector scoring, and API contract validity.

---

## 📁 Project Structure

```text
rag-xray/
├── .cursor/
│   └── mcp.json            # Pre-configured Cursor MCP SSE integration
├── backend/
│   ├── app/
│   │   ├── engine.py       # FastEmbed ONNX chunking, embeddings, cosine similarity, LLM client
│   │   ├── main.py         # FastAPI endpoints, MCP SSE mount, session routes, data journey
│   │   ├── mcp_server.py   # RAG X-Ray Inspector & Evaluator MCP server (6 diagnostic tools)
│   │   └── storage.py      # Local session isolation, document metadata, ChromaDB persistence
│   ├── tests/              # Pytest test suites
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── app.component.html   # Main layout, drawer journeys, JSON tree viewer
│   │   │   ├── app.component.ts     # State management, Top-K runner, scrolling logic
│   │   │   └── app.component.css    # Responsive styles, pulse animations, dark accents
│   │   └── index.html
│   ├── angular.json
│   └── package.json
├── sample_docs/            # Sample files for quick testing (e.g. employee handbooks)
├── docker-compose.yml
├── .gitignore              # Clean rules (virtual environments, caches, node_modules)
└── README.md
```

---

## 🛡️ License

This project is licensed under the [MIT License](LICENSE).
