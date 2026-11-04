"""
Engine Module for RAG X-Ray
===========================
Contains all core RAG calculations and algorithms:
1. Text extraction from files (PyMuPDF for PDF, utf-8 for TXT/MD).
2. Text cleaning and whitespace normalization.
3. Sliding-window chunking (preserving context across word boundaries).
4. Generating 384-dimensional dense embeddings (Sentence Transformers).
5. Vector Cosine Similarity calculation with pure NumPy math.
6. Generating structured answers with local Ollama LLM.
"""
from __future__ import annotations

import hashlib
import io
import os
import re
import threading
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF library for fast PDF text reading
import numpy as np
import requests

# Lazy-loaded embedding model to save memory until first needed
_EMBEDDER = None
_EMBED_LOCK = threading.Lock()
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def get_embedder():
    """
    Loads and returns the SentenceTransformer model.
    Uses lazy-loading (thread-safe) so the server starts instantly.
    """
    global _EMBEDDER
    if _EMBEDDER is None:
        with _EMBED_LOCK:
            if _EMBEDDER is None:
                from sentence_transformers import SentenceTransformer
                _EMBEDDER = SentenceTransformer(MODEL_NAME)
    return _EMBEDDER


def embed(texts: list[str]) -> np.ndarray:
    """
    Converts a list of text strings into normalized vector embeddings.
    Each vector has 384 floating-point numbers.
    'normalize_embeddings=True' scales vector length to 1.0,
    which makes the dot product equal to cosine similarity.
    """
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    embedder = get_embedder()
    vectors = embedder.encode(texts, normalize_embeddings=True)
    return np.asarray(vectors, dtype=np.float32)


def excerpt(text: str, limit: int = 340) -> str:
    """
    Shortens long text strings with an ellipsis (…) for preview in the UI.
    """
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def preview_vector(vector: np.ndarray, count: int = 8) -> list[float]:
    """
    Returns the first few rounded float values of a vector (e.g. [0.045, -0.012, ...])
    to display in the UI without overwhelming the screen.
    """
    return [round(float(val), 6) for val in vector[:count]]


def extract(filename: str, content: bytes) -> list[dict[str, Any]]:
    """
    Extracts text page-by-page from an uploaded file:
    - For PDF: uses PyMuPDF (fitz) to read each page text.
    - For TXT/MD: decodes bytes into text as page 1.
    """
    extension = Path(filename).suffix.lower()

    if extension == '.pdf':
        pages = []
        with fitz.open(stream=content, filetype="pdf") as pdf_doc:
            for page_index, page in enumerate(pdf_doc):
                page_text = page.get_text("text")
                pages.append({
                    "page": page_index + 1,
                    "text": page_text
                })

        # Ensure the PDF contains readable text (not just scanned image scans)
        has_text = any(p["text"].strip() for p in pages)
        if not has_text:
            raise ValueError("No readable PDF text found. Scanned PDFs need OCR.")
        return pages

    if extension in {'.md', '.txt'}:
        # Decode text files (handles UTF-8 with or without BOM)
        text = content.decode('utf-8-sig')
        return [{"page": 1, "text": text}]

    raise ValueError("Only PDF, TXT and Markdown files are supported.")


def clean(text: str) -> str:
    """
    Normalizes whitespace (replaces multiple spaces, tabs, and newlines with a single space)
    while keeping all original words intact.
    """
    return re.sub(r'\s+', ' ', text).strip()


def chunk_pages(pages: list[dict], size: int = 90, overlap: int = 18) -> list[dict]:
    """
    Splits document text into overlapping chunks using a sliding window.

    Why overlap?
    If a sentence gets cut off at the 90th word, the 18-word overlap ensures
    the next chunk starts 18 words earlier, preventing context loss.

    - size: Number of words per chunk (default: 90 words)
    - overlap: Number of shared words between consecutive chunks (default: 18 words)
    """
    if not (0 <= overlap < size):
        raise ValueError("overlap must be >= 0 and smaller than chunk size")

    chunks = []
    step = size - overlap  # Advance window by (size - overlap) words each step

    for page in pages:
        words = page['text'].split()
        for start in range(0, len(words), step):
            chunk_words = words[start:start + size]
            if not chunk_words:
                continue

            chunk_text = ' '.join(chunk_words)
            chunks.append({
                "page": page['page'],
                "start_word": start,
                "end_word": start + len(chunk_words),
                "text": chunk_text
            })

            # Stop when we have covered the end of this page's words
            if start + size >= len(words):
                break

    return chunks


def cosines(query_vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Calculates Cosine Similarity between a query vector and a matrix of chunk vectors.

    Formula:
        Cosine Similarity = (A • B) / (||A|| * ||B||)

    Returns:
        1D array of float scores between -1.0 and 1.0 (higher = more relevant).
    """
    q = np.asarray(query_vector, dtype=np.float64).reshape(-1)
    m = np.asarray(matrix, dtype=np.float64)

    # Calculate norms (magnitudes)
    matrix_norms = np.linalg.norm(m, axis=1)
    query_norm = np.linalg.norm(q)
    norms = matrix_norms * query_norm

    # Dot product divided by norms
    dot_products = m @ q
    scores = np.divide(
        dot_products,
        norms,
        out=np.zeros(m.shape[0], dtype=np.float64),
        where=norms > 0
    )
    return scores


def ollama_answer(question: str, contexts: list[dict]) -> tuple[str, str]:
    """
    Sends the user's question and retrieved context chunks to Ollama
    to generate a grounded, ChatGPT-style structured answer with citations [1], [2].

    If Ollama is down, returns clearly labeled extractive excerpts as a fallback.
    """
    if not contexts:
        return "I couldn't find supporting evidence in the uploaded documents.", "no_evidence"

    # Format the retrieved chunks with citation numbers [1], [2], etc.
    formatted_contexts = []
    for index, chunk in enumerate(contexts):
        citation_number = index + 1
        formatted_contexts.append(f"[{citation_number}] {chunk['text']}")
    evidence_text = '\n\n'.join(formatted_contexts)

    # Clear instructions to produce structured, ChatGPT-style output
    system_prompt = (
        "You are an expert AI assistant like ChatGPT.\n"
        "Your task is to answer the user's question clearly, thoroughly, and professionally based strictly on the provided EVIDENCE.\n\n"
        "Formatting & Style Guidelines:\n"
        "1. Structure your answer like ChatGPT: Start directly with a clear summary or direct answer. Use clean Markdown formatting with bold highlights (**key term**), numbered steps or bullet points, and neat paragraphs.\n"
        "2. Language: If the user asks in Hindi or Hinglish, explain in natural, professional Hindi/Hinglish. If they ask in English, answer in English.\n"
        "3. Never echo or repeat the user's question at the start of your answer.\n"
        "4. Integrate citations naturally using [1], [2], [3] at the end of claims or steps where that evidence is used.\n"
        "5. If evidence is insufficient, politely state what is missing instead of inventing facts.\n"
        "6. Do not invent missing facts outside the evidence."
    )
    user_prompt = f"EVIDENCE:\n{evidence_text}\n\nQUESTION:\n{question}"

    model = os.getenv('OLLAMA_MODEL', 'qwen2.5-coder:7b')
    base_url = os.getenv('OLLAMA_URL', 'http://localhost:11434').rstrip('/')

    # 1. Try Ollama's conversational /api/chat endpoint first
    try:
        chat_endpoint = f"{base_url}/api/chat"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": {"temperature": 0.2}
        }
        response = requests.post(chat_endpoint, json=payload, timeout=75)
        if response.status_code == 200:
            answer = response.json().get('message', {}).get('content', '').strip()
            if answer:
                # Remove question if the model accidentally repeated it on the first line
                lines = answer.splitlines()
                if lines:
                    first_clean = re.sub(r'\[\d+\]', '', lines[0]).strip().rstrip('?.:').lower()
                    if first_clean == question.strip().rstrip('?.:').lower():
                        answer = '\n'.join(lines[1:]).strip()
                return answer, f"ollama:{model}"
    except (requests.RequestException, ValueError):
        pass

    # 2. Fallback to /api/generate if /api/chat is not supported
    try:
        gen_endpoint = f"{base_url}/api/generate"
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"
        response = requests.post(
            gen_endpoint,
            json={"model": model, "prompt": combined_prompt, "stream": False, "options": {"temperature": 0.2}},
            timeout=75
        )
        if response.status_code == 200:
            answer = response.json().get('response', '').strip()
            if answer:
                lines = answer.splitlines()
                if lines:
                    first_clean = re.sub(r'\[\d+\]', '', lines[0]).strip().rstrip('?.:').lower()
                    if first_clean == question.strip().rstrip('?.:').lower():
                        answer = '\n'.join(lines[1:]).strip()
                return answer, f"ollama:{model}"
    except (requests.RequestException, ValueError):
        pass

    # 3. Safe fallback if Ollama is unavailable: extract the first sentence from each chunk
    excerpts = []
    for index, chunk in enumerate(contexts):
        first_sentence = re.split(r'(?<=[.!?])\s+', chunk['text'])[0]
        excerpts.append(f"{excerpt(first_sentence, 260)} [{index + 1}]")

    fallback_message = (
        "Ollama unavailable. Here are the closest source excerpts (not AI-generated):\n"
        + '\n'.join(excerpts)
    )
    return fallback_message, "extractive_fallback"


def sha256(content: bytes) -> str:
    """
    Computes and returns the SHA-256 hash of raw file bytes for integrity check.
    """
    return hashlib.sha256(content).hexdigest()
