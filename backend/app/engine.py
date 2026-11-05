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


def _clean_llm_response(answer: str, question: str) -> str:
    """Removes echoed user question if present on the first line of the answer."""
    lines = answer.splitlines()
    if lines:
        first_clean = re.sub(r'\[\d+\]', '', lines[0]).strip().rstrip('?.:').lower()
        if first_clean == question.strip().rstrip('?.:').lower():
            return '\n'.join(lines[1:]).strip()
    return answer.strip()


def call_groq(system_prompt: str, user_prompt: str, model: str | None = None, api_key: str | None = None) -> tuple[str, str]:
    """Calls Groq Cloud API for ultra-fast Llama-3.3 / Mixtral inference."""
    chosen_model = (model or os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')).strip()
    key = api_key or os.getenv('GROQ_API_KEY')
    if not key:
        raise ValueError('Groq API Key is missing. Please click the model settings ⚙ in the chat bar to enter your Groq API key.')

    headers = {
        'Authorization': f'Bearer {key.strip()}',
        'Content-Type': 'application/json'
    }
    payload = {
        'model': chosen_model,
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
        'temperature': 0.2
    }
    resp = requests.post('https://api.groq.com/openai/v1/chat/completions', headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        error_detail = resp.json().get('error', {}).get('message', resp.text)
        raise ValueError(f'Groq API error ({resp.status_code}): {error_detail}')
    content = resp.json()['choices'][0]['message']['content']
    return content, chosen_model


def call_openai(system_prompt: str, user_prompt: str, model: str | None = None, api_key: str | None = None) -> tuple[str, str]:
    """Calls OpenAI API for GPT-4o-mini / GPT-4o inference."""
    chosen_model = (model or os.getenv('OPENAI_MODEL', 'gpt-4o-mini')).strip()
    key = api_key or os.getenv('OPENAI_API_KEY')
    if not key:
        raise ValueError('OpenAI API Key is missing. Please click the model settings ⚙ in the chat bar to enter your OpenAI API key.')

    headers = {
        'Authorization': f'Bearer {key.strip()}',
        'Content-Type': 'application/json'
    }
    payload = {
        'model': chosen_model,
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
        'temperature': 0.2
    }
    resp = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        error_detail = resp.json().get('error', {}).get('message', resp.text)
        raise ValueError(f'OpenAI API error ({resp.status_code}): {error_detail}')
    content = resp.json()['choices'][0]['message']['content']
    return content, chosen_model


def call_gemini(system_prompt: str, user_prompt: str, model: str | None = None, api_key: str | None = None) -> tuple[str, str]:
    """Calls Google Gemini API for Gemini-1.5-Flash / Gemini-1.5-Pro inference."""
    raw_model = (model or os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')).strip()
    chosen_model = raw_model.replace('models/', '').strip()
    key = api_key or os.getenv('GEMINI_API_KEY')
    if not key:
        raise ValueError('Google Gemini API Key is missing. Please click the model settings ⚙ in the chat bar to enter your Gemini API key.')

    url = f'https://generativelanguage.googleapis.com/v1beta/models/{chosen_model}:generateContent?key={key.strip()}'
    payload = {
        'system_instruction': {
            'parts': [{'text': system_prompt}]
        },
        'contents': [
            {
                'role': 'user',
                'parts': [{'text': user_prompt}]
            }
        ],
        'generationConfig': {
            'temperature': 0.2
        }
    }
    resp = requests.post(url, json=payload, timeout=60)
    if resp.status_code != 200:
        error_detail = resp.json().get('error', {}).get('message', resp.text)
        raise ValueError(f'Gemini API error ({resp.status_code}): {error_detail}')
    candidates = resp.json().get('candidates', [])
    if not candidates:
        raise ValueError('Gemini returned no response candidates.')
    parts = candidates[0].get('content', {}).get('parts', [])
    content = parts[0].get('text', '') if parts else ''
    return content, chosen_model


def call_ollama(system_prompt: str, user_prompt: str, model: str | None = None) -> tuple[str, str]:
    """Calls local Ollama API."""
    chosen_model = model or os.getenv('OLLAMA_MODEL', 'qwen2.5-coder:7b')
    base_url = os.getenv('OLLAMA_URL', 'http://localhost:11434').rstrip('/')

    # 1. Try /api/chat
    chat_endpoint = f"{base_url}/api/chat"
    payload = {
        "model": chosen_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False,
        "options": {"temperature": 0.2}
    }
    try:
        resp = requests.post(chat_endpoint, json=payload, timeout=75)
        if resp.status_code == 200:
            content = resp.json().get('message', {}).get('content', '').strip()
            if content:
                return content, chosen_model
    except Exception:
        pass

    # 2. Try /api/generate
    gen_endpoint = f"{base_url}/api/generate"
    combined_prompt = f"{system_prompt}\n\n{user_prompt}"
    resp = requests.post(
        gen_endpoint,
        json={"model": chosen_model, "prompt": combined_prompt, "stream": False, "options": {"temperature": 0.2}},
        timeout=75
    )
    if resp.status_code == 200:
        content = resp.json().get('response', '').strip()
        if content:
            return content, chosen_model

    raise ValueError(f"Could not connect to local Ollama at {base_url} with model '{chosen_model}'. Make sure Ollama is running, or switch AI engine to Groq or Gemini in settings.")


def generate_llm_answer(
    question: str,
    contexts: list[dict],
    provider: str = 'ollama',
    model: str | None = None,
    api_key: str | None = None
) -> tuple[str, str, str]:
    """
    Unified multi-provider LLM answer generator:
    Supports: 'ollama' (local), 'groq' (cloud), 'openai' (cloud), 'gemini' (cloud).
    Returns: (answer_text, actual_model_name, provider_name)
    """
    default_model_map = {
        'groq': os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile'),
        'gemini': os.getenv('GEMINI_MODEL', 'gemini-1.5-flash'),
        'openai': os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
        'ollama': os.getenv('OLLAMA_MODEL', 'qwen2.5-coder:7b')
    }
    provider_clean = (provider or 'ollama').lower().strip()
    active_requested_model = (model or default_model_map.get(provider_clean, 'qwen2.5-coder:7b')).strip()

    if not contexts:
        return "I couldn't find supporting evidence in the uploaded documents.", active_requested_model, "no_evidence"

    # Format retrieved chunks with citation numbers [1], [2], etc.
    formatted_contexts = []
    for index, chunk in enumerate(contexts):
        formatted_contexts.append(f"[{index + 1}] {chunk['text']}")
    evidence_text = '\n\n'.join(formatted_contexts)

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

    try:
        if provider_clean == 'groq':
            raw_answer, active_model = call_groq(system_prompt, user_prompt, model, api_key)
            cleaned = _clean_llm_response(raw_answer, question)
            return cleaned, active_model, 'groq'
        elif provider_clean == 'openai':
            raw_answer, active_model = call_openai(system_prompt, user_prompt, model, api_key)
            cleaned = _clean_llm_response(raw_answer, question)
            return cleaned, active_model, 'openai'
        elif provider_clean == 'gemini':
            raw_answer, active_model = call_gemini(system_prompt, user_prompt, model, api_key)
            cleaned = _clean_llm_response(raw_answer, question)
            return cleaned, active_model, 'gemini'
        else:
            raw_answer, active_model = call_ollama(system_prompt, user_prompt, model)
            cleaned = _clean_llm_response(raw_answer, question)
            return cleaned, active_model, 'ollama'
    except Exception as exc:
        excerpts = []
        for index, chunk in enumerate(contexts):
            first_sentence = re.split(r'(?<=[.!?])\s+', chunk['text'])[0]
            excerpts.append(f"{excerpt(first_sentence, 260)} [{index + 1}]")

        prov_display = {
            'groq': 'Groq Cloud',
            'gemini': 'Google Gemini',
            'openai': 'OpenAI',
            'ollama': 'Local Ollama'
        }.get(provider_clean, provider_clean.title())

        fallback_msg = (
            f"**Notice ({prov_display}):** {str(exc)}\n\n"
            "**Retrieved source evidence passages:**\n\n"
            + '\n\n'.join(excerpts)
        )
        return fallback_msg, active_requested_model, "extractive_fallback"


def ollama_answer(question: str, contexts: list[dict], model: str | None = None) -> tuple[str, str]:
    """
    Backward-compatible wrapper for existing tests and callers.
    """
    answer, active_model, provider = generate_llm_answer(question, contexts, provider='ollama', model=model)
    return answer, f"{provider}:{active_model}"


def sha256(content: bytes) -> str:
    """
    Computes and returns the SHA-256 hash of raw file bytes for integrity check.
    """
    return hashlib.sha256(content).hexdigest()
