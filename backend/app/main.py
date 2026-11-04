"""
Main FastAPI Server for RAG X-Ray
=================================
This file provides the HTTP API endpoints for:
1. Document management (upload, process, list, get, delete).
2. Question answering (search relevant chunks with vector cosine similarity and generate LLM answer).

Key Design Feature:
Every processing step (stages/steps) records its 'before' state,
the operation performed, and the 'after' state so beginners can see
the exact transformations happening inside the RAG system.
"""
from __future__ import annotations

import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import engine, storage

# Initialize FastAPI application
app = FastAPI(
    title='RAG X-Ray',
    version='1.0.0',
    description='Beginner-friendly, inspectable RAG pipeline'
)

# Configure CORS so our Angular frontend (localhost:4200) can communicate with the backend
allowed_origins_env = os.getenv('CORS_ORIGINS', 'http://localhost:4200')
allowed_origins = [origin.strip() for origin in allowed_origins_env.split(',')]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=['GET', 'POST', 'DELETE'],
    allow_headers=['*']
)

# Background workers for document processing (so file upload returns immediately without freezing)
workers = ThreadPoolExecutor(max_workers=2)

# File upload limit (configurable via MAX_FILE_SIZE_MB, default 50 MB)
MAX_FILE_SIZE_MB = int(os.getenv('MAX_FILE_SIZE_MB', '50'))
MAX_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


def record_stage(doc: dict, name: str, before: Any, operation: str, after: Any, next_step: str):
    """
    Helper function to record each transformation stage of a document.
    Stores 'before', 'operation', and 'after' data so the user can inspect it in the UI.
    """
    doc['stages'].append({
        "name": name,
        "before": before,
        "operation": operation,
        "after": after,
        "next": next_step
    })
    storage.put(doc)


def process_document_background(doc_id: str, content: bytes):
    """
    Runs in the background to process an uploaded document through 6 clear stages:
    1. Upload & Hash check (SHA-256)
    2. Extract raw text page-by-page
    3. Clean and normalize whitespace
    4. Split into overlapping word chunks (size=90, overlap=18)
    5. Convert chunks into 384-dimensional vector embeddings
    6. Store chunks and vectors into ChromaDB
    """
    doc = storage.get(doc_id)
    if not doc:
        return

    try:
        doc['status'] = 'processing'
        storage.put(doc)

        # ---------------------------------------------------------
        # STAGE 1: Upload & Validate
        # ---------------------------------------------------------
        file_sha256 = engine.sha256(content)
        record_stage(
            doc=doc,
            name='Upload',
            before={'filename': doc['filename'], 'bytes': len(content)},
            operation='Validate file format and size; calculate SHA-256 fingerprint',
            after={'sha256': file_sha256, 'accepted': True},
            next_step='Extract document text'
        )

        # ---------------------------------------------------------
        # STAGE 2: Extract Text Page-by-Page
        # ---------------------------------------------------------
        pages = engine.extract(doc['filename'], content)
        pages_preview = [
            {
                'page': p['page'],
                'text': engine.excerpt(p['text'], 600),
                'characters': len(p['text'])
            }
            for p in pages
        ]
        record_stage(
            doc=doc,
            name='Extract text',
            before={'pages_in_file': len(pages)},
            operation='Read text page by page using PyMuPDF',
            after={'page_count': len(pages), 'pages': pages_preview},
            next_step='Clean extracted text'
        )

        # ---------------------------------------------------------
        # STAGE 3: Clean & Normalize Text
        # ---------------------------------------------------------
        cleaned_pages = [
            {'page': p['page'], 'text': engine.clean(p['text'])}
            for p in pages
        ]
        total_cleaned_words = sum(len(p['text'].split()) for p in cleaned_pages)
        record_stage(
            doc=doc,
            name='Clean text',
            before={'first_page': engine.excerpt(pages[0]['text'])},
            operation='Normalize excessive whitespace and line breaks while preserving words',
            after={
                'first_page': engine.excerpt(cleaned_pages[0]['text']),
                'total_words': total_cleaned_words
            },
            next_step='Split into chunks'
        )

        # ---------------------------------------------------------
        # STAGE 4: Make Chunks (Sliding Window)
        # ---------------------------------------------------------
        chunk_size = 90
        chunk_overlap = 18
        chunks = engine.chunk_pages(cleaned_pages, size=chunk_size, overlap=chunk_overlap)
        if not chunks:
            raise ValueError('The document contains no usable text after cleaning.')

        record_stage(
            doc=doc,
            name='Make chunks',
            before={
                'cleaned_pages': len(cleaned_pages),
                'chunk_size_words': chunk_size,
                'overlap_words': chunk_overlap
            },
            operation='Split each page into overlapping word-based chunks',
            after={'count': len(chunks), 'preview': chunks[:6]},
            next_step='Convert each chunk into an embedding vector'
        )

        # ---------------------------------------------------------
        # STAGE 5: Create Vector Embeddings
        # ---------------------------------------------------------
        chunk_texts = [c['text'] for c in chunks]
        vectors = engine.embed(chunk_texts)
        if len(vectors) != len(chunks):
            raise ValueError('Embedding count mismatch')

        first_vector_norm = round(float(np.linalg.norm(vectors[0])), 6)
        first_vectors_preview = [
            {
                'chunk_id': f"{doc_id}_{i}",
                'values': [round(float(val), 6) for val in vec]
            }
            for i, vec in enumerate(vectors[:5])
        ]

        record_stage(
            doc=doc,
            name='Create embeddings',
            before={'chunks': len(chunks), 'first_chunk': engine.excerpt(chunks[0]['text'])},
            operation='Convert text chunks into 384-dimensional dense vectors using SentenceTransformer',
            after={
                'model': engine.MODEL_NAME,
                'shape': list(vectors.shape),
                'first_vectors': first_vectors_preview,
                'first_vector_norm': first_vector_norm
            },
            next_step='Save vectors and chunk references in the index'
        )

        # ---------------------------------------------------------
        # STAGE 6: Save to ChromaDB Vector Store
        # ---------------------------------------------------------
        chunk_ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                'doc_id': doc_id,
                'filename': doc['filename'],
                'page': c['page'],
                'start_word': c['start_word']
            }
            for c in chunks
        ]

        chroma_col = storage.collection()
        chroma_col.upsert(
            ids=chunk_ids,
            embeddings=vectors.tolist(),
            documents=chunk_texts,
            metadatas=metadatas
        )

        # Store chunk references in the document object
        doc['chunks'] = [
            {**c, 'id': chunk_ids[i]}
            for i, c in enumerate(chunks)
        ]

        record_stage(
            doc=doc,
            name='Save to vector DB',
            before={'vector_count': len(vectors), 'dimension': int(vectors.shape[1])},
            operation='Upsert chunks and vectors into persistent ChromaDB collection',
            after={
                'stored_count': len(chunk_ids),
                'first_ids': chunk_ids[:5],
                'collection': 'rag_xray_chunks'
            },
            next_step='Ask a question about this document'
        )

        doc['status'] = 'ready'

    except Exception as exc:
        doc['status'] = 'failed'
        doc['error'] = str(exc)

    finally:
        storage.put(doc)


# Backward-compatible alias for any background caller
process = process_document_background


# =====================================================================
# HTTP API Endpoints
# =====================================================================

@app.get('/api/health')
def health():
    """Health check endpoint to verify backend is up."""
    return {'status': 'ok'}


@app.get('/api/documents')
def list_documents():
    """
    Returns a lightweight summary of all uploaded documents:
    their ID, filename, status, error (if any), stage count, and chunk count.
    """
    all_stored = storage.all_docs()
    summary_list = []
    for d in all_stored:
        summary_list.append({
            'id': d.get('id'),
            'filename': d.get('filename'),
            'status': d.get('status'),
            'error': d.get('error'),
            'stage_count': len(d.get('stages', [])),
            'chunk_count': len(d.get('chunks', []))
        })
    return summary_list


@app.get('/api/documents/{doc_id}')
def get_document(doc_id: str):
    """
    Returns full details for a single document, including all stages and chunks.
    """
    doc = storage.get(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail='Document not found')
    return doc


@app.post('/api/documents', status_code=202)
async def upload_document(file: UploadFile = File(...)):
    """
    Uploads a document (.pdf, .txt, .md), creates a queued record,
    and starts processing in the background thread.
    """
    file_extension = Path(file.filename or '').suffix.lower()
    if file_extension not in {'.pdf', '.txt', '.md'}:
        raise HTTPException(status_code=400, detail='Supported formats: PDF, TXT, MD')

    # Read content up to max limit + 1 byte (to detect oversized files)
    content = await file.read(MAX_BYTES + 1)
    if not content or len(content) > MAX_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f'File must be nonempty and no more than {MAX_FILE_SIZE_MB} MB'
        )

    safe_filename = Path(file.filename or 'upload').name
    doc = storage.new_doc(safe_filename)

    # Hand off processing to background thread
    workers.submit(process_document_background, doc['id'], content)

    return {'document_id': doc['id'], 'status': 'queued'}


@app.delete('/api/documents/{doc_id}')
def delete_document(doc_id: str):
    """
    Deletes a document from the local store and removes all its chunks from ChromaDB.
    """
    doc = storage.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail='Document not found')
    if doc['status'] in ('queued', 'processing'):
        raise HTTPException(status_code=409, detail='Wait until processing finishes')

    # Remove chunk vectors from ChromaDB
    chunk_ids = [c['id'] for c in doc.get('chunks', [])]
    if chunk_ids:
        storage.collection().delete(ids=chunk_ids)

    # Delete JSON metadata file
    storage._path(doc_id).unlink(missing_ok=True)
    return {'deleted': True}


# Pydantic model for incoming questions
class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    document_ids: list[str] = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)


# Backward-compatible alias for QuestionRequest
Question = QuestionRequest


@app.post('/api/ask')
def ask_question(body: QuestionRequest):
    """
    Handles user questions using the RAG pipeline:
    1. Validates selected documents.
    2. Encodes the question into a 384-dimensional query vector.
    3. Fetches stored embeddings from ChromaDB for selected documents.
    4. Computes exact Cosine Similarity scores using NumPy.
    5. Ranks chunks by score and selects Top-K context chunks.
    6. Calls Ollama LLM to synthesize a grounded answer.
    7. Returns the answer, sources, all candidate rankings, and step-by-step X-Ray explanations.
    """
    # 1. Validate that all selected documents are ready
    documents = [storage.get(doc_id) for doc_id in body.document_ids]
    for doc in documents:
        if not doc or doc.get('status') != 'ready':
            raise HTTPException(status_code=400, detail='Choose ready documents only')

    selected_doc_ids = {d['id'] for d in documents}

    # Collect all chunks belonging to selected documents
    candidate_chunks = [c for d in documents for c in d.get('chunks', [])]
    if not candidate_chunks:
        raise HTTPException(status_code=400, detail='No indexed chunks available')

    # 2. Fetch actual stored embeddings from ChromaDB
    chroma_col = storage.collection()
    candidate_ids = [c['id'] for c in candidate_chunks]
    chroma_data = chroma_col.get(
        ids=candidate_ids,
        include=['embeddings', 'documents', 'metadatas']
    )

    if chroma_data['embeddings'] is None or len(chroma_data['ids']) == 0:
        raise HTTPException(status_code=400, detail='Vector index is empty')

    # 3. Compute question embedding vector
    question_vector = engine.embed([body.question])[0]
    stored_matrix = np.asarray(chroma_data['embeddings'], dtype=np.float32)

    if stored_matrix.ndim != 2 or stored_matrix.shape[1] != len(question_vector):
        raise HTTPException(
            status_code=409,
            detail='Embedding model dimension mismatch; reindex documents'
        )

    # 4. Pure math: Compute cosine similarities between question vector and chunk vectors
    cosine_scores = engine.cosines(question_vector, stored_matrix)

    # 5. Build candidate ranking list
    ranked_chunks = []
    for i, chunk_id in enumerate(chroma_data['ids']):
        metadata = chroma_data['metadatas'][i] or {}
        if metadata.get('doc_id') not in selected_doc_ids:
            continue

        score = round(float(cosine_scores[i]), 6)
        chunk_vector = stored_matrix[i]
        ranked_chunks.append({
            'id': chunk_id,
            'filename': metadata.get('filename', ''),
            'page': metadata.get('page', 1),
            'text': chroma_data['documents'][i],
            'score': score,
            'vector_preview': engine.preview_vector(chunk_vector),
            'vector_norm': round(float(np.linalg.norm(chunk_vector)), 6)
        })

    # Sort chunks in descending order of similarity score (highest score first)
    ranked_chunks.sort(key=lambda c: c['score'], reverse=True)

    # 6. Select Top-K chunks as context for the LLM
    top_contexts = ranked_chunks[:body.top_k]

    # 7. Generate grounded answer using Ollama LLM
    answer_text, answer_mode = engine.ollama_answer(body.question, top_contexts)

    # 8. Record the 5 Question Journey X-Ray steps for UI inspection
    query_preview = [round(float(x), 6) for x in question_vector]
    top_scores_preview = [
        {'chunk_id': c['id'], 'score': c['score']}
        for c in ranked_chunks[:30]
    ]

    question_steps = [
        {
            'name': 'Your question',
            'before': {'question': body.question},
            'operation': 'Capture user question string',
            'after': {'query_text': body.question},
            'next': 'Encode the question into a vector'
        },
        {
            'name': 'Query embedding',
            'before': {'query_text': body.question},
            'operation': 'query_embedding = model.encode([query_text], normalize_embeddings=True)[0]',
            'after': {
                'model': engine.MODEL_NAME,
                'shape': [len(question_vector)],
                'values': query_preview,
                'norm': round(float(np.linalg.norm(question_vector)), 6)
            },
            'next': 'Compare query vector with saved document vectors'
        },
        {
            'name': 'Compare vectors',
            'before': {
                'query_dimension': len(question_vector),
                'stored_chunks': len(ranked_chunks)
            },
            'operation': 'similarity_scores = cosine_similarity(query_embedding, document_embeddings)',
            'after': {
                'scores': top_scores_preview,
                'note': 'Scores are actual cosine similarities of stored embeddings'
            },
            'next': 'Sort candidates by score'
        },
        {
            'name': 'Rank & choose',
            'before': {
                'candidate_count': len(ranked_chunks),
                'top_k': body.top_k
            },
            'operation': 'ranked_chunks = sorted(candidates, key=score, reverse=True)[:top_k]',
            'after': {
                'selected': top_contexts,
                'excluded_preview': ranked_chunks[body.top_k:body.top_k + 5]
            },
            'next': 'Build LLM context from the selected chunks'
        },
        {
            'name': 'Generate answer',
            'before': {
                'question': body.question,
                'context_chunks': [
                    {'citation': idx + 1, 'text': c['text']}
                    for idx, c in enumerate(top_contexts)
                ]
            },
            'operation': 'Generate grounded answer with Ollama; use clearly-labeled extractive fallback if unavailable',
            'after': {
                'answer': answer_text,
                'mode': answer_mode
            },
            'next': 'Click a source citation to inspect its exact chunk text'
        }
    ]

    return {
        'answer': answer_text,
        'answer_mode': answer_mode,
        'sources': top_contexts,
        'ranked': ranked_chunks[:50],
        'steps': question_steps,
        'top_k': body.top_k,
        'total_candidates': len(ranked_chunks),
        'model': engine.MODEL_NAME
    }
