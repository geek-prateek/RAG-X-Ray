"""
Main FastAPI Server for RAG X-Ray (Local Session Scoped)
========================================================
This file provides the HTTP API endpoints for:
1. Document management (upload, process, list, get, delete) scoped by local browser session.
2. Question answering (search relevant chunks with vector cosine similarity and generate LLM answer).
3. Local Query History (retrieve past queries and clear history per session).

Key Design Feature:
- Session Isolation (X-Session-ID): Documents and queries are stored locally per browser session.
- No global mixing: Another user or browser tab with a different session ID will not see your documents.
- Every processing step (stages/steps) records its 'before', 'operation', and 'after' state.
"""
from __future__ import annotations

import datetime
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Header, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

from . import engine, storage

# Initialize FastAPI application
app = FastAPI(
    title='RAG X-Ray',
    version='1.0.0',
    description='Beginner-friendly, inspectable RAG pipeline with local session isolation'
)

# Configure CORS so our Angular frontend (localhost, Vercel, and custom domains) can communicate with backend
allowed_origins_env = os.getenv('CORS_ORIGINS', 'http://localhost:4200,http://127.0.0.1:4200,http://localhost:8000,http://127.0.0.1:8000')
allowed_origins = [origin.strip() for origin in allowed_origins_env.split(',') if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r'https://.*\.vercel\.app',
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Background workers for document processing
workers = ThreadPoolExecutor(max_workers=2)

# File upload limit (configurable via MAX_FILE_SIZE_MB, default 50 MB)
MAX_FILE_SIZE_MB = int(os.getenv('MAX_FILE_SIZE_MB', '50'))
MAX_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


def record_stage(doc: dict, name: str, before: Any, operation: str, after: Any, next_step: str, session_id: str = 'default'):
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
    storage.put(doc, session_id=session_id)


def process_document_background(doc_id: str, content: bytes, session_id: str = 'default'):
    """
    Runs in the background to process an uploaded document through 6 clear stages:
    1. Upload & Hash check (SHA-256)
    2. Extract raw text page-by-page
    3. Clean and normalize whitespace
    4. Split into overlapping word chunks (size=90, overlap=18)
    5. Convert chunks into 384-dimensional vector embeddings
    6. Store chunks and vectors into ChromaDB (tagged with session_id)
    """
    doc = storage.get(doc_id, session_id=session_id)
    if not doc:
        return

    try:
        doc['status'] = 'processing'
        storage.put(doc, session_id=session_id)

        # ---------------------------------------------------------
        # STAGE 1: Upload & Validate
        # ---------------------------------------------------------
        file_sha256 = engine.sha256(content)
        record_stage(
            doc=doc,
            name='Upload',
            before={'uploaded_filename': doc['filename'], 'file_size_bytes': len(content)},
            operation='Validate file format and size; calculate SHA-256 fingerprint',
            after={'file_hash_sha256': file_sha256, 'is_valid_format': True, 'is_ready_for_extraction': True},
            next_step='Extract text page by page from the document',
            session_id=session_id
        )

        # ---------------------------------------------------------
        # STAGE 2: Extract Text Page-by-Page
        # ---------------------------------------------------------
        pages = engine.extract(doc['filename'], content)
        doc['pages'] = [
            {'page': p['page'], 'text': p['text'], 'characters': len(p['text'])}
            for p in pages
        ]
        pages_preview = [
            {
                'page_number': p['page'],
                'character_count': len(p['text']),
                'extracted_text_preview': engine.excerpt(p['text'], 500)
            }
            for p in pages
        ]
        record_stage(
            doc=doc,
            name='Extract text',
            before={'source_filename': doc['filename'], 'total_pages_in_document': len(pages)},
            operation='Read text page by page using PyMuPDF',
            after={'extracted_pages_count': len(pages), 'extracted_pages_sample': pages_preview},
            next_step='Clean and normalize whitespace across extracted pages',
            session_id=session_id
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
            before={'raw_text_page_1': engine.excerpt(pages[0]['text']), 'raw_characters_page_1': len(pages[0]['text'])},
            operation='Normalize excessive whitespace and line breaks while preserving document words',
            after={
                'cleaned_text_page_1': engine.excerpt(cleaned_pages[0]['text']),
                'total_cleaned_words_in_document': total_cleaned_words
            },
            next_step='Split cleaned document text into overlapping chunks',
            session_id=session_id
        )

        # ---------------------------------------------------------
        # STAGE 4: Make Chunks (Sliding Window)
        # ---------------------------------------------------------
        chunk_size = 90
        chunk_overlap = 18
        chunks = engine.chunk_pages(cleaned_pages, size=chunk_size, overlap=chunk_overlap)
        if not chunks:
            raise ValueError('The document contains no usable text after cleaning.')

        chunks_formatted_sample = [
            {
                'chunk_index': i + 1,
                'source_page': c['page'],
                'chunk_word_count': len(c['text'].split()),
                'chunk_text': c['text']
            }
            for i, c in enumerate(chunks[:5])
        ]

        record_stage(
            doc=doc,
            name='Make chunks',
            before={
                'cleaned_document_pages': len(cleaned_pages),
                'chunk_size_words': chunk_size,
                'chunk_overlap_words': chunk_overlap
            },
            operation='Split each document page into overlapping word-based chunks for vector search',
            after={
                'total_document_chunks_created': len(chunks),
                'document_chunks_sample': chunks_formatted_sample
            },
            next_step='Convert each document chunk into a 384-dimensional embedding vector',
            session_id=session_id
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
                'vector_384_dimensions_sample': [round(float(val), 6) for val in vec[:8]]
            }
            for i, vec in enumerate(vectors[:3])
        ]

        record_stage(
            doc=doc,
            name='Create embeddings',
            before={
                'document_chunks_to_embed': len(chunks),
                'first_document_chunk_sample': engine.excerpt(chunks[0]['text'])
            },
            operation='Convert text chunks into 384-dimensional dense vectors using SentenceTransformer (all-MiniLM-L6-v2)',
            after={
                'embedding_model_used': engine.MODEL_NAME,
                'total_chunk_vectors_generated': int(vectors.shape[0]),
                'vector_dimensions': int(vectors.shape[1]),
                'shape': list(vectors.shape),
                'first_chunk_vector_length_norm': first_vector_norm,
                'document_chunk_vectors_sample': first_vectors_preview
            },
            next_step='Save vectors and chunk references into ChromaDB vector store',
            session_id=session_id
        )

        # ---------------------------------------------------------
        # STAGE 6: Save to ChromaDB Vector Store (Scoped by session_id)
        # ---------------------------------------------------------
        chunk_ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                'doc_id': doc_id,
                'session_id': session_id,
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
            before={
                'document_chunk_vectors_to_store': len(vectors),
                'vector_dimensions': int(vectors.shape[1]),
                'target_vector_store': 'ChromaDB'
            },
            operation='Upsert document chunks, embeddings, and metadata into persistent ChromaDB collection',
            after={
                'stored_document_chunks_in_db': len(chunk_ids),
                'db_collection_name': 'rag_xray_chunks',
                'stored_chunk_ids_sample': chunk_ids[:5]
            },
            next_step='Document is fully indexed! Ask any question to trace the retrieval pipeline',
            session_id=session_id
        )

        doc['status'] = 'ready'

    except Exception as exc:
        doc['status'] = 'failed'
        doc['error'] = str(exc)

    finally:
        storage.put(doc, session_id=session_id)


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
def list_documents(x_session_id: str = Header(default='default', alias='X-Session-ID')):
    """
    Returns a lightweight summary of all uploaded documents for this session:
    their ID, filename, status, error (if any), stage count, and chunk count.
    """
    all_stored = storage.all_docs(session_id=x_session_id)
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
def get_document(doc_id: str, x_session_id: str = Header(default='default', alias='X-Session-ID')):
    """
    Returns full details for a single document, including all stages and chunks,
    scoped to this session.
    """
    doc = storage.get(doc_id, session_id=x_session_id)
    if doc is None:
        raise HTTPException(status_code=404, detail='Document not found')
    return doc


@app.post('/api/documents', status_code=202)
async def upload_document(
    file: UploadFile = File(...),
    x_session_id: str = Header(default='default', alias='X-Session-ID')
):
    """
    Uploads a document (.pdf, .txt, .md), creates a queued record in the local session,
    and starts processing in the background thread.
    """
    file_extension = Path(file.filename or '').suffix.lower()
    if file_extension not in {'.pdf', '.txt', '.md'}:
        raise HTTPException(status_code=400, detail='Supported formats: PDF, TXT, MD')

    content = await file.read(MAX_BYTES + 1)
    if not content or len(content) > MAX_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f'File must be nonempty and no more than {MAX_FILE_SIZE_MB} MB'
        )

    safe_filename = Path(file.filename or 'upload').name
    doc = storage.new_doc(safe_filename, session_id=x_session_id)
    # Save the raw file so users can view and read the original file in the browser
    storage.save_raw_file(doc['id'], safe_filename, content, session_id=x_session_id)

    # Hand off processing to background thread with session_id
    workers.submit(process_document_background, doc['id'], content, x_session_id)

    return {'document_id': doc['id'], 'status': 'queued'}


@app.post('/api/documents/sample', status_code=202)
def load_sample_document(x_session_id: str = Header(default='default', alias='X-Session-ID')):
    """
    Loads the built-in sample_docs/employee_handbook.txt with 1 click.
    Provides instant hands-on demonstration without requiring manual file uploads.
    """
    sample_path = Path(__file__).resolve().parent.parent.parent / 'sample_docs' / 'employee_handbook.txt'
    if not sample_path.exists():
        sample_path = Path('sample_docs/employee_handbook.txt')
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail='Sample document file not found on server')

    content = sample_path.read_bytes()
    filename = 'employee_handbook.txt'
    doc = storage.new_doc(filename, session_id=x_session_id)
    storage.save_raw_file(doc['id'], filename, content, session_id=x_session_id)
    workers.submit(process_document_background, doc['id'], content, x_session_id)
    return {'document_id': doc['id'], 'filename': filename, 'status': 'queued'}


@app.get('/api/documents/{doc_id}/content')
def get_document_content(doc_id: str, x_session_id: str = Header(default='default', alias='X-Session-ID')):
    """
    Returns full readable content for the document page-by-page so the user can read it in the browser.
    Works for newly uploaded files as well as previously uploaded documents.
    """
    doc = storage.get(doc_id, session_id=x_session_id)
    if doc is None:
        raise HTTPException(status_code=404, detail='Document not found')

    pages = doc.get('pages', [])
    if not pages:
        # Check if saved in raw file
        raw_file = storage.get_raw_file(doc_id, session_id=x_session_id)
        if raw_file and raw_file.exists():
            try:
                extracted = engine.extract(doc['filename'], raw_file.read_bytes())
                pages = [{'page': p['page'], 'text': p['text'], 'characters': len(p['text'])} for p in extracted]
                doc['pages'] = pages
                storage.put(doc, session_id=x_session_id)
            except Exception:
                pass

    if not pages:
        # Fallback 1: Reconstruct text from stages if available
        stages = doc.get('stages', [])
        for st in stages:
            if st.get('name') == 'Extract text' and 'pages' in st.get('after', {}):
                pages = [
                    {'page': p['page'], 'text': p.get('text', ''), 'characters': p.get('characters', len(p.get('text', '')))}
                    for p in st['after']['pages']
                ]
                break

    if not pages:
        # Fallback 2: Assemble from chunks
        chunks = doc.get('chunks', [])
        page_dict: dict[int, list[str]] = {}
        for c in chunks:
            p_num = c.get('page', 1)
            page_dict.setdefault(p_num, []).append(c.get('text', ''))
        pages = [
            {'page': p_num, 'text': '\n\n'.join(texts), 'characters': sum(len(t) for t in texts)}
            for p_num, texts in sorted(page_dict.items())
        ]

    full_text = '\n\n--- Page Break ---\n\n'.join(p['text'] for p in pages) if pages else ''
    total_words = sum(len(p['text'].split()) for p in pages) if pages else 0

    return {
        'id': doc_id,
        'filename': doc.get('filename', 'Document'),
        'status': doc.get('status', 'unknown'),
        'total_pages': len(pages),
        'total_words': total_words,
        'pages': pages,
        'full_text': full_text
    }


@app.get('/api/documents/{doc_id}/raw')
def get_document_raw(doc_id: str, x_session_id: str = Header(default='default', alias='X-Session-ID')):
    """
    Streams the raw document file directly to the browser (inline) so users can view the PDF or text.
    """
    doc = storage.get(doc_id, session_id=x_session_id)
    if doc is None:
        raise HTTPException(status_code=404, detail='Document not found')

    raw_file = storage.get_raw_file(doc_id, session_id=x_session_id)
    if raw_file and raw_file.exists():
        media_type = 'application/pdf' if doc['filename'].lower().endswith('.pdf') else 'text/plain; charset=utf-8'
        return FileResponse(
            path=str(raw_file),
            media_type=media_type,
            filename=doc['filename'],
            headers={'Content-Disposition': f'inline; filename="{doc["filename"]}"'}
        )

    # If raw file is not present on disk, stream generated text
    content_data = get_document_content(doc_id, x_session_id)
    return PlainTextResponse(
        content=content_data['full_text'],
        headers={'Content-Disposition': f'inline; filename="{doc["filename"]}.txt"'}
    )


@app.delete('/api/documents/{doc_id}')
def delete_document(doc_id: str, x_session_id: str = Header(default='default', alias='X-Session-ID')):
    """
    Deletes a document from the local session and removes all its chunks from ChromaDB.
    """
    doc = storage.get(doc_id, session_id=x_session_id)
    if not doc:
        raise HTTPException(status_code=404, detail='Document not found')
    if doc['status'] in ('queued', 'processing'):
        raise HTTPException(status_code=409, detail='Wait until processing finishes')

    # Remove chunk vectors from ChromaDB
    chunk_ids = [c['id'] for c in doc.get('chunks', [])]
    if chunk_ids:
        storage.collection().delete(ids=chunk_ids)

    # Delete raw document files if present
    storage.delete_raw_file(doc_id, session_id=x_session_id)

    # Delete JSON metadata file
    storage._path(doc_id, session_id=x_session_id).unlink(missing_ok=True)
    # Also unlink legacy path if present
    (storage.DATA / f'{doc_id}.json').unlink(missing_ok=True)
    return {'deleted': True}


# Pydantic model for incoming questions
class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    document_ids: list[str] = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)
    provider: str = Field(default='ollama')  # 'ollama', 'groq', 'openai', 'gemini'
    model: str | None = None
    api_key: str | None = None


# Backward-compatible alias for QuestionRequest
Question = QuestionRequest


@app.post('/api/ask')
def ask_question(
    body: QuestionRequest,
    x_session_id: str = Header(default='default', alias='X-Session-ID')
):
    """
    Handles user questions using the RAG pipeline scoped to the local session:
    1. Validates selected documents within this session.
    2. Encodes the question into a 384-dimensional query vector.
    3. Fetches stored embeddings from ChromaDB.
    4. Computes exact Cosine Similarity scores using NumPy.
    5. Ranks chunks by score and selects Top-K context chunks.
    6. Calls Ollama LLM to synthesize a grounded answer.
    7. Automatically saves query & answer into local session history.
    8. Returns the answer, sources, all candidate rankings, and step-by-step X-Ray explanations.
    """
    # 1. Validate that all selected documents are ready
    documents = [storage.get(doc_id, session_id=x_session_id) for doc_id in body.document_ids]
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

    # 7. Generate grounded answer using chosen LLM provider (Ollama, Groq, OpenAI, Gemini)
    if body.provider == 'ollama' and not body.model and not body.api_key:
        answer_text, answer_mode = engine.ollama_answer(body.question, top_contexts)
        actual_model_name = answer_mode.replace('ollama:', '') if answer_mode.startswith('ollama:') else os.getenv('OLLAMA_MODEL', 'qwen2.5-coder:7b')
    else:
        answer_text, actual_model_name, answer_mode = engine.generate_llm_answer(
            question=body.question,
            contexts=top_contexts,
            provider=body.provider,
            model=body.model,
            api_key=body.api_key
        )

    # 8. Record the 5 Question Journey X-Ray steps with REAL dynamic runtime values
    query_preview = [round(float(x), 6) for x in question_vector]
    top_scores_preview = [
        {'chunk_id': c['id'], 'filename': c['filename'], 'page': c['page'], 'score': c['score']}
        for c in ranked_chunks[:15]
    ]

    word_count_q = len(body.question.split())
    char_count_q = len(body.question)
    highest_score = top_contexts[0]['score'] if top_contexts else 0.0
    lowest_score = ranked_chunks[-1]['score'] if ranked_chunks else 0.0
    total_candidates_count = len(ranked_chunks)
    excluded_count = max(0, total_candidates_count - len(top_contexts))

    selected_preview = ", ".join([
        f"{c['filename']} (p.{c['page']}, score {c['score']:.3f})"
        for c in top_contexts
    ])
    vector_preview_str = ", ".join(f"{x:.4f}" for x in query_preview[:4])

    provider_label = {
        'groq': 'Groq Cloud',
        'gemini': 'Google Gemini',
        'openai': 'OpenAI',
        'ollama': 'Local Ollama'
    }.get(body.provider.lower(), body.provider)

    question_steps = [
        {
            'name': 'Your question',
            'summary': f'User Query: "{body.question}" ({word_count_q} words, {char_count_q} chars across {len(selected_doc_ids)} selected document(s) with {total_candidates_count} indexed document chunks in scope)',
            'before': {
                'user_question': body.question,
                'question_character_count': char_count_q,
                'question_word_count': word_count_q,
                'selected_document_files': [d['filename'] for d in documents]
            },
            'operation': f'Captured and tokenized query string ({word_count_q} words across {len(selected_doc_ids)} document(s) with {total_candidates_count} indexed document chunks in scope)',
            'after': {
                'user_question': body.question,
                'selected_documents_count': len(selected_doc_ids),
                'total_document_chunks_in_scope': total_candidates_count,
                'top_k_chunks_to_retrieve': body.top_k
            },
            'next': f'Encode user question into a {len(question_vector)}-dimensional dense query vector using {engine.MODEL_NAME}'
        },
        {
            'name': 'Query embedding',
            'summary': f'Dense query vector generated [{vector_preview_str}...] | {len(question_vector)} dimensions',
            'before': {
                'user_question_text': body.question,
                'embedding_model': engine.MODEL_NAME
            },
            'operation': f'SentenceTransformer("{engine.MODEL_NAME}").encode([user_question], normalize_embeddings=True)',
            'after': {
                'embedding_model': engine.MODEL_NAME,
                'query_vector_dimensions': len(question_vector),
                'query_vector_preview': query_preview[:8],
                'values': query_preview[:12],
                'query_vector_norm': round(float(np.linalg.norm(question_vector)), 6)
            },
            'next': f'Calculate cosine similarity between query vector and all {total_candidates_count} stored document chunk vectors'
        },
        {
            'name': 'Compare vectors',
            'summary': f'Calculated cosine similarity against {total_candidates_count} stored document chunks (Top match: {highest_score:.3f}, Lowest: {lowest_score:.3f})',
            'before': {
                'query_vector_dimensions': len(question_vector),
                'stored_document_chunks_to_compare': total_candidates_count
            },
            'operation': f'Calculated cosine similarity dot products between query vector and {total_candidates_count} document chunk vectors in ChromaDB (Highest similarity: {highest_score:.4f})',
            'after': {
                'total_document_chunks_scored': total_candidates_count,
                'top_similarity_score': highest_score,
                'lowest_similarity_score': lowest_score,
                'top_matching_document_chunks': top_scores_preview
            },
            'next': f'Rank all {total_candidates_count} document chunks descending by cosine similarity and select Top-{body.top_k} highest-scoring chunks'
        },
        {
            'name': 'Rank & choose',
            'summary': f'Selected Top-{body.top_k} document chunks: {selected_preview}',
            'before': {
                'total_document_chunks_evaluated': total_candidates_count,
                'top_k_chunks_to_retrieve': body.top_k
            },
            'operation': f'Ranked {total_candidates_count} document chunks descending by similarity score; retained top {len(top_contexts)} best-matching chunks; filtered out {excluded_count} lower-scoring chunks',
            'after': {
                'selected_top_document_chunks': [
                    {
                        'rank': i + 1,
                        'document_file': c['filename'],
                        'page_number': c['page'],
                        'similarity_score': round(float(c['score']), 4),
                        'chunk_text_preview': c['text'][:160] + ('...' if len(c['text']) > 160 else '')
                    }
                    for i, c in enumerate(top_contexts)
                ],
                'excluded_lower_score_chunks_count': excluded_count,
                'excluded_chunks_sample': [
                    {
                        'document_file': c['filename'],
                        'page_number': c['page'],
                        'similarity_score': round(float(c['score']), 4),
                        'chunk_text_preview': c['text'][:120] + ('...' if len(c['text']) > 120 else '')
                    }
                    for c in ranked_chunks[body.top_k:body.top_k + 4]
                ]
            },
            'next': f'Pass selected {len(top_contexts)} document chunks as grounded context to {actual_model_name} ({provider_label})'
        },
        {
            'name': 'Generate answer',
            'summary': f'Generated {len(answer_text.split())} words using {actual_model_name} ({provider_label}) citing {len(top_contexts)} document chunks',
            'before': {
                'user_question': body.question,
                'grounded_context_chunks_count': len(top_contexts),
                'total_words_in_context_chunks': sum(len(c['text'].split()) for c in top_contexts),
                'llm_model': actual_model_name,
                'llm_provider': provider_label
            },
            'operation': f'Grounded LLM synthesis using model "{actual_model_name}" via {provider_label} (mode: {answer_mode}) with {len(top_contexts)} cited document chunks',
            'after': {
                'generated_answer': answer_text,
                'generated_answer_words': len(answer_text.split()),
                'cited_document_chunks_count': len(top_contexts),
                'llm_model': actual_model_name,
                'llm_provider': provider_label,
                'synthesis_mode': answer_mode
            },
            'next': 'Inspect source citations and cosine similarity scores below'
        }
    ]

    result_payload = {
        'id': 'query_' + uuid.uuid4().hex[:12],
        'question': body.question,
        'answer': answer_text,
        'answer_mode': answer_mode,
        'sources': top_contexts,
        'ranked': ranked_chunks[:50],
        'steps': question_steps,
        'top_k': body.top_k,
        'total_candidates': len(ranked_chunks),
        'model': actual_model_name,
        'provider': body.provider,
        'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat()
    }

    # Automatically persist this query & answer in local session history
    storage.add_history(x_session_id, result_payload)

    return result_payload


class ModelTestRequest(BaseModel):
    provider: str
    model: str | None = None
    api_key: str | None = None


@app.post('/api/models/test')
def test_model_connection(body: ModelTestRequest):
    """
    Tests whether the selected provider (Ollama, Groq, Gemini, OpenAI)
    is reachable and the API key / model is valid.
    """
    try:
        prov = (body.provider or 'ollama').lower().strip()
        system_prompt = "You are an AI connection tester. Answer in 1 word."
        user_prompt = "Say 'Ready' in one word."
        if prov == 'groq':
            reply, active_model = engine.call_groq(system_prompt, user_prompt, body.model, body.api_key)
        elif prov == 'openai':
            reply, active_model = engine.call_openai(system_prompt, user_prompt, body.model, body.api_key)
        elif prov == 'gemini':
            reply, active_model = engine.call_gemini(system_prompt, user_prompt, body.model, body.api_key)
        elif prov == 'ollama':
            reply, active_model = engine.call_ollama(system_prompt, user_prompt, body.model)
        else:
            raise ValueError(f"Unknown provider '{prov}'")
        return {
            'status': 'ok',
            'provider': prov,
            'model': active_model,
            'message': f"Connected successfully to {active_model}!"
        }
    except Exception as exc:
        return {
            'status': 'error',
            'provider': body.provider,
            'message': str(exc)
        }


# -------------------------------------------------------------
# Local Query History Endpoints (Session Isolated)
# -------------------------------------------------------------

@app.get('/api/history')
def get_session_history(x_session_id: str = Header(default='default', alias='X-Session-ID')):
    """
    Returns all past queries, answers, and X-Ray journeys for this local session.
    """
    return storage.get_history(session_id=x_session_id)


@app.delete('/api/history')
def clear_session_history(x_session_id: str = Header(default='default', alias='X-Session-ID')):
    """
    Clears all saved query history for this local session.
    """
    storage.clear_history(session_id=x_session_id)
    return {'cleared': True}


# -------------------------------------------------------------
# Local Workspaces Endpoints (List, Switch, Create, Delete)
# -------------------------------------------------------------

@app.get('/api/workspaces')
def get_workspaces():
    """
    Returns a list of all existing local workspaces sorted by latest activity,
    with document count, query count, and latest flag.
    """
    return storage.all_workspaces()


class WorkspaceCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    id: str | None = None


@app.post('/api/workspaces')
def create_workspace(body: WorkspaceCreate):
    """
    Creates or registers a workspace with a readable name.
    """
    session_id = body.id or ('sess_' + uuid.uuid4().hex[:12])
    storage.set_workspace_name(session_id, body.name)
    return {'id': session_id, 'name': body.name}


@app.delete('/api/workspaces/{session_id}')
def delete_workspace_endpoint(session_id: str):
    """
    Deletes a specific workspace and cleans up its stored documents and vectors.
    """
    success = storage.delete_workspace(session_id)
    if not success:
        raise HTTPException(status_code=400, detail='Cannot delete default or non-existent workspace')
    return {'deleted': True}


# -------------------------------------------------------------
# SEO & Web Standards Endpoints (/robots.txt, /sitemap.xml)
# -------------------------------------------------------------

@app.get('/robots.txt', response_class=PlainTextResponse)
def robots_txt():
    """
    Standard robots.txt telling search engine crawlers what to index.
    """
    return "User-agent: *\nAllow: /\n\nSitemap: http://localhost:4200/sitemap.xml\n"


@app.get('/sitemap.xml')
def sitemap_xml():
    """
    XML Sitemap pointing search engines to the main application URLs.
    """
    content = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>http://localhost:4200/</loc>
    <lastmod>2026-11-05</lastmod>
    <changefreq>weekly</changefreq>
    <priority>1.0</priority>
  </url>
</urlset>"""
    return Response(content=content, media_type="application/xml")
