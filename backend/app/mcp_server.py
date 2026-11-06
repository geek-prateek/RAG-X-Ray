"""
RAG X-Ray Inspector & Evaluator MCP Server
==========================================
Model Context Protocol (MCP) server providing deep diagnostic inspection,
semantic evaluation, vector math analysis, and grounded knowledge retrieval
for AI coding agents (Cursor IDE, Claude Desktop, Antigravity, Windsurf).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from mcp.server.mcpserver import MCPServer

from app import engine, storage

mcp = MCPServer("rag-xray-evaluator")


@mcp.tool()
def evaluate_retrieval(query: str, top_k: int = 5, session_id: str = "default") -> str:
    """
    Evaluates semantic retrieval for a query against indexed documents.
    Returns cosine similarity scores, confidence verdict, score dropoff,
    and ranked citations so AI agents can detect hallucination risks.
    """
    try:
        col = storage.collection()
        if col.count() == 0:
            return json.dumps({
                "status": "empty",
                "message": "Knowledge base is empty. Index documents first using index_document_text.",
                "total_candidates": 0
            }, indent=2)

        data = col.get(include=["embeddings", "documents", "metadatas"])
        if not data or not data.get("ids") or data.get("embeddings") is None:
            return json.dumps({
                "status": "empty",
                "message": "No indexed embeddings found in vector store.",
                "total_candidates": 0
            }, indent=2)

        # 1. Compute query vector
        query_vector = engine.embed([query])[0]
        stored_matrix = np.asarray(data["embeddings"], dtype=np.float32)

        # 2. Compute cosine similarity scores
        scores = engine.cosines(query_vector, stored_matrix)

        # 3. Rank candidates
        ranked_indices = np.argsort(-scores)
        top_k = max(1, min(top_k, 20))
        top_indices = ranked_indices[:top_k]

        highest_score = round(float(scores[top_indices[0]]), 4)
        lowest_top_score = round(float(scores[top_indices[-1]]), 4)
        dropoff = round(highest_score - float(scores[top_indices[1]]), 4) if len(top_indices) > 1 else 0.0

        # 4. Formulate diagnostic verdict
        if highest_score >= 0.70:
            verdict = "EXCELLENT_GROUNDING"
            guidance = "High semantic alignment. The retrieved context directly covers the query. Safe to synthesize answers."
        elif highest_score >= 0.45:
            verdict = "MODERATE_RELEVANCE"
            guidance = "Partial semantic alignment. Verify evidence carefully before making definitive claims."
        else:
            verdict = "POOR_MATCH_HIGH_HALLUCINATION_RISK"
            guidance = "Warning: Cosine similarity is below 0.45 threshold. The knowledge base likely does not contain this information. Do not hallucinate."

        ranked_chunks = []
        for rank, idx in enumerate(top_indices, 1):
            meta = data["metadatas"][idx] if data.get("metadatas") else {}
            text = data["documents"][idx] if data.get("documents") else ""
            ranked_chunks.append({
                "rank": rank,
                "score": round(float(scores[idx]), 4),
                "filename": meta.get("filename", "unknown"),
                "page": meta.get("page", 1),
                "chunk_id": data["ids"][idx],
                "content_preview": engine.excerpt(text, 200),
                "full_text": text
            })

        report = {
            "query": query,
            "retrieval_health": {
                "verdict": verdict,
                "confidence_score": highest_score,
                "score_dropoff_rank1_to_rank2": dropoff,
                "lowest_topk_score": lowest_top_score,
                "ai_synthesis_guidance": guidance
            },
            "total_candidates_evaluated": len(scores),
            "top_ranked_chunks": ranked_chunks
        }
        return json.dumps(report, indent=2)

    except Exception as exc:
        return json.dumps({"status": "error", "error": str(exc)}, indent=2)


@mcp.tool()
def compare_semantic_similarity(text_a: str, text_b: str) -> str:
    """
    Computes exact Cosine Similarity between two arbitrary pieces of text.
    Helps developers test why a chunk matched or missed a query.
    """
    try:
        vectors = engine.embed([text_a, text_b])
        vec_a = vectors[0]
        vec_b = vectors[1]

        score = float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 1e-9))
        score = max(-1.0, min(1.0, score))
        score_round = round(score, 4)

        if score >= 0.85:
            verdict = "Near-Identical Semantic Intent"
        elif score >= 0.65:
            verdict = "High Semantic Overlap"
        elif score >= 0.40:
            verdict = "Moderate Thematic Relatedness"
        else:
            verdict = "Low / Distinct / Unrelated"

        result = {
            "cosine_similarity": score_round,
            "semantic_overlap_percentage": f"{max(0.0, score_round) * 100:.1f}%",
            "diagnostic_verdict": verdict,
            "embedding_model": engine.MODEL_NAME,
            "dimensions": int(vectors.shape[1]),
            "text_a_excerpt": engine.excerpt(text_a, 120),
            "text_b_excerpt": engine.excerpt(text_b, 120)
        }
        return json.dumps(result, indent=2)
    except Exception as exc:
        return json.dumps({"status": "error", "error": str(exc)}, indent=2)


@mcp.tool()
def inspect_vector_embedding(text: str) -> str:
    """
    Inspects dense embedding vector math (dimensions, preview components, L2 norm).
    Provides true X-Ray visibility into how the text is represented in vector space.
    """
    try:
        vec = engine.embed([text])[0]
        norm = round(float(np.linalg.norm(vec)), 6)
        preview = [round(float(v), 6) for v in vec[:8]]

        result = {
            "text_sample": engine.excerpt(text, 150),
            "word_count": len(text.split()),
            "character_count": len(text),
            "embedding_model": f"{engine.MODEL_NAME} (FastEmbed ONNX)",
            "vector_dimensions": len(vec),
            "vector_l2_norm": norm,
            "first_8_dimensions_preview": preview
        }
        return json.dumps(result, indent=2)
    except Exception as exc:
        return json.dumps({"status": "error", "error": str(exc)}, indent=2)


@mcp.tool()
def list_indexed_documents(session_id: str = "default") -> str:
    """
    Lists all indexed documents currently stored in the RAG X-Ray knowledge base.
    """
    try:
        docs = storage.all_docs(session_id=session_id)
        summary = []
        for d in docs:
            stages = d.get("stages", [])
            summary.append({
                "document_id": d.get("id"),
                "filename": d.get("filename"),
                "status": d.get("status"),
                "total_pages": len(d.get("pages", [])),
                "total_chunks": len(d.get("chunks", [])),
                "stages_completed": f"{len(stages)}/6",
                "created_at": d.get("created_at")
            })
        return json.dumps({"total_documents": len(summary), "documents": summary}, indent=2)
    except Exception as exc:
        return json.dumps({"status": "error", "error": str(exc)}, indent=2)


@mcp.tool()
def diagnose_document_chunking(document_id: str, session_id: str = "default") -> str:
    """
    Diagnoses chunk boundary health, token distribution, and page coverage
    for a specific indexed document.
    """
    try:
        doc = storage.get(document_id, session_id=session_id)
        if not doc:
            return json.dumps({"status": "not_found", "error": f"Document '{document_id}' not found."}, indent=2)

        chunks = doc.get("chunks", [])
        if not chunks:
            return json.dumps({"status": "no_chunks", "message": "Document has no chunks indexed."}, indent=2)

        word_counts = [len(c.get("text", "").split()) for c in chunks]
        min_words = min(word_counts) if word_counts else 0
        max_words = max(word_counts) if word_counts else 0
        avg_words = round(sum(word_counts) / len(word_counts), 1) if word_counts else 0

        result = {
            "document_id": doc.get("id"),
            "filename": doc.get("filename"),
            "status": doc.get("status"),
            "stages_completed": f"{len(doc.get('stages', []))}/6",
            "chunk_diagnostics": {
                "total_chunks": len(chunks),
                "sliding_window_size": 90,
                "sliding_window_overlap": 18,
                "min_chunk_words": min_words,
                "max_chunk_words": max_words,
                "average_chunk_words": avg_words,
                "chunk_health": "OPTIMAL" if min_words >= 15 else "CONTAINS_SHORT_FRAGMENTS"
            },
            "first_3_chunks_sample": [
                {
                    "chunk_id": c.get("id"),
                    "page": c.get("page"),
                    "words": len(c.get("text", "").split()),
                    "preview": engine.excerpt(c.get("text", ""), 140)
                }
                for c in chunks[:3]
            ]
        }
        return json.dumps(result, indent=2)
    except Exception as exc:
        return json.dumps({"status": "error", "error": str(exc)}, indent=2)


@mcp.tool()
def index_document_text(filename: str, content: str, session_id: str = "default") -> str:
    """
    Directly indexes a text document, code snippet, or policy into the knowledge base.
    Immediately creates overlapping chunks, computes dense vectors, and upserts into ChromaDB.
    """
    try:
        raw_bytes = content.encode("utf-8")
        doc = storage.new_doc(filename, session_id=session_id)
        doc_id = doc["id"]
        storage.save_raw_file(doc_id, filename, raw_bytes, session_id=session_id)

        # 1. Clean
        cleaned_text = engine.clean(content)
        pages = [{"page": 1, "text": cleaned_text}]

        # 2. Chunk
        chunks = engine.chunk_pages(pages, size=90, overlap=18)
        if not chunks:
            return json.dumps({"status": "error", "error": "Content has no usable text after cleaning."}, indent=2)

        # 3. Embed
        chunk_texts = [c["text"] for c in chunks]
        vectors = engine.embed(chunk_texts)

        # 4. Upsert into ChromaDB
        chunk_ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "doc_id": doc_id,
                "session_id": session_id,
                "filename": filename,
                "page": 1,
                "start_word": c["start_word"]
            }
            for c in chunks
        ]

        col = storage.collection()
        col.upsert(
            ids=chunk_ids,
            embeddings=vectors.tolist(),
            documents=chunk_texts,
            metadatas=metadatas
        )

        doc["pages"] = [{"page": 1, "text": cleaned_text, "characters": len(cleaned_text)}]
        doc["chunks"] = [{**c, "id": chunk_ids[i]} for i, c in enumerate(chunks)]
        doc["status"] = "ready"
        storage.put(doc, session_id=session_id)

        return json.dumps({
            "status": "success",
            "message": f"Successfully indexed '{filename}' with {len(chunks)} chunks.",
            "document_id": doc_id,
            "total_chunks": len(chunks),
            "embedding_model": engine.MODEL_NAME
        }, indent=2)

    except Exception as exc:
        return json.dumps({"status": "error", "error": str(exc)}, indent=2)


if __name__ == "__main__":
    mcp.run()
