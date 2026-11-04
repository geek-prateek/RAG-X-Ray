# RAG X-Ray — User guide

1. Open http://localhost:4200, choose `sample_docs/employee_handbook.txt` (or your own PDF/TXT/MD).
2. Select the document card, click **Explore Document Journey**. Use each step's **BEFORE → OPERATION → AFTER → NEXT** to inspect transformations. Expand long values.
3. Wait until status **ready**; document checkboxes determine which knowledge sources are searched.
4. Ask **How do employees request annual leave?**. Answer displays the nearest evidence.
5. Click a **Source evidence** row to show the actual stored passage.
6. Click **Explore Question Journey**. Navigate to **Query embedding** to inspect the real `query_embedding` preview, then **Compare vectors** to see actual calculated cosine scores.
7. In **Rank & choose**, inspect why chunks ranked as shown; selected scores come from the stored 384D vectors.
8. Click **What if I change Top-K?**, adjust the slider, see the new source list, then **Re-run** to rebuild the response.

**No fake values:** If an embedding model or vector database fails, the upload becomes `failed`. If Ollama is absent, the answer is explicitly labeled as source excerpts.
