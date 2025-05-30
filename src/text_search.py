import numpy as np

def query_text_faiss(query, model, index, metadata, top_k=5):
    """Query FAISS index for most similar PubMed abstracts."""
    query_vec = model.encode([query], normalize_embeddings=True)
    scores, indices = index.search(np.array(query_vec, dtype=np.float32), top_k)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        result = metadata[idx]
        results.append({
            "pmid": result["pmid"],
            "title": result["title"],
            "score": float(score)
        })
    return results