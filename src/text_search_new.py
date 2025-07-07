
import numpy as np

def query_text_faiss(query: str, index, metadata: list[dict], top_k=5) -> list[dict]:
    """
    Query FAISS index for most similar PubMed abstracts using SPECTER2.
    
    Parameters:
        query (str): The search string.
        index (faiss.Index): Prebuilt FAISS index.
        metadata (list of dict): List of dicts with metadata (pmid, title, etc.).
        top_k (int): Number of top results to return.

    Returns:
        List of dicts with pmid, title, and similarity score.
    """
    # Embed query and reshape for FAISS
    query_vec = embed_texts_specter2([query])

    # Perform FAISS search
    scores, indices = index.search(query_vec.astype(np.float32), top_k)

    # Collect top-k results
    results = []
    for idx, score in zip(indices[0], scores[0]):
        result = metadata[idx]
        results.append({
            "pmid": result["pmid"],
            "title": result["title"],
            "score": float(score)
        })

    return results