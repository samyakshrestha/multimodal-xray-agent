import numpy as np

def query_text_faiss(query, model, index, metadata, top_k=5):
    """Query FAISS index for most similar PubMed abstracts."""
    # Encode the input query into a vector using the provided model
    query_vec = model.encode([query], normalize_embeddings=True)
    # Search the FAISS index for the top_k most similar vectors to the query
    scores, indices = index.search(np.array(query_vec, dtype=np.float32), top_k)

    results = []
    # Iterate over the top_k results and their similarity scores
    for idx, score in zip(indices[0], scores[0]):
        result = metadata[idx]  # Retrieve metadata for the matched document
        results.append({
            "pmid": result["pmid"],   # PubMed ID of the document
            "title": result["title"], # Title of the document
            "score": float(score)     # Similarity score
        })
    return results