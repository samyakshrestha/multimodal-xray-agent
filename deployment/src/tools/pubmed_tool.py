from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import os
import json
import torch
import faiss
import numpy as np
from pathlib import Path
from .specter2_embedder import embed_texts_specter2  # Import from same folder

# Input schema for the tool, expects a caption string
class PubmedQueryInput(BaseModel):
    caption: str

# Main tool class for PubMed retrieval
class PubmedRetrievalTool(BaseTool):
    # Tool name and description
    name: str = "pubmed_retrieval_tool"
    description: str = (
        "Retrieves the most relevant PubMed articles for a given radiology caption."
    )
    args_schema: type = PubmedQueryInput
    metadata: dict = {}
    
    def __init__(self, **data):
        # Initialize the base tool with provided data
        super().__init__(**data)
    
    def _run(self, caption: str = None, **kwargs) -> str:
        """
        Retrieves relevant PubMed articles based on a radiology caption.
        """
        # Handle edge case where caption might be in kwargs
        if not caption and 'caption' in kwargs:
            caption = kwargs['caption']
        
        # Validate input: ensure caption is provided and not empty
        if not caption or not str(caption).strip():
            return "Error: No caption provided. Unable to search PubMed."
        
        caption = str(caption).strip()
        
        # Configuration - Updated path handling
        BASE_DIR = Path(__file__).parent.parent.parent  # Up to main folder
        default_data_dir = str(BASE_DIR / "data")
        # Use metadata config if available, otherwise default
        data_dir = self.metadata.get("DATA_DIR", default_data_dir)
        top_k = self.metadata.get("TOP_K", 3)
        
        try:
            # Load FAISS index and metadata
            index_path = os.path.join(data_dir, "text_faiss.bin")
            metadata_path = os.path.join(data_dir, "raw_abstracts.jsonl")
            
            # Check if files exist
            if not os.path.exists(index_path):
                return f"Error: FAISS index not found at {index_path}"
            if not os.path.exists(metadata_path):
                return f"Error: Metadata file not found at {metadata_path}"
            
            # Read FAISS index from disk
            index = faiss.read_index(index_path)
            # Load metadata (PubMed abstracts) from JSONL file
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = [json.loads(line) for line in f]
            
            # Embed the input caption using Specter2 model
            query_vec = embed_texts_specter2([caption]).astype("float32")
            # Search for top_k most similar articles in FAISS index
            scores, indices = index.search(query_vec, top_k)
            
            # Format results for output
            formatted = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
                entry = metadata[idx]
                formatted.append(
                    f"Citation {i}:\n"
                    f"PMID: {entry.get('pmid', 'Unknown')}\n"
                    f"Similarity Score: {score:.3f}\n"
                    f"Title: {entry.get('title', 'Untitled').strip()}\n"
                    f"Abstract: {entry.get('abstract', 'No abstract available.').strip()}\n"
                )
            
            # Return formatted citations separated by ---
            return "\n---\n".join(formatted)
            
        except Exception as e:
            # Handle any errors during retrieval
            return f"Error during PubMed search: {str(e)}"