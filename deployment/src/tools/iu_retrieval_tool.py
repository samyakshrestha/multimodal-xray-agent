from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional, Dict
import torch
import numpy as np
import json
import os
from PIL import Image
from pathlib import Path
from open_clip import create_model_from_pretrained
import torch.nn.functional as F

# ----- Input schema for the tool -----
class IUImageInput(BaseModel):
    # Defines the expected input: absolute path to the query image
    image_path: str = Field(..., description="Absolute path to the query image")

# ----- Tool class -----
class IUImpressionSearchTool(BaseTool):
    # Tool metadata
    name: str = "iu_impression_search_tool"
    description: str = (
        "Retrieves the most similar IU X-ray image impression based on visual similarity "
        "using BiomedCLIP embeddings."
    )
    args_schema: type = IUImageInput  # Specifies input schema
    metadata: dict = {}  # Optional metadata for config overrides

    def _run(self, image_path: str) -> str:
        # Dynamic path resolution - same pattern as pubmed_tool
        BASE_DIR = Path(__file__).parent.parent.parent  # Up to main folder
        default_vecs_path = str(BASE_DIR / "data" / "iu_vecs.npy")  # Default path for IU vectors
        default_impr_path = str(BASE_DIR / "data" / "iu_impr.jsonl")  # Default path for IU impressions
        
        # Resolve config paths with dynamic defaults (can be overridden via metadata)
        vecs_path = self.metadata.get("VEC_PATH", default_vecs_path)
        impr_path = self.metadata.get("IMPR_PATH", default_impr_path)
        model_id = self.metadata.get("MODEL_ID", "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

        # Check if files exist before proceeding
        if not os.path.exists(vecs_path):
            return f"Error: IU vectors file not found at {vecs_path}"
        if not os.path.exists(impr_path):
            return f"Error: IU impressions file not found at {impr_path}"

        # Load BiomedCLIP model and processor
        try:
            model, preprocess = create_model_from_pretrained(model_id)
            model = model.to(device).eval()  # Move model to device and set to eval mode
        except Exception as e:
            return f"Error loading BiomedCLIP model: {e}"

        # Embed the input image
        try:
            image = Image.open(image_path).convert("RGB")  # Open and convert image to RGB
            tensor_img = preprocess(image).unsqueeze(0).to(device)  # Preprocess and add batch dimension
            with torch.no_grad():
                query_vec = model.encode_image(tensor_img)  # Get image embedding
            query_vec = F.normalize(query_vec, dim=-1).cpu().numpy()  # Normalize and move to CPU numpy
        except Exception as e:
            return f"Error processing input image: {e}"

        # Load stored IU embeddings
        try:
            iu_vecs = np.load(vecs_path)  # Load precomputed IU image vectors
            iu_vecs = iu_vecs / np.linalg.norm(iu_vecs, axis=1, keepdims=True)  # Normalize vectors
        except Exception as e:
            return f"Error loading IU vectors: {e}"

        # Compute cosine similarity between query and all IU vectors
        similarities = np.dot(iu_vecs, query_vec.squeeze())  # Dot product for cosine similarity
        best_idx = int(np.argmax(similarities))  # Index of most similar IU image

        # Load the matched impression
        try:
            with open(impr_path, "r", encoding="utf-8") as f:
                records = [json.loads(line) for line in f]  # Load all impression records
            best_match = records[best_idx]  # Get the best match by index
        except Exception as e:
            return f"Error loading impressions metadata: {e}"

        # Return formatted result with UUID and impression text
        return (
            f"Closest Match UUID: {best_match['uuid']}\n\n"
            f"Impression:\n{best_match['impression'].strip()}"
        )