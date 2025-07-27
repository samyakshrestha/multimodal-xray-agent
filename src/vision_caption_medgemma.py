from crewai.tools import BaseTool, tool
from pydantic import BaseModel, Field
from typing import Optional
from transformers import pipeline
from PIL import Image
import torch
import os

class MedGemmaToolSchema(BaseModel):
    image_path: str
    prompt: Optional[str] = None

class MedGemmaTool(BaseTool):
    name: str = "medgemma_tool"
    description: str = (
        "Generates a structured radiology report from a chest X-ray image using MedGemma-4B (Google's medical specialist model)."
    )
    args_schema: type = MedGemmaToolSchema
    
    # Class variable to store the pipeline (shared across instances)
    _pipeline = None
    
    @classmethod
    def get_pipeline(cls):
        if cls._pipeline is None:
            print("Loading MedGemma-4B model... This may take a few minutes on first run.")
            cls._pipeline = pipeline(
                "image-text-to-text",
                model="google/medgemma-4b-it",
                torch_dtype=torch.bfloat16,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            print("MedGemma-4B model loaded successfully!")
        return cls._pipeline

    def _run(self, image_path: str, prompt: Optional[str] = None) -> str:
        # Get the pipeline
        pipe = self.get_pipeline()
        
        # Use fallback prompt if none provided
        if prompt is None:
            prompt = (
                '''
                Examine the chest X-ray step by step, following a structured A–G radiological workflow (Airway, Bones & soft tissues, Cardiac silhouette, Diaphragm, Lung fields, Pleura, and any Devices/foreign objects).
                For each region, mentally assess both normal and abnormal findings before synthesizing them into a cohesive narrative report.

                When describing findings, always prefer specific visual observations over vague statements. 
                If abnormal patterns are seen (e.g., opacities, effusion, pneumothorax), propose the most likely clinical significance in a cautious, professional tone, mirroring expert radiology style.
                
                Your final output must be formatted with two sections only:
                FINDINGS: Detailed observations based strictly on image evidence.
                IMPRESSION: A concise, prioritized interpretation, integrating the key findings into a diagnostic hypothesis.

                Always explicitly comment on signs of chronic lung disease (emphysema, fibrosis, interstitial changes) if present or absent.
                Avoid patient identifiers, clinical indication, comparison sections, or generic placeholders like 'No acute findings.'
                The tone should be confident but never speculative beyond what the image supports.
                '''
            )

        # Verify image exists
        if not os.path.exists(image_path):
            raise ValueError(f"Image file not found: {image_path}")

        # Load image using PIL
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to load image {image_path}: {str(e)}")

        # Create messages in MedGemma format (no system prompt as requested)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image}
                ]
            }
        ]

        # Generate response
        try:
            output = pipe(text=messages, max_new_tokens=512)
            
            # Extract the generated text from MedGemma output format
            if isinstance(output, list) and len(output) > 0:
                response_text = output[0]["generated_text"][-1]["content"]
            else:
                response_text = str(output)
            
            return response_text.strip()
            
        except Exception as e:
            return f"ERROR: Failed to generate report with MedGemma-4B: {str(e)}"