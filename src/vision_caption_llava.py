from crewai.tools import BaseTool, tool
from pydantic import BaseModel, Field
from typing import Optional
from transformers import pipeline
from PIL import Image
import os

class LLaVAMedToolSchema(BaseModel):
    image_path: str
    prompt: Optional[str] = None

class LLaVAMedTool(BaseTool):
    name: str = "llava_med_tool"
    description: str = (
        "Generates a structured radiology report from a chest X-ray image using LLaVA-Med (fine-tuned for medical imaging)."
    )
    args_schema: type = LLaVAMedToolSchema
    
    def __init__(self):
        super().__init__()
        # Initialize the pipeline once when the tool is created
        print("Loading LLaVA-Med model... This may take a few minutes on first run.")
        self.pipe = pipeline(
            "image-text-to-text", 
            model="microsoft/llava-med-v1.5-mistral-7b",
            device_map="auto"  # Automatically use GPU if available
        )
        print("LLaVA-Med model loaded successfully!")

    def _run(self, image_path: str, prompt: Optional[str] = None) -> str:
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

        # Load image using PIL (required for HuggingFace pipeline)
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to load image {image_path}: {str(e)}")

        # Create messages in the format expected by LLaVA-Med
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},  # Pass PIL Image object directly
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Generate response
        try:
            result = self.pipe(messages)
            # Extract text from the pipeline result
            if isinstance(result, list) and len(result) > 0:
                response_text = result[0].get('generated_text', '')
            else:
                response_text = str(result)
            
            return response_text.strip()
            
        except Exception as e:
            return f"ERROR: Failed to generate report with LLaVA-Med: {str(e)}"