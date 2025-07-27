
from crewai.tools import BaseTool, tool
from pydantic import BaseModel, Field
from typing import Optional
from google import genai
from google.genai.types import Part
import os

class VisionCaptionToolSchema(BaseModel):
    image_path: str
    prompt: Optional[str] = None

class VisionCaptionTool(BaseTool):
    name: str = "vision_caption_tool"
    description: str = (
        "Generates a structured radiology report from a chest X-ray image using Gemini 2.5 Flash."
    )
    args_schema: type = VisionCaptionToolSchema
    metadata: dict = {}

    def _run(self, image_path: str, prompt: Optional[str] = None) -> str:
        # Resolve API key
        api_key = self.metadata.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in metadata.")
        client = genai.Client(api_key=api_key)

        # Use fallback prompt if none provided
        if prompt is None:
            prompt = (
                '''
                Examine the chest X-ray step by step, following a structured A–G radiological workflow (Airway, Bones & soft tissues, Cardiac silhouette, Diaphragm, Lung fields, Pleura, and any Devices/foreign objects).
                For each region, mentally assess both normal and abnormal findings before synthesizing them into a cohesive narrative report.

                When describing findings, always prefer specific visual observations over vague statements. 
                If abnormal patterns are seen (e.g., opacities, effusion, pneumothorax), propose the most likely clinical significance in a cautious, professional tone, mirroring expert radiology style.
                
                Your final output must be formatted with two sections only:
                Findings: Detailed observations based strictly on image evidence.
                Impression: A concise, prioritized interpretation, integrating the key findings into a diagnostic hypothesis.

                Always explicitly comment on signs of chronic lung disease (emphysema, fibrosis, interstitial changes) if present or absent.
                Avoid patient identifiers, clinical indication, comparison sections, or generic placeholders like ‘No acute findings.
                The tone should be confident but never speculative beyond what the image supports.
                '''
            )
            
        # Determine MIME type
        ext = os.path.splitext(image_path)[-1].lower()
        if ext == ".png":
            mime_type = "image/png"
        elif ext in [".jpg", ".jpeg"]:
            mime_type = "image/jpeg"
        else:
            raise ValueError(f"Unsupported image type: {ext}")

        # Load image
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        # Create input parts
        parts = [
            Part.from_bytes(data=image_bytes, mime_type=mime_type),
            prompt  # must be a valid string
        ]

        # Generate content
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=parts
        )

        return response.text.strip()
