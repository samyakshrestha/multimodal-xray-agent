from crewai.tools import BaseTool, tool
from pydantic import BaseModel, Field
from typing import Optional
from groq import Groq
import base64
import os

class VisionCaptionToolSchema(BaseModel):
    image_path: str
    prompt: Optional[str] = None

class VisionCaptionTool(BaseTool):
    name: str = "vision_caption_tool"
    description: str = (
        "Generates a structured radiology report from a chest X-ray image using Llama 4 Maverick via Groq."
    )
    args_schema: type = VisionCaptionToolSchema
    metadata: dict = {}

    def _run(self, image_path: str, prompt: Optional[str] = None) -> str:
        # Resolve API key
        api_key = self.metadata.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in metadata.")
        client = Groq(api_key=api_key)

        # Use fallback prompt if none provided
        if prompt is None:
            prompt = (
                """Examine the chest X-ray step by step, following a structured A–G radiological workflow (Airway, Bones & soft tissues, Cardiac silhouette, Diaphragm, Lung fields, Pleura, and any Devices/foreign objects).  
For each region, mentally assess both normal and abnormal findings, and ensure that every step is addressed, even if normal or limited by image quality. If a structure cannot be evaluated, state this explicitly.

All directional terms (LEFT/RIGHT) must refer strictly to the PATIENT’S perspective, and must be cross-referenced with radiographic markers (e.g., “L”, “R”) and expected anatomical landmarks when available.

When describing findings, always prefer specific visual observations over vague generalities.  
If abnormal patterns are seen (e.g., opacities, effusion, pneumothorax, atelectasis, consolidation, interstitial thickening, volume loss), propose the most likely clinical significance in a professional and cautious tone, consistent with expert radiology language.  
Weigh the clinical importance of each finding, and simulate a brief SECOND-PASS REVIEW of the image to verify that no significant abnormalities were overlooked.

Describe all medical devices (e.g., endotracheal tube, central lines, pacemakers, chest tubes) with attention to:
- PATIENT-side laterality
- Entry site and course
- Tip position
- Placement appropriateness

Your final output must be formatted with two sections only:

FINDINGS:  
Detailed, anatomically organized observations based strictly on image evidence. Always explicitly comment on signs of chronic lung disease (emphysema, fibrosis, interstitial changes) as either present or absent.

IMPRESSION:  
A concise, prioritized interpretation that integrates key findings into a diagnostic hypothesis.  
Rank findings by CLINICAL SIGNIFICANCE, listing the most urgent or actionable abnormalities first. Use confident, direct language—but avoid speculation beyond what is visibly supported.

Avoid patient identifiers, clinical indications, comparison sections, or generic placeholders like “no acute findings.”
"""
            )

        # Determine MIME type
        ext = os.path.splitext(image_path)[-1].lower()
        if ext == ".png":
            mime_type = "image/png"
        elif ext in [".jpg", ".jpeg"]:
            mime_type = "image/jpeg"
        else:
            raise ValueError(f"Unsupported image type: {ext}")

        # Load and encode image to base64
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # Create Groq API call with system prompt and proper multimodal format
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a world-renowned senior thoracic radiologist with over 20 years of clinical experience interpreting chest X-rays. You are meticulous, systematic, and articulate—blending sharp visual analysis with deep diagnostic reasoning. You never overlook subtle findings, always interpret anatomy from the patient’s perspective, and prioritize abnormalities by clinical urgency. Your reports are complete, precise, and modeled on the highest standards of radiologic practice."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}}
                    ]
                }
            ],
            model="meta-llama/llama-4-maverick-17b-128e-instruct"
        )

        return response.choices[0].message.content.strip()