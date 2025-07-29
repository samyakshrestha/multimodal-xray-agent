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
    """
Examine the chest X-ray step by step, following a structured A–G radiological workflow (Airway & Mediastinum, Bones & soft tissues, Cardiac silhouette, Diaphragm, Lung fields, Pleura, any Devices/foreign objects, and “Global” sanity checks).For each region, mentally assess both normal and abnormal findings before synthesizing them into a cohesive narrative report.
All directional terms (left/right) must refer strictly to the PATIENT’S perspective.
When evaluating cardiac size on AP films, assume that mild to moderate enlargement may be projectional unless the heart silhouette is clearly disproportionate or supported by additional findings (e.g., pulmonary congestion).If abnormal patterns are seen (e.g., opacities, effusion, pneumothorax, atelectasis, consolidation), propose the most likely clinical significance in a professional and cautious tone, consistent with expert radiology language. Weigh the clinical importance of each finding, and simulate a brief SECOND-PASS REVIEW of the image to verify that no significant abnormalities were overlooked.
Your final output must be formatted with two sections only:
FINDINGS:Structured prose following (but not explicitly labeling) the A–G sweep. Mention technical limitations only if they meaningfully impact interpretation. Always explicitly comment on signs of chronic lung disease (emphysema, fibrosis, interstitial changes) as either present or absent.
IMPRESSION:A concise, prioritized interpretation that integrates key findings into a diagnostic hypothesis.Rank findings by CLINICAL SIGNIFICANCE, listing the most urgent or actionable abnormalities first. Remaining key diagnoses/differentials, each with a probability qualifier (“probable”, “possible”, etc.). Use confident, direct language—but avoid speculation beyond what is visibly supported.
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
                    "content": "You are a board-certified thoracic radiologist with over 20 years of experience in interpreting chest X-rays. You are known for your meticulous attention to detail, clinical restraint, and deep respect for image-grounded reasoning. You prioritize accuracy over speculation and communicate with diagnostic clarity, always aligning your impressions with what is visibly demonstrable in the radiograph."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}}
                    ]
                }
            ],
            temperature=0.2,
            model="meta-llama/llama-4-maverick-17b-128e-instruct"
        )

        return response.choices[0].message.content.strip()