from crewai.tools import BaseTool, tool
from pydantic import BaseModel, Field
from typing import Optional
from google import genai
from google.genai.types import Part, GenerateContentConfig
import os

# Schema for tool arguments, specifying image path and optional prompt
class VisionCaptionToolSchema(BaseModel):
    image_path: str
    prompt: Optional[str] = None

# Main tool class for generating radiology reports from chest X-ray images
class VisionCaptionTool(BaseTool):
    name: str = "vision_caption_tool"
    description: str = (
        "Generates a structured radiology report from a chest X-ray image using optimized Gemini 2.5 Flash."
    )
    args_schema: type = VisionCaptionToolSchema
    metadata: dict = {}

    # Core method to run the tool logic
    def _run(self, image_path: str, prompt: Optional[str] = None) -> str:
        # Retrieve Gemini API key from metadata
        api_key = self.metadata.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in metadata.")
        client = genai.Client(api_key=api_key)

        # System prompt defines the expert persona and approach for the model
        system_prompt = (
            "You are a board-certified thoracic radiologist with over 20 years of experience in interpreting chest X-rays. "
            "You are known for your meticulous attention to detail, clinical restraint, and deep respect for image-grounded reasoning. "
            "You prioritize accuracy over speculation and communicate with diagnostic clarity, always aligning your impressions with what is visibly demonstrable in the radiograph."
        )

        # Use default optimized prompt if none is provided by the user
        if prompt is None:
            prompt = (
                "Examine the chest X-ray step by step, following a structured A–G radiological workflow (Airway & Mediastinum, Bones & soft tissues, Cardiac silhouette, Diaphragm, Lung fields, Pleura, any Devices/foreign objects, and \"Global\" sanity checks). "
                "For each region, mentally assess both normal and abnormal findings before synthesizing them into a cohesive narrative report.\n"
                "All directional terms (left/right) must refer strictly to the PATIENT'S perspective.\n"
                "When evaluating cardiac size on AP films, assume that mild to moderate enlargement may be projectional unless the heart silhouette is clearly disproportionate or supported by additional findings (e.g., pulmonary congestion). "
                "If abnormal patterns are seen (e.g., opacities, effusion, pneumothorax, atelectasis, consolidation), propose the most likely clinical significance in a professional and cautious tone, consistent with expert radiology language. "
                "Weigh the clinical importance of each finding, and simulate a brief SECOND-PASS REVIEW of the image to verify that no significant abnormalities were overlooked.\n"
                "Your final output must be formatted with two sections only:\n"
                "FINDINGS: Structured prose following (but not explicitly labeling) the A–G sweep. Mention technical limitations only if they meaningfully impact interpretation. Always explicitly comment on signs of chronic lung disease (emphysema, fibrosis, interstitial changes) as either present or absent.\n"
                "IMPRESSION: A concise, prioritized interpretation that integrates key findings into a diagnostic hypothesis. Rank findings by CLINICAL SIGNIFICANCE, listing the most urgent or actionable abnormalities first. Remaining key diagnoses/differentials, each with a probability qualifier (\"probable\", \"possible\", etc.). Use confident, direct language—but avoid speculation beyond what is visibly supported."
            )
            
        # Determine the MIME type of the image based on its file extension
        ext = os.path.splitext(image_path)[-1].lower()
        if ext == ".png":
            mime_type = "image/png"
        elif ext in [".jpg", ".jpeg"]:
            mime_type = "image/jpeg"
        else:
            raise ValueError(f"Unsupported image type: {ext}")

        # Read the image file as bytes
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        
        # Prepare input parts for the Gemini model: image and prompt
        parts = [
            Part.from_bytes(data=image_bytes, mime_type=mime_type),
            prompt
        ]

        # Generate content using Gemini 2.5 Flash with specified parameters
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=parts,
            config=GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.2,
                max_output_tokens=2048
            )
        )

        # Return the generated report text, stripped of leading/trailing whitespace
        return response.text.strip()