from crewai.tools import BaseTool, tool
from pydantic import BaseModel, Field
from typing import Optional
from openai import OpenAI
import base64
import os

class GPT4oVisionToolSchema(BaseModel):
    image_path: str
    prompt: Optional[str] = None

class GPT4oVisionTool(BaseTool):
    name: str = "gpt4o_vision_tool"
    description: str = (
        "Generates a structured radiology report from a chest X-ray image using GPT-4o."
    )
    args_schema: type = GPT4oVisionToolSchema
    metadata: dict = {}

    def _run(self, image_path: str, prompt: Optional[str] = None) -> str:
        # Resolve API key
        api_key = self.metadata.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in metadata.")
        client = OpenAI(api_key=api_key)

        # Use fallback prompt if none provided
        if prompt is None:
            prompt = (
                '''
                You are performing a detailed visual analysis and description of a chest radiograph image for educational purposes only. This is not for clinical diagnosis or patient care.

                TASK: Examine the chest X-ray step by step, following a structured A–G radiological workflow (Airway, Bones & soft tissues, Cardiac silhouette, Diaphragm, Lung fields, Pleura, and any Devices/foreign objects).
                For each region, mentally assess both normal and abnormal findings before synthesizing them into a cohesive narrative report.

                When describing findings, always prefer specific visual observations over vague statements. 
                If abnormal patterns are seen (e.g., opacities, effusion, pneumothorax), propose the most likely clinical significance in a cautious, professional tone, mirroring expert radiology style.
                
                Your final output must be formatted with two sections only:
                FINDINGS: Detailed observations based strictly on image evidence.
                IMPRESSION: A concise, prioritized interpretation, integrating the key findings into a diagnostic hypothesis.

                Always explicitly comment on signs of chronic lung disease (emphysema, fibrosis, interstitial changes) if present or absent.
                Avoid patient identifiers, clinical indication, comparison sections, or generic placeholders like 'No acute findings.'
                The tone should be confident but never speculative beyond what the image supports.

                This is for educational image analysis only, not clinical decision-making.
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

        # Load and encode image to base64
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # Create OpenAI API call with vision
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000,
                temperature=0.5
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            raise ValueError(f"Failed to generate report with GPT-4o: {str(e)}")