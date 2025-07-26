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
        "Generates a detailed description on the image."
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
                "You are an expert historian with extensive knowledge about the past."
                "Describe the person shown in the image."
                "Name what he is famous for and his name."
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