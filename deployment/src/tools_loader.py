import os
from pathlib import Path
from .tools.vision_caption_tool import VisionCaptionTool  # Import vision caption tool
from .tools.pubmed_tool import PubmedRetrievalTool        # Import PubMed retrieval tool
from .tools.iu_retrieval_tool import IUImpressionSearchTool  # Import IU impression search tool

def get_tools():
    """Create and return all configured tools"""
    
    # Get paths and API keys
    data_dir = Path(__file__).parent.parent / "data"  # Define path to the data directory
    groq_key = os.getenv("GROQ_API_KEY")              # Retrieve GROQ API key from environment
    gemini_key = os.getenv("GEMINI_API_KEY")          # Retrieve Gemini API key from environment
    
    # Create tools
    vision_tool = VisionCaptionTool(metadata={"GEMINI_API_KEY": gemini_key})  # Initialize vision caption tool with Gemini API key
    pubmed_tool = PubmedRetrievalTool(metadata={"DATA_DIR": str(data_dir), "TOP_K": 3})  # Initialize PubMed tool with data directory and top-k parameter
    iu_tool = IUImpressionSearchTool(metadata={
        "VEC_PATH": str(data_dir / "iu_vecs.npy"),    # Path to IU vectors file
        "IMPR_PATH": str(data_dir / "iu_impr.jsonl"), # Path to IU impressions file
        "MODEL_ID": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"  # Model identifier for retrieval
    })
    
    # Return all tools in a dictionary
    return {
        "vision_tool": vision_tool,
        "pubmed_tool": pubmed_tool,
        "iu_tool": iu_tool
    }