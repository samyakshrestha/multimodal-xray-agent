import gradio as gr
import time
from src.pipeline import generate_report

# ------------------------------------------------------------------
# 1. Pre-load models (unchanged)
# ------------------------------------------------------------------
from src.tools_loader import get_tools
_ = get_tools()  # Pre-load necessary models/tools for report generation

# ------------------------------------------------------------------
# 2. Helper: streaming generator with progress
# ------------------------------------------------------------------
def process_upload(image_path: str):
    """
    Streamed generator: yields loading states then final report.
    Gradio shows spinner automatically during execution.
    """
    if image_path is None:
        yield "**Please upload a chest X-ray image to begin analysis.**"  # Prompt user to upload image if none provided
        return

    start = time.time()  # Record start time for performance measurement
    
    # Generate the actual report
    report = generate_report(image_path)  # Call pipeline to generate radiology report
    
    elapsed = time.time() - start  # Calculate elapsed time
    
    # Return final formatted report
    yield f"""### Radiology Report

{report}

---
*Generated in {elapsed:.1f} seconds*"""  # Display report and time taken

# ------------------------------------------------------------------
# 3. Gradio UI - Vertical Layout
# ------------------------------------------------------------------
with gr.Blocks(
    theme=gr.themes.Soft(),  # Use Gradio's Soft theme for UI
    title="Multi-Agent Radiology Assistant",  # Set page title
    css="""
    .image-container { max-width: 600px; margin: 0 auto; }
    .report-container { margin-top: 40px; padding-top: 20px; }
    .generate-btn { margin: 30px auto; display: block; }
    .progress-bar { z-index: 1000 !important; position: relative; }
    .gradio-container .wrap { z-index: auto; }
    """  # Custom CSS for layout and styling
) as demo:
    
    # Header
    gr.Markdown(
        "# Multi-Agent Radiology Assistant\n"
        "Upload a chest X-ray image to receive an AI-powered radiology report"
    )  # Display main header and instructions
    
    # Image upload section (centered, top)
    with gr.Column():
        input_image = gr.Image(
            type="filepath",  # Accept image file path
            label="Upload Chest X-ray Image",  # Label for upload box
            height=400,  # Set image box height
            elem_classes=["image-container"]  # Apply custom CSS class
        )
        
        # Generate button (centered with more spacing)
        generate_btn = gr.Button(
            "Generate Report",  # Button text
            variant="primary",  # Primary button style
            size="lg",  # Large button size
            elem_classes=["generate-btn"]  # Apply custom CSS class
        )
    
    # Report output section (bottom)
    with gr.Column(elem_classes=["report-container"]):
        output_report = gr.Markdown(
            value="**Ready to analyze**\n\nUpload an X-ray image above and click 'Generate Report' to begin.",  # Initial output message
            label="Analysis Results"  # Output label
        )
    
    # Event handler with progress settings
    generate_btn.click(
        fn=process_upload,  # Function to call on button click
        inputs=input_image,  # Pass uploaded image as input
        outputs=output_report,  # Display output in Markdown box
        show_progress="full",  # Shows Gradio's built-in progress bar
        concurrency_limit=1   # Prevent multiple simultaneous requests
    )
    
    # Footer with example hint
    gr.Markdown(
        "Download and use any frontal chest X-ray PNG/JPG file from the internet and click **Generate Report**.\n"
        "### NOTE: This is just a demo. It is not intended to diagnose or suggest treatment of any disease or condition, and should not be used for medical advice."
    )  # Display disclaimer and usage hint

if __name__ == "__main__":
    demo.launch()  # Launch Gradio app if script is run directly