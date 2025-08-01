import gradio as gr
import time
from src.pipeline import generate_report

# ------------------------------------------------------------------
# 1. Pre-load models (unchanged)
# ------------------------------------------------------------------
from src.tools_loader import get_tools
_ = get_tools()

# ------------------------------------------------------------------
# 2. Helper: streaming generator
# ------------------------------------------------------------------
def process_upload(image_path: str):
    """
    Streamed generator: yields placeholder first,
    then final report with inference time.
    Gradio automatically shows a spinner / progress bar
    while the function is running.
    """
    if image_path is None:
        yield "Please upload a chest-X-ray image."
        return

    start = time.time()
    # Initial placeholder so the textarea updates immediately
    yield " **Generating report...**\n\nThis may take a few seconds..."

    # Optional manual progress bar
    # with gr.Progress(track_tqdm=True) as progress:
    #     report = generate_report(image_path, progress=progress)

    report = generate_report(image_path)

    elapsed = time.time() - start
    yield f"### Radiology Report\n{report}\n\n---\n*Generated in `{elapsed:0.1f}` s*"

# ------------------------------------------------------------------
# 3. Gradio UI
# ------------------------------------------------------------------
with gr.Blocks() as demo:
    gr.Markdown("# Multi-Agent Radiology Assistant")
    with gr.Row():
        input_image = gr.Image(type="filepath", label="Upload Chest X-ray", height=400)
        output_report = gr.Markdown()

    generate_btn = gr.Button("Generate Report")

    generate_btn.click(
        fn=process_upload,            # generator function
        inputs=input_image,
        outputs=output_report
    )

if __name__ == "__main__":
    demo.launch()