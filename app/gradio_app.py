#!/usr/bin/env python3
"""
Gradio Web Interface for Neural Style Transfer

Run with:
    python app/gradio_app.py

Then open http://localhost:7860 in your browser.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from PIL import Image

from models.gatys import GatysStyleTransfer
from models.fast_style import FastStyleTransfer
from utils.image_utils import load_image, get_device


# Global model instances (lazy loaded)
_gatys_model = None
_fast_models = {}


def get_gatys_model(content_weight: float, style_weight: float, num_steps: int):
    """Get or create Gatys model with specified parameters."""
    global _gatys_model
    _gatys_model = GatysStyleTransfer(
        content_weight=content_weight,
        style_weight=style_weight,
        num_steps=num_steps
    )
    return _gatys_model


def gatys_transfer(
    content_image: Image.Image,
    style_image: Image.Image,
    content_weight: float,
    style_weight: float,
    num_steps: int,
    max_size: int,
    progress=gr.Progress()
) -> Image.Image:
    """
    Perform Gatys-style neural style transfer.
    """
    if content_image is None:
        raise gr.Error("Please upload a content image")
    if style_image is None:
        raise gr.Error("Please upload a style image")
    
    # Resize images
    content_image = resize_image(content_image, max_size)
    style_image = resize_image(style_image, max_size)
    
    progress(0, desc="Initializing...")
    
    # Create model
    model = get_gatys_model(content_weight, style_weight, int(num_steps))
    
    # Progress callback
    def update_progress(step, loss):
        progress(step / num_steps, desc=f"Step {step}/{int(num_steps)}")
    
    progress(0.1, desc="Starting style transfer...")
    
    # Run transfer
    result = model.transfer(
        content_image=content_image,
        style_image=style_image,
        init_image="content",
        optimizer_type="lbfgs",
        progress_callback=update_progress
    )
    
    progress(1.0, desc="Complete!")
    return result


def resize_image(image: Image.Image, max_size: int) -> Image.Image:
    """Resize image to fit within max_size while preserving aspect ratio."""
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = tuple(int(dim * ratio) for dim in image.size)
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    return image


def create_interface():
    """Create the Gradio interface."""
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        font-family: 'Helvetica Neue', Arial, sans-serif;
    }
    .main-title {
        text-align: center;
        margin-bottom: 20px;
    }
    """
    
    with gr.Blocks(css=css, title="Neural Style Transfer") as demo:
        gr.Markdown(
            """
            # 🎨 Neural Style Transfer
            
            Transform your photos using the artistic style of famous paintings!
            
            **How to use:**
            1. Upload a content image (your photo)
            2. Upload a style image (artwork you want to mimic)
            3. Adjust settings and click "Transfer Style"
            
            *Optimized for 8GB Mac systems with MPS acceleration*
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                content_input = gr.Image(
                    label="📷 Content Image",
                    type="pil",
                    height=300
                )
            
            with gr.Column(scale=1):
                style_input = gr.Image(
                    label="🖼️ Style Image",
                    type="pil",
                    height=300
                )
        
        with gr.Row():
            with gr.Column(scale=2):
                output_image = gr.Image(
                    label="✨ Stylized Result",
                    type="pil",
                    height=400
                )
        
        with gr.Accordion("⚙️ Advanced Settings", open=False):
            with gr.Row():
                max_size = gr.Slider(
                    minimum=128,
                    maximum=768,
                    value=512,
                    step=64,
                    label="Max Image Size",
                    info="Lower = faster, higher = more detail"
                )
                num_steps = gr.Slider(
                    minimum=50,
                    maximum=500,
                    value=200,
                    step=50,
                    label="Optimization Steps",
                    info="More steps = better quality"
                )
            
            with gr.Row():
                content_weight = gr.Slider(
                    minimum=0.1,
                    maximum=10,
                    value=1,
                    step=0.1,
                    label="Content Weight",
                    info="Higher = preserve more content"
                )
                style_weight = gr.Slider(
                    minimum=1e4,
                    maximum=1e8,
                    value=1e6,
                    step=1e4,
                    label="Style Weight",
                    info="Higher = apply more style"
                )
        
        transfer_btn = gr.Button(
            "🎨 Transfer Style",
            variant="primary",
            size="lg"
        )
        
        # Examples
        gr.Markdown("### 📚 Example Styles")
        gr.Markdown(
            """
            Try these famous art styles:
            - **Starry Night** by Van Gogh - Swirling, expressive brushwork
            - **The Great Wave** by Hokusai - Bold Japanese woodblock style
            - **Composition VIII** by Kandinsky - Abstract geometric patterns
            - **The Scream** by Munch - Intense, wavy distortions
            """
        )
        
        # Device info
        device = get_device()
        device_info = {
            "mps": "🍎 Apple Metal (MPS)",
            "cuda": "🎮 NVIDIA CUDA",
            "cpu": "💻 CPU"
        }
        gr.Markdown(f"**Device:** {device_info.get(device.type, device.type)}")
        
        # Connect the button
        transfer_btn.click(
            fn=gatys_transfer,
            inputs=[
                content_input,
                style_input,
                content_weight,
                style_weight,
                num_steps,
                max_size
            ],
            outputs=output_image
        )
    
    return demo


def main():
    """Launch the Gradio app."""
    print("Starting Neural Style Transfer Web App...")
    print(f"Device: {get_device()}")
    
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
