from diffusers import StableDiffusionPipeline
import torch

# Detect available device
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
elif torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float16
else:
    device = "cpu"
    dtype = torch.float32

print(f"Using device: {device}")

model_id = "runwayml/stable-diffusion-v1-5"

# Global variable - loaded once
_pipe = None


def get_pipeline():
    """Load pipeline (cached after first call)."""
    global _pipe
    if _pipe is None:
        print("Loading Stable Diffusion model (this may take a few minutes on first run)...")
        _pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype
        )
        _pipe = _pipe.to(device)
        print("Model loaded successfully!")
    return _pipe


def generate_art(prompt, num_inference_steps=50, guidance_scale=7.5):
    """Generate an image from a text prompt."""
    pipe = get_pipeline()
    
    with torch.inference_mode():
        image = pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        ).images[0]

    return image


if __name__ == "__main__":
    img = generate_art("Painting in style of Van Gogh, oil painting")
    img.save("output.png")
    print("Saved to output.png")
