from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
)

pipe = pipe.to("cuda")

def generate_art(prompt, num_inference_steps=50, guidance_scale=8):

    with torch.autocast("cuda"):
        image = pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        ).images[0]

    return image


if __name__ == "__main__":
    img = generate_art("Painting in style of Van Gogh, oil painting")
    img.save("output.png")