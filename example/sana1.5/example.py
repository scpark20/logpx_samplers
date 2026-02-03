import torch
from diffusers import SanaPipeline

pipe = SanaPipeline.from_pretrained(
    "Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

pipe.text_encoder.to(torch.bfloat16)

# pipe.enable_model_cpu_offload()

prompt = 'Self-portrait oil painting, a beautiful cyborg with golden hair, 8k'
image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=4.5,
    num_inference_steps=20,
)[0]

image[0].save(f"sana1.5.png")
