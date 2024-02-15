import torch
from diffusers import (
    DiffusionPipeline,
    StableDiffusionXLPipeline,
    KDPM2AncestralDiscreteScheduler,
    AutoencoderKL
)
import gradio as gr

model_choices = [
    "ehristoforu/dalle-3-xl",
    "dataautogpt3/ProteusV0.2",
    "dataautogpt3/ProteusV0.3"
]

def get_pipeline(model_name: str):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        print ("CUDA & MPS devices not found.")

    print("Testing torch device")
    torch.ones(2, device=device) 

    if model_name == model_choices[0]:
        pipeline = DiffusionPipeline.from_pretrained("stablediffusionapi/juggernaut-xl-v5").to(device)
        pipeline.load_lora_weights("ehristoforu/dalle-3-xl")
    else:
        # Load VAE component
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16
        )
        # Configure the pipeline
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_name, 
            vae=vae,
            torch_dtype=torch.float16
        )
        pipeline.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to(device)
    return pipeline

def infer(
    prompt: str,
    negative_prompt: str,
    cfg_scale: float,
    guidance_scale: float,
    num_inference_steps: int,
    images_per_prompt: int,
    width: int,
    height: int,
    model_choice: str
):
    pipeline = get_pipeline(model_choice)
    output = pipeline(
        prompt, 
        negative_prompt=negative_prompt, 
        width=width,
        height=height,
        num_images_per_prompt=images_per_prompt,
        guidance_scale=guidance_scale,
        cfg_scale=cfg_scale,
        num_inference_steps=num_inference_steps
    )
    return output.images

if __name__=='__main__':
    demo = gr.Interface(
        fn=infer,
        allow_flagging=False,
        inputs=[
            gr.TextArea("", label="Image Prompt"),
            gr.Text(
                "nsfw, bad quality, bad anatomy, worst quality, low quality, low resolutions, extra fingers, blur, blurry, ugly, wrongs proportions, watermark, image artifacts, lowres, ugly, jpeg artifacts, deformed, noisy image",
                label="Negative Prompts"
            ),
            gr.Slider(6, 9, label="CFG Scale"),
            gr.Slider(4, 12, label="Guidance Scale"),
            gr.Slider(20, 60, step=1, label="Number of Inference Steps"),
            gr.Slider(1, 10, step=1, label="Image Outputs", value=2),
            gr.Number(value=512, label="Width of output image (should be divisible by 8)"),
            gr.Number(value=608, label="Height of output image (should be divisible by 8)"),
            gr.Dropdown(
                choices=model_choices,
                value=model_choices[0],
                label="Model Choices"
            ),
        ],
        outputs=gr.Gallery(label="Output Images")
    )

    demo.launch(share=True)