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
    "prompthero/openjourney-v4",
    "dataautogpt3/ProteusV0.2",
    "dataautogpt3/ProteusV0.3"
]

logger = open("log.txt", "wt")

def pipeline_callback(pipe, index, timestamp, callback_kwargs):
    logger.write(f"index: {index}, timestamp: {timestamp}")
    logger.write("\n")
    return callback_kwargs

def get_pipeline(model_name: str):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        print("CUDA & MPS devices not found.")

    print("Testing torch device")
    torch.ones(2, device=device)

    if model_name == model_choices[0]:
        pipeline = DiffusionPipeline.from_pretrained(
            "stablediffusionapi/juggernaut-xl-v5",
            torch_dtype=torch.float16
        ).to(device)
        pipeline.load_lora_weights("ehristoforu/dalle-3-xl")

    elif model_name == model_choices[1]:
        pipeline = DiffusionPipeline.from_pretrained(
            "prompthero/openjourney-v4"
        ).to(device)
        pipeline.load_lora_weights("prompthero/openjourney-lora")
        
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
        callback_on_step_end=pipeline_callback,
        num_inference_steps=num_inference_steps
    )
    return output.images


if __name__ == '__main__':
    demo = gr.Interface(
        fn=infer,
        allow_flagging='never',
        inputs=[
            gr.TextArea("", label="Image Prompt"),
            gr.Text(
                "nsfw, bad quality, bad anatomy, worst quality, low quality, low resolutions, extra fingers, blur, "
                "blurry, ugly, wrongs proportions, watermark, image artifacts, lowres, ugly, jpeg artifacts, "
                "deformed, noisy image",
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

    demo.queue()
    demo.launch(share=False)
