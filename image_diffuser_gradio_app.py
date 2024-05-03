import torch
from diffusers import (
    DiffusionPipeline,
    StableDiffusionXLPipeline,
    AutoPipelineForText2Image,
    KDPM2AncestralDiscreteScheduler,
    LCMScheduler,
    AutoencoderKL
)
from PIL import Image
import gradio as gr
# from datetime import datetime


model_choices = [
    "ehristoforu/dalle-3-xl",
    "ehristoforu/dalle-3-xl-v2",
    "Lykon/dreamshaper-7",
    "dataautogpt3/ProteusV0.3"
]

pipelines = {}  # Store loaded pipelines for reuse
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    print("CUDA & MPS devices not found.")
print("Testing torch device")
print(torch.ones(2, device=device))
logger = open("log.txt", "at")

def pipeline_callback(pipe, index, timestamp, callback_kwargs):
    logger.write(f"index: {index}, timestamp: {timestamp}")
    logger.write("\n")
    return callback_kwargs

def get_pipeline(model_name: str):
    if model_name not in pipelines:
        if model_name == model_choices[0]:
            pipeline = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float16
            )
            pipeline.load_lora_weights(model_name)
            pipeline = pipeline.to(device)
            
        elif model_name == model_choices[1]:
            pipeline = DiffusionPipeline.from_pretrained("fluently/Fluently-XL-v2")
            pipeline.load_lora_weights("ehristoforu/dalle-3-xl-v2")
            pipeline = pipeline.to(device)

        elif model_name == model_choices[2]:
            pipeline = AutoPipelineForText2Image.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                variant="fp16"
            ).to(device)
            pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config, torch_dtype=torch.float16)
            pipeline.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
            pipeline.fuse_lora()

        else:
            vae = AutoencoderKL.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix",
                torch_dtype=torch.float16
            )
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                model_name,
                vae=vae,
                torch_dtype=torch.float16
            )
            pipeline.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
            pipeline = pipeline.to(device)

        pipelines[model_name] = pipeline  # Store the loaded pipeline

    return pipelines[model_name]  # Return the cached pipeline


def infer(
    prompt: str,
    negative_prompt: str,
    input_image: Image,
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
        image=input_image,
        width=width,
        height=height,
        num_images_per_prompt=images_per_prompt,
        guidance_scale=guidance_scale,
        cfg_scale=cfg_scale,
        callback_on_step_end=pipeline_callback,
        num_inference_steps=num_inference_steps
    )
    # for image in output.images:
    #     try:
    #         filename = datetime.now().strftime("%H-%M-%S-%f")
    #         image.save(f"outputs/image_{filename}.png")
    #     except:
    #         print("error saving")
    return output.images


if __name__ == '__main__':
  with gr.Blocks() as interface:
    with gr.Row():
        # Output section (remains outside the columns)
        with gr.Column(elem_classes=["left-column"]):
            # Prompt and Negative Prompt section
            output = gr.Gallery(label="Output Images")
            with gr.Row():
                prompt_input = gr.TextArea(label="Image Prompt")
                negative_prompt_input = gr.TextArea(
                    value="nsfw, bad quality, bad anatomy, worst quality, low quality, low resolutions, extra fingers, blur, "
                    "blurry, ugly, wrongs proportions, watermark, image artifacts, lowres, ugly, jpeg artifacts, "
                    "deformed, noisy image",  # Pre-fill negative prompt
                    label="Negative Prompts"
                )
            with gr.Accordion("Additional Inputs"):
                input_img = gr.Image(label="Refrence image", type='pil')
            submit_btn = gr.Button("Generate!", variant='primary')
        with gr.Column(elem_classes=["right-column"]):
            # Other parameters section
            model_dropdown = gr.Dropdown(choices=model_choices, label="Model Choices", value=model_choices[0])
            cfg_scale_slider = gr.Slider(6, 9, label="CFG Scale", step=0.5, value=6.5)
            guidance_scale_slider = gr.Slider(0, 12, label="Guidance Scale", step=0.5, value=5.5)
            num_inference_steps_slider = gr.Slider(20, 60, step=1, label="Number of Inference Steps", value=30)
            image_outputs_slider = gr.Slider(1, 10, step=1, label="Image Outputs", value=2)
            image_width_number = gr.Number(value=608, label="Width", info="(Divisible by 8)")
            image_height_number = gr.Number(value=768, label="Height", info="(Divisible by 8)")

        inputs = [
            prompt_input,
            negative_prompt_input,
            input_img,
            cfg_scale_slider,
            guidance_scale_slider,
            num_inference_steps_slider,
            image_outputs_slider,
            image_width_number,
            image_height_number,
            model_dropdown,
        ]

        submit_btn.click(fn=infer, inputs=inputs, outputs=output)
        interface.launch()