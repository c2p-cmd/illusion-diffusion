import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    AutoencoderKL,
    StableDiffusionControlNetImg2ImgPipeline,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler
)
import gradio as gr
from datetime import datetime

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    print ("CUDA & MPS devices not found.")

print("Testing torch device")
print(device)

BASE_MODELS = [
    "SG161222/Realistic_Vision_V5.1_noVAE",
    "runwayml/stable-diffusion-v1-5"
]

# Sampler map
SAMPLER_MAP = {
    "DPM++ Karras SDE": lambda config: DPMSolverMultistepScheduler.from_config(config, use_karras=True, algorithm_type="sde-dpmsolver++"),
    "Euler": lambda config: EulerDiscreteScheduler.from_config(config),
}

# pre-trained vae
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)

# pre-trained controlnet
controlnet = ControlNetModel.from_pretrained("monster-labs/control_v1p_sd15_qrcode_monster", torch_dtype=torch.float16)


def setup_piepline(
    base_model: str,
    sampler = "Euler"
) -> tuple:
    # Load the pipeline and move it to the GPU
    latent_pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model,
        controlnet=controlnet,
        vae=vae,
        torch_dtype=torch.float16
    ).to(device)

    image_pipe_components = {k: v for k, v in latent_pipe.components.items() if k != 'image_encoder'}

    image_pipe = StableDiffusionControlNetImg2ImgPipeline(**image_pipe_components)
    image_pipe = image_pipe.to(device)

    latent_pipe.scheduler = SAMPLER_MAP[sampler](latent_pipe.scheduler.config)
    generator = torch.Generator(device=device)

    return (latent_pipe, generator, image_pipe)


# pre-processing images
def center_crop_resize(img, output_size=(512, 512)):
    width, height = img.size

    # Calculate dimensions to crop to the center
    new_dimension = min(width, height)
    left = (width - new_dimension)/2
    top = (height - new_dimension)/2
    right = (width + new_dimension)/2
    bottom = (height + new_dimension)/2

    # Crop and resize
    img = img.crop((left, top, right, bottom))
    img = img.resize(output_size)

    return img


def common_upscale(samples, width, height, upscale_method, crop=False):
    if crop == "center":
        old_width = samples.shape[3]
        old_height = samples.shape[2]
        old_aspect = old_width / old_height
        new_aspect = width / height
        x = 0
        y = 0
        if old_aspect > new_aspect:
            x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
        elif old_aspect < new_aspect:
            y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
        s = samples[:,:,y:old_height-y,x:old_width-x]
    else:
        s = samples

    return torch.nn.functional.interpolate(s, size=(height, width), mode=upscale_method)


def upscale(samples, upscale_method, scale_by):
    width = round(samples["images"].shape[3] * scale_by)
    height = round(samples["images"].shape[2] * scale_by)
    s = common_upscale(samples["images"], width, height, upscale_method, "disabled")
    return (s)


def inference(
    image,
    prompt: str,
    negative_prompt: str,
    guidance_scale: float,
    controlnet_conditioning_scale: float,
    latent_num_of_inference_steps: int,
    image_num_of_inference_steps: int,
    base_model: str,
    sampler: str
):
    (latent_pipe, generator, image_pipe) = setup_piepline(base_model, sampler)

    control_image_small = center_crop_resize(image)
    control_image_large = center_crop_resize(image, (1024, 1024))

    output = latent_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=control_image_small,
        guidance_scale=float(guidance_scale),
        controlnet_conditioning_scale=float(controlnet_conditioning_scale),
        num_inference_steps=latent_num_of_inference_steps,
        generator=generator,
        output_type="latent"
    )

    upscaled_latents = upscale(output, "nearest-exact", 2)

    out_image = image_pipe(
        prompt=prompt,
        image=upscaled_latents,
        negative_prompt=negative_prompt,
        control_image=control_image_large,
        guidance_scale=float(guidance_scale),
        controlnet_conditioning_scale=float(controlnet_conditioning_scale),
        num_inference_steps=image_num_of_inference_steps,
        generator=generator
    )

    final_output = out_image.images[0]

    now = datetime.now()
    image_name = now.strftime("%m-%d-%Y_%H-%M-%S")
    final_output.save(f"outputs/illusion_{image_name}.png")

    return final_output


def gradio_app(share=False):
    demo = gr.Interface(
        fn=inference,
        inputs=[
            gr.Image(label="Illusion Image", type='pil'),
            gr.TextArea(label="Prompt"),
            gr.TextArea(label="Negative Prompt", value="low quality, NSFW, violence, anomalies, blurry, ugly, wrong proportions, watermark, image artifacts, low-res, ugly, jpeg artifacts, deformed, noisy image"),
            gr.Slider(minimum=0.0, maximum=50.0, step=0.25, value=7.5, label="Guidance Scale"),
            gr.Slider(minimum=0.0, maximum=5.0, step=0.01, value=0.8, label="Illusion strength", info="ControlNet conditioning scale"),
            gr.Slider(minimum=10, maximum=30, label="Latent Inference Steps", value=15, step=1),
            gr.Slider(minimum=10, maximum=50, label="Image Inference Steps", value=25, step=1),
            gr.Dropdown(choices=BASE_MODELS, label="Base Model", value=BASE_MODELS[1]),
            gr.Dropdown(choices=SAMPLER_MAP, label="Sampler Choice", value="Euler")
        ],
        outputs="image",
        concurrency_limit=1
    )
    demo.queue()
    demo.launch(share)

if __name__=='__main__':
    gradio_app(share=True)