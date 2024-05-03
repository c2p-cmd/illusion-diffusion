import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import gradio as gr

checkpoints = {
    "1-Step" : ["sdxl_lightning_1step_unet_x0.safetensors", 1],
    "2-Step" : ["sdxl_lightning_2step_unet.safetensors", 2],
    "4-Step" : ["sdxl_lightning_4step_unet.safetensors", 4],
    "8-Step" : ["sdxl_lightning_8step_unet.safetensors", 8],
}
base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
ckpt = checkpoints["4-Step"][0] # Use the correct ckpt for your step setting!

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    gr.Warning("CUDA & MPS devices not found.")

print("Testing torch device")
torch.ones(2, device=device)

pipeline = None

def get_pipeline() -> StableDiffusionXLPipeline:
    unet = UNet2DConditionModel.from_config(base, subfolder='unet').to(device)
    ckpt_file = hf_hub_download(repo, ckpt)
    unet_state_dict = load_file(ckpt_file)
    unet.load_state_dict(unet_state_dict)
    pipeline = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant='fp16').to(device)
    pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config, timestep_spacing="trailing")

    return pipeline

def infer(
    prompt: str,
    negative_prompt: str,
    num_steps: int = 4,
):
    global pipeline
    if pipeline is None:
        pipeline = get_pipeline()
    output = pipeline(prompt, negative_prompt=negative_prompt, num_inference_steps=num_steps, guidance_scale=0)
    return output.images[0]


if __name__ == '__main__':
    with gr.Blocks() as interface:
        with gr.Column():
            output = gr.Image(label="SDXL output", type='pil')
            with gr.Row():
                prompt_area = gr.TextArea(label="Prompt")
                negative_prompt = gr.TextArea(label="Negative prompt", value="nsfw, bad quality, bad anatomy, worst quality, low quality, low resolutions, extra fingers, blur, "
                "blurry, ugly, wrongs proportions, watermark, image artifacts, lowres, ugly, jpeg artifacts, "
                "deformed, noisy image")
                model_dropdown = gr.Dropdown(label="Model choices", choices=checkpoints.keys(), value=checkpoints['4-Step'])
            btn = gr.Button("Generate! ⚡️")
        
        btn.click(fn=infer, inputs=[prompt_area, negative_prompt], outputs=[output])
    
    interface.queue().launch(share=False)
