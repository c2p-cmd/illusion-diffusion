# Illusion Diffusion

**Illusion Diffusion** is a project exploring AI image generation with a focus on creative effects and artistic control. This repository provides basic tools and instructions for generating unique images using Stable Diffusion models and the Metal Performance Shaders (MPS) on Apple devices.

## Installation

1. **PyTorch with MPS:**
    - Install Xcode Command Line Tools: `xcode-select --install`
    - Install PyTorch (nightly): `pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu`
    - Install dependencies: `pip3 install -r requirements.txt`
2. **Optional GPU Drivers:** Install latest drivers for optimal performance.

## Usage

1. **Run the script:** See script documentation for commands and configurations.
2. **Provide a prompt:** Describe the desired image (e.g., "a dreamlike landscape").
3. **Optional:** Use a control image for additional guidance.
4. **Generate image:** The script will create and save an image based on your prompt and settings.

## Models

* **QR Monster:** This model specializes in incorporating images within generated artworks.
* **Illusion Diffusion Pattern:** This model utilizes patterns as control elements for creative effects.

## Resources

* QR Monster: [https://huggingface.co/guumaster/blot-monster-diffusion](https://huggingface.co/guumaster/blot-monster-diffusion)
* Illusion Diffusion Pattern: [https://huggingface.co/spaces/hysts/LoRA-SD-training](https://huggingface.co/spaces/hysts/LoRA-SD-training)
* PyTorch Metal Installation: [https://developer.apple.com/metal/pytorch/](https://developer.apple.com/metal/pytorch/)
