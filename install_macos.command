# !/bin/bash

# Installing Jupyter
pip3 install notebook

# Installing libraries
pip3 install accelerate transformers diffusers pillow

# Installing Gradio for Interface
pip3 install gradio

# Installing PyTorch
echo "Installing PyTorch"
pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu

# Ask for Tensorflow
echo "Do you wish to install tensorflow?"
select yn in "Yes" "No"; do
    case $yn in
        Yes ) echo "Installing Tensorflow"; pip3 install tensorflow tensorflow-metal; break;;
        No ) echo "Not installing Tensorflow"; break;;
    esac
done