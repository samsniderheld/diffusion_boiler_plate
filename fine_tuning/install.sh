#!/bin/bash

# Create a virtual environment
python3 -m venv hotd

# Activate the virtual environment
source hotd/bin/activate

pip install bitsandbytes transformers accelerate peft gradio datasets -q
pip install git+https://github.com/huggingface/diffusers.git -q

wget https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/train_dreambooth_lora_sdxl.py

