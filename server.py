from flask import Flask, request, send_file
import os
from io import BytesIO

import json
import os
import random

import torch
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from PIL import Image
from pipelines.lora_controlnet_pipelines import SDXL_Pipeline
from pipelines.sr_pipeline import SR_Pipeline

base_path = "stabilityai/stable-diffusion-xl-base-1.0"
additional_controlnet_paths = [
     "diffusers/controlnet-depth-sdxl-1.0",
    "diffusers/controlnet-canny-sdxl-1.0",

]

additional_loras = [
    {
      "model_id":"prithivMLmods/Canes-Cars-Model-LoRA",
      "filename":"Canes-Cars-Model-LoRA.safetensors"
     }
]

pipe = SDXL_Pipeline(base_path, additional_controlnet_paths,additional_loras, use_refiner=True)
pipe.load_pipeline()

base_prompt = "cars, A car parked in a beach parking lot, the ocean horizon in the background. professinal photography --ar 16:9 --v 6.0 --style raw"

second_ref_base_prompt = base_prompt

negative_prompt = "poor quality, bad photography"

second_ref_negative_prompt = negative_prompt

lora_weights = [1.0]

controlnet_scale = [0.1,0.9]
control_guidance_start = [0.0,0.0]
control_guidance_end = [0.5,0.3]

cfg = 3.5
steps = 10

refine_1_steps = 20
refine_1_str = 0.15

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/upload', methods=['POST'])
def upload_and_respond():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    # Save the file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)


    seed = random.randint(0,10000)

    base_image, params, control_imgs = pipe.generate_img(
      prompt = base_prompt,
      negative_prompt = negative_prompt,
      controlnet_image_path = file_path,
      controlnet_scale = controlnet_scale,
      control_guidance_start = control_guidance_start,
      control_guidance_end = control_guidance_end,
      lora_weights = lora_weights,
      cfg = cfg,
      seed = seed,
      steps = steps,
      width = 1920,
      height = 1080,
    )

    base_image.save(file_path)

    # refined_image = pipe.refine_image(
    #   prompt = base_prompt,
    #   negative_prompt = negative_prompt,
    #   image=base_image,
    #   cfg=cfg,
    #   steps=refine_1_steps,
    #   strength=refine_1_str,
    #   seed = seed
    # )

    # refined_image.save(file_path)


    # Read the file back into memory to send it as a response
    with open(file_path, 'rb') as f:
        img_data = BytesIO(f.read())

    return send_file(img_data, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)