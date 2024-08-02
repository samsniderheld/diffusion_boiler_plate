"""
lora_controlnet_pipelines.py

This module contains the implementation of the SDXL_Pipeline class, which is responsible for generating images using the Stable Diffusion XL ControlNet pipeline.

The SDXL_Pipeline class provides methods to load the pipeline, generate images based on given prompts and control parameters, and print the parameters used for image generation.

The module also includes other necessary imports and helper classes used by the SDXL_Pipeline class.
"""
import sys
sys.path.append('..')

import gc
import json
import random
import time
import torch
import transformers

from accelerate import PartialState
from compel import Compel, ReturnedEmbeddingsType
from controlnet_aux.processor import Processor
from diffusers import AutoPipelineForImage2Image, DPMSolverMultistepScheduler
from diffusers.models import ControlNetModel, AutoencoderKL
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image
from pipelines.controlnets import sd15_preprocessors, sdxl_preprocessors

class SDXL_Pipeline():
    """class to house all the pipeline loading and generation functions for easy looping and iteration"""
    def __init__(self, base_pipeline_path, additional_controlnet_paths=None, use_refiner=False,use_ip_adapter=False, use_distributed=False):
        self.base_pipeline_path = base_pipeline_path
        self.additional_controlnet_paths = additional_controlnet_paths
        self.controlnet_preprocessors = sdxl_preprocessors
        self.use_ip_adapter = use_ip_adapter
        self.use_refiner = use_refiner
        self.use_distributed = use_distributed
        self.controlnets = []

    def load_pipeline(self):
        """loads all the necessary components for the pipeline to function"""
        if(self.additional_controlnet_paths):
            for path in self.additional_controlnet_paths:
                controlnet = ControlNetModel.from_pretrained(path, torch_dtype=torch.float16)
                self.controlnets.append(controlnet)

            
        #load pipeline
        self.pipeline = AutoPipelineForImage2Image.from_pretrained(
            self.base_pipeline_path, 
            controlnet=self.controlnets, 
            torch_dtype=torch.float16,
            # vae=vae,
        )

        self.pipeline.scheduler =  DPMSolverMultistepScheduler.from_config(
            self.pipeline.scheduler.config, 
            use_karras=True, 
            euler_at_final=True,
            rescale_betas_zero_snr=True, 
            # vae=vae,
        )

        # self.pipeline.enable_vae_tiling()

        if self.use_distributed:
            self.pipeline.to("cuda:0")
        else:
            self.pipeline.to("cuda")

        if(self.use_ip_adapter):
           self.pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin") 

        self.compel = Compel(
            tokenizer=[self.pipeline.tokenizer, self.pipeline.tokenizer_2] ,
            text_encoder=[self.pipeline.text_encoder, self.pipeline.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
            truncate_long_prompts=True
        )

        if self.use_refiner:

            self.ref_pipeline = AutoPipelineForImage2Image.from_pretrained(
                "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
            )

            self.ref_pipeline.enable_vae_tiling()

            if self.use_distributed:
                self.ref_pipeline.to("cuda:1")
            else:
                self.ref_pipeline.to("cuda")


    def flush(self):
      gc.collect()
      torch.cuda.empty_cache()

    def generate_img(self, prompt,negative_prompt, controlnet_image_path, controlnet_scale, control_guidance_start, control_guidance_end, cfg, steps, seed=None, width=1024, height=1024,ip_adapter_weights=.6,clip_skip=0,strength=.15):
        """generates an image based on the given prompts and control parameters"""
        # Check that the sizes of the controlnets, images, and controlnet_scale match
        if len(self.controlnets) != len(controlnet_scale):
            raise ValueError("The sizes of the controlnets, images, and controlnet_scale must match")
        
        #deterministic seed
        if seed is None:
            seed = random.randint(0,10000)

        generator = torch.Generator(device='cuda').manual_seed(seed)
        controlnet_image = load_image(controlnet_image_path).resize((width,height))
        images = []

        # prepare controlnet images
        if(self.additional_controlnet_paths):
            for path in self.additional_controlnet_paths:
                processor_id = self.controlnet_preprocessors[path]
                processor = Processor(processor_id)
                processed_image = processor(controlnet_image, to_pil=True).resize((width,height))
                images.append(processed_image)


        conditioning, pooled = self.compel(prompt)

        negative_conditioning, negative_pooled = self.compel(negative_prompt)

        if(self.use_ip_adapter):
            self.pipeline.set_ip_adapter_scale(ip_adapter_weights)

        # generate image
        start_time = time.time()

        if(self.use_ip_adapter):
            image = self.pipeline(
                prompt_embeds=conditioning,
                pooled_prompt_embeds=pooled,
                negative_prompt_embeds=negative_conditioning,
                negative_pooled_prompt_embeds=negative_pooled,
                num_inference_steps=steps,
                image=controlnet_image,
                control_image=images, 
                ip_adapter_image=controlnet_image,
                generator=generator,
                content_guidance_scale=cfg,
                controlnet_conditioning_scale=controlnet_scale,
                control_guidance_start = control_guidance_start,
                control_guidance_end = control_guidance_end,
                width=width,
                height=height,
                clip_skip=clip_skip,
                strength=strength,
            ).images[0]
        
        else:
            image = self.pipeline(
                prompt_embeds=conditioning,
                pooled_prompt_embeds=pooled,
                negative_prompt_embeds=negative_conditioning,
                negative_pooled_prompt_embeds=negative_pooled,
                num_inference_steps=steps,
                image=controlnet_image,
                control_image=images, 
                generator=generator,
                content_guidance_scale=cfg,
                controlnet_conditioning_scale=controlnet_scale,
                control_guidance_start = control_guidance_start,
                control_guidance_end = control_guidance_end,
                width=width,
                height=height,
                clip_skip=clip_skip,
                strength=strength,
            ).images[0]

        end_time = time.time()

        elapsed_time = end_time - start_time

        print(f"The script took {elapsed_time:.2f} seconds to execute.")
        
        params = self.print_parameters(prompt, negative_prompt, controlnet_image_path, width, height, controlnet_scale, control_guidance_start, control_guidance_end, cfg, steps, seed=seed)

        image.info['comment'] = params

        return image, params, images
    
    
    def refine_image(self,prompt,negative_prompt, image, cfg, steps, strength=.15, seed=None,clip_skip=0):
        """refines an image using the refiner model"""
        if not self.use_refiner:
            raise ValueError("The refiner is not enabled")
        
        #deterministic seed
        if seed is None:
            seed = random.randint(0,10000)

        generator = torch.Generator(device='cuda').manual_seed(seed)

        start_time = time.time()

        image = self.ref_pipeline(
            prompt,
            negative_prompt=negative_prompt,
            image=image,
            strength=strength,
            generator=generator,
            content_guidance_scale=cfg,
            num_inference_steps=steps,
            clip_skip=clip_skip
        ).images[0]

        end_time = time.time()
        
        elapsed_time = end_time - start_time

        print(f"The script took {elapsed_time:.2f} seconds to execute.")

        return image


    def print_parameters(self, prompt, negative_prompt, controlnet_image_path, width, height, controlnet_scale, control_guidance_start, control_guidance_end, cfg, steps, seed=None):
        """a function to collate all parameters into a json string for easy saving and loading of parameters"""
        parameters = {
            "base_pipeline_path": self.base_pipeline_path,
            "controlnet_paths": self.additional_controlnet_paths,
            "input_image": controlnet_image_path,
            "width": width,
            "height": height,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "controlnet_scale": controlnet_scale,
            "control_guidance_start": control_guidance_start,
            "control_guidance_end": control_guidance_end,
            "cfg": cfg,
            "steps": steps,
            "seed": seed,
            "refiner_str":.15
        }

        return json.dumps(parameters, indent=4)
