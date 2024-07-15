"""
lora_controlnet_pipelines.py

This module contains the implementation of the SDXL_Pipeline class, which is responsible for generating images using the Stable Diffusion XL ControlNet pipeline.

The SDXL_Pipeline class provides methods to load the pipeline, generate images based on given prompts and control parameters, and print the parameters used for image generation.

The module also includes other necessary imports and helper classes used by the SDXL_Pipeline class.
"""
import gc
import json
import random
import time
import torch
import transformers

from compel import Compel, ReturnedEmbeddingsType
from controlnet_aux.processor import Processor
from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLControlNetInpaintPipeline, StableDiffusionControlNetInpaintPipeline,DEISMultistepScheduler
from diffusers.models import ControlNetModel, AutoencoderKL
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image
from RealESRGAN import RealESRGAN

class SDXL_Pipeline():
    """class to house all the pipeline loading and generation functions for easy looping and iteration"""
    def __init__(self, base_pipeline_path, additional_controlnet_paths=None, additional_loras=None, use_refiner=True, clip_skip=0, use_distributed=False):
        self.base_pipeline_path = base_pipeline_path
        self.additional_controlnet_paths = additional_controlnet_paths
        self.additional_loras = additional_loras
        self.controlnet_preprocessors = {

            "thibaud/controlnet-openpose-sdxl-1.0": "openpose_full",
            "diffusers/controlnet-depth-sdxl-1.0": "depth_midas",
            "diffusers/controlnet-canny-sdxl-1.0": "canny"
        }
        self.use_refiner = use_refiner
        self.clip_skip = clip_skip
        self.use_distributed = use_distributed
        self.controlnets = []

    def load_pipeline(self):
        """loads all the necessary components for the pipeline to function"""
        if(self.additional_controlnet_paths):
            for path in self.additional_controlnet_paths:
                controlnet = ControlNetModel.from_pretrained(path, torch_dtype=torch.float16)
                self.controlnets.append(controlnet)

        # vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

        if(self.clip_skip != 0): 
            # Load the CLIP text encoder from the stable diffusion 1.5 pipeline,
            # and specify the number of layers to use.
            text_encoder = transformers.CLIPTextModel.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                subfolder = "text_encoder",
                num_hidden_layers = 12 - (self.clip_skip - 1),
                torch_dtype = torch.float16
            )


            #load pipeline
            self.pipeline = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
                self.base_pipeline_path, 
                controlnet=self.controlnets, 
                torch_dtype=torch.float16,
                text_encoder=text_encoder
                # vae=vae,
            )

        else:
            #load pipeline
            self.pipeline = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
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

        if(self.additional_loras):
            for lora in self.additional_loras:
                lora_model_id = lora['model_id']
                lora_filename = lora['filename']
                self.pipeline.load_lora_weights(lora_model_id, weight_name=lora_filename, adapter_name=lora_model_id)

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

    def generate_img(self, prompt,negative_prompt, bg_image_path, mask_image_path,controlnet_image_path, controlnet_scale, control_guidance_start, control_guidance_end, lora_weights, cfg, steps, seed=None, width=1024, height=1024):
        """generates an image based on the given prompts and control parameters"""
        # Check that the sizes of the controlnets, images, and controlnet_scale match
        if len(self.controlnets) != len(controlnet_scale):
            raise ValueError("The sizes of the controlnets, images, and controlnet_scale must match")
        
        if(self.additional_loras):
          if len(self.additional_loras) != len(lora_weights):
              raise ValueError("The sizes of the additional_loras and lora_weights must match")

        #deterministic seed
        if seed is None:
            seed = random.randint(0,10000)

        generator = torch.Generator(device='cuda').manual_seed(seed)
        controlnet_image = load_image(controlnet_image_path)
        bg_image = load_image(bg_image_path).resize((width,height))
        mask_image = load_image(mask_image_path).resize((width,height))
        images = []

        # prepare controlnet images
        if(self.additional_controlnet_paths):
            for path in self.additional_controlnet_paths:
                processor_id = self.controlnet_preprocessors[path]
                processor = Processor(processor_id)
                processed_image = processor(controlnet_image, to_pil=True).resize((width,height))
                images.append(processed_image)

                
        if(self.additional_loras):
          if len(self.additional_loras) != len(lora_weights):
              raise ValueError("The sizes of the additional_loras and lora_weights must match")

        conditioning, pooled = self.compel(prompt)

        negative_conditioning, negative_pooled = self.compel(negative_prompt)

        # generate image
        start_time = time.time()

        image = self.pipeline(
            prompt_embeds=conditioning,
            pooled_prompt_embeds=pooled,
            negative_prompt_embeds=negative_conditioning,
            negative_pooled_prompt_embeds=negative_pooled,
            num_inference_steps=steps,
            image=bg_image,
            mask_image=mask_image,
            control_image=images,
            generator=generator,
            content_guidance_scale=cfg,
            controlnet_conditioning_scale=controlnet_scale,
            control_guidance_start = control_guidance_start,
            control_guidance_end = control_guidance_end,
            width=width,
            height=height
        ).images[0]

        end_time = time.time()

        elapsed_time = end_time - start_time

        print(f"The script took {elapsed_time:.2f} seconds to execute.")
        
        params = self.print_parameters(prompt, negative_prompt, controlnet_image_path, width, height, controlnet_scale, control_guidance_start, control_guidance_end, lora_weights, cfg, steps, seed=seed)

        image.info['comment'] = params

        return image, params, images
    
    
    def refine_image(self,prompt,negative_prompt, image, cfg, steps, strength=.15, seed=None):
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
        ).images[0]

        end_time = time.time()
        
        elapsed_time = end_time - start_time

        print(f"The script took {elapsed_time:.2f} seconds to execute.")

        return image


    def print_parameters(self, prompt, negative_prompt, controlnet_image_path, width, height, controlnet_scale, control_guidance_start, control_guidance_end, lora_weights, cfg, steps, seed=None):
        """a function to collate all parameters into a json string for easy saving and loading of parameters"""
        parameters = {
            "base_pipeline_path": self.base_pipeline_path,
            "controlnet_paths": self.additional_controlnet_paths,
            "loras": self.additional_loras,
            "input_image": controlnet_image_path,
            "width": width,
            "height": height,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "controlnet_scale": controlnet_scale,
            "control_guidance_start": control_guidance_start,
            "control_guidance_end": control_guidance_end,
            "lora_weights": lora_weights,
            "cfg": cfg,
            "steps": steps,
            "seed": seed,
            "refiner_str":.15
        }

        return json.dumps(parameters, indent=4)
    


class SD15_Pipeline():
    """class to house all the pipeline loading and generation functions for easy looping and iteration"""
    def __init__(self, base_pipeline_path, additional_controlnet_paths=None, additional_loras=None, use_refiner=True, clip_skip=0, use_distributed=False):
        self.base_pipeline_path = base_pipeline_path
        self.additional_controlnet_paths = additional_controlnet_paths
        self.additional_loras = additional_loras
        self.use_refiner = use_refiner
        self.controlnet_preprocessors = {

            "lllyasviel/sd-controlnet-depth": "depth_midas",
            "lllyasviel/sd-controlnet-canny": "canny"
        }
        self.clip_skip = clip_skip
        self.use_distributed = use_distributed
        self.controlnets = []

    def load_pipeline(self):
        """loads all the necessary components for the pipeline to function"""
        if(self.additional_controlnet_paths):
            for path in self.additional_controlnet_paths:
                controlnet = ControlNetModel.from_pretrained(path, torch_dtype=torch.float16)
                self.controlnets.append(controlnet)

        # vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        if(self.clip_skip != 0): 
            # Load the CLIP text encoder from the stable diffusion 1.5 pipeline,
            # and specify the number of layers to use.
            text_encoder = transformers.CLIPTextModel.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                subfolder = "text_encoder",
                num_hidden_layers = 12 - (self.clip_skip - 1),
                torch_dtype = torch.float16
            )


            #load pipeline
            self.pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                self.base_pipeline_path, 
                controlnet=self.controlnets, 
                torch_dtype=torch.float16,
                text_encoder=text_encoder
                # vae=vae,
            )
        else:

            #load pipeline
            self.pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                self.base_pipeline_path, 
                controlnet=self.controlnets, 
                torch_dtype=torch.float16,
                # vae=vae,
            )

        self.pipeline.scheduler =  DEISMultistepScheduler.from_config(self.pipeline.scheduler.config)

        # self.pipeline.enable_vae_tiling()

        if self.use_distributed:
            self.pipeline.to("cuda:0")
        else:
            self.pipeline.to("cuda")

        if(self.additional_loras):
            for lora in self.additional_loras:
                lora_model_id = lora['model_id']
                lora_filename = lora['filename']
                self.pipeline.load_lora_weights(lora_model_id, weight_name=lora_filename, adapter_name=lora_model_id)

        self.compel = Compel(
            tokenizer=self.pipeline.tokenizer ,
            text_encoder=self.pipeline.text_encoder,
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

    def generate_img(self, prompt,negative_prompt, bg_image_path, mask_image_path,controlnet_image_path, controlnet_scale, control_guidance_start, control_guidance_end, lora_weights, cfg, steps, seed=None, width=1024, height=1024):
        """generates an image based on the given prompts and control parameters"""
        # Check that the sizes of the controlnets, images, and controlnet_scale match
        if len(self.controlnets) != len(controlnet_scale):
            raise ValueError("The sizes of the controlnets, images, and controlnet_scale must match")
        
        if(self.additional_loras):
          if len(self.additional_loras) != len(lora_weights):
              raise ValueError("The sizes of the additional_loras and lora_weights must match")

        #deterministic seed
        if seed is None:
            seed = random.randint(0,10000)

        generator = torch.Generator(device='cuda').manual_seed(seed)
        controlnet_image = load_image(controlnet_image_path)
        bg_image = load_image(bg_image_path).resize((width,height))
        mask_image = load_image(mask_image_path).resize((width,height))
        images = []

        # prepare controlnet images
        if(self.additional_controlnet_paths):
            for path in self.additional_controlnet_paths:
                processor_id = self.controlnet_preprocessors[path]
                processor = Processor(processor_id)
                processed_image = processor(controlnet_image, to_pil=True).resize((width,height))
                images.append(processed_image)

                
        if(self.additional_loras):
          if len(self.additional_loras) != len(lora_weights):
              raise ValueError("The sizes of the additional_loras and lora_weights must match")

        conditioning = self.compel(prompt)

        negative_conditioning = self.compel(negative_prompt)

        # generate image
        start_time = time.time()

        image = self.pipeline(
            prompt_embeds=conditioning,
            negative_prompt_embeds=negative_conditioning,
            num_inference_steps=steps,
            image=bg_image,
            mask_image=mask_image,
            control_image=images,
            generator=generator,
            content_guidance_scale=cfg,
            controlnet_conditioning_scale=controlnet_scale,
            control_guidance_start = control_guidance_start,
            control_guidance_end = control_guidance_end,
            width=width,
            height=height
        ).images[0]

        end_time = time.time()

        elapsed_time = end_time - start_time

        print(f"The script took {elapsed_time:.2f} seconds to execute.")
        
        params = self.print_parameters(prompt, negative_prompt, controlnet_image_path, width, height, controlnet_scale, control_guidance_start, control_guidance_end, lora_weights, cfg, steps, seed=seed)

        image.info['comment'] = params

        return image, params, images
    
    
    def refine_image(self,prompt,negative_prompt, image, cfg, steps, strength=.15, seed=None):
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
        ).images[0]

        end_time = time.time()
        
        elapsed_time = end_time - start_time

        print(f"The script took {elapsed_time:.2f} seconds to execute.")

        return image


    def print_parameters(self, prompt, negative_prompt, controlnet_image_path, width, height, controlnet_scale, control_guidance_start, control_guidance_end, lora_weights, cfg, steps, seed=None):
        """a function to collate all parameters into a json string for easy saving and loading of parameters"""
        parameters = {
            "base_pipeline_path": self.base_pipeline_path,
            "controlnet_paths": self.additional_controlnet_paths,
            "loras": self.additional_loras,
            "input_image": controlnet_image_path,
            "width": width,
            "height": height,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "controlnet_scale": controlnet_scale,
            "control_guidance_start": control_guidance_start,
            "control_guidance_end": control_guidance_end,
            "lora_weights": lora_weights,
            "cfg": cfg,
            "steps": steps,
            "seed": seed,
            "refiner_str":.15
        }

        return json.dumps(parameters, indent=4)