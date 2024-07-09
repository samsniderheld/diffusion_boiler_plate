"""
sr_pipeline.py:

This module provides a class for super-resolution (SR) tasks using the RealESRGAN model.

Dependencies:
    gc: The Python garbage collector for memory management.
    torch: The PyTorch library for machine learning tasks.
    RealESRGAN: The RealESRGAN model for super-resolution tasks.

The SR_Pipeline class provides methods to load the necessary components for the pipeline, free up memory, and upscale images.
"""
import gc
import torch
from RealESRGAN import RealESRGAN

class SR_Pipeline():
    """class to house the super resolution pipeline"""
    def __init__(self, scale=2,use_distributed=False):
        self.scale = scale
        self.use_distributed = use_distributed

    def load_pipeline(self):
        """loads all the necessary components for the pipeline to function"""

        if self.use_distributed:
            device = torch.device('cuda:2')
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sr_model = RealESRGAN(device, scale=self.scale)
        self.sr_model.load_weights(f'weights/RealESRGAN_x{self.scale}.pth', download=True)

    def flush(self):
        """flushes the memory and cache"""
        gc.collect()
        torch.cuda.empty_cache()

    def upscale(self, image):
        """upscales the image using the super resolution model"""
        image = self.sr_model.predict(image)
        self.flush()
        return image

#maybe implement different model choices
#https://github.com/csslc/CCSR