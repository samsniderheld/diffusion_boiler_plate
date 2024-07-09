"""
utils.py: 

This module provides utility functions for image processing.

Functions:
    draw_text(draw, text, position, font, max_width): Draws text on an image with word wrapping.
    add_text_to_image(image, text, font_path, font_size, max_width): Adds text to an image.
    ensure_PIL_rgba(image): Ensures that an image is a PIL Image in RGBA format.
    ensure_numpy(image): Ensures that an image is a numpy array.
    load_params_from_image(image): Loads parameters from an image.

Dependencies:
    numpy: Library used for numerical operations.
    PIL: Python Imaging Library used for image operations.

"""
import json
import numpy as np
from PIL import ImageDraw, ImageFont, Image

def draw_text(draw, text, position, font, max_width):
    """Draw the text on the image with word wrapping."""
    lines = []
    words = text.split()
    while words:
        line = ''
        while words and font.getsize(line + words[0])[0] <= max_width:
            line += (words.pop(0) + ' ')
        lines.append(line)

    # Draw each line of text
    y = position[1]
    for line in lines:
        draw.text((position[0], y), line, font=font, fill=(0,0,0))
        y += font.getsize(line)[1]

def add_text_to_image(image, text, font_path='OpenSans-Regular.ttf', font_size=50, max_width=400):
    """Wrapper function for the draw_text_function"""
    # Load the image
    draw = ImageDraw.Draw(image)

    # Load the font
    font = ImageFont.truetype(font_path, font_size)

    # Starting position for the text
    position = (50, 50)

    # Draw the text
    draw_text(draw, text, position, font, max_width)

def ensure_PIL_rgba(image):
    """Ensure that the input image is a PIL Image in RGBA format."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    if isinstance(image, Image.Image):
        if image.mode != 'RGBA':
            return image.convert('RGBA')
        else:
            return image
    else:
        raise ValueError("Input is not a PIL Image")
    

def ensure_numpy(image):
    """Ensure that the input image is a numpy array."""
    if isinstance(image, Image.Image):
        return np.array(image)
    elif isinstance(image, np.ndarray):
        return image
    else:
        raise ValueError("Input is neither a PIL Image nor a numpy array")
    
def load_params_from_image(base_img_path):
    """Loads parameters from an image."""
    base_image = Image.open(base_img_path)
    base_image_params = json.loads(base_image.info["comment"])
    return base_image_params
