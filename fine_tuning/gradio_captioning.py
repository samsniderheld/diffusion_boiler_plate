"""
gradio_captioning.py:

This module provides a Gradio interface for image captioning tasks.

Functions:
    write_caption_file(captions): Writes the generated captions to a file.
    caption_image(elements): Generates a caption for the next image based on the selected elements.
    get_next_image(): Returns the next image to be captioned.

The script takes two command-line arguments: 'elements_file' and 'image_directory'. 'elements_file' is a file that contains the elements to be used in the captioning. 'image_directory' is the directory containing the images to be captioned.

The Gradio interface consists of a checkbox group for selecting elements, a submit button, an image display, and a textbox for the generated caption. When the submit button is clicked, a caption is generated for the next image based on the selected elements, and the caption is displayed in the textbox.
"""

import argparse
import glob
import gradio as gr
from PIL import Image


parser = argparse.ArgumentParser(description='Resize and rename images.')
parser.add_argument('--elements_file', type=str, help='A file that contains the elements to be used in the captioning.')
parser.add_argument('--image_directory', type=str, help='The directory containing the images to be captioned.')
parser.add_argument('--base_caption', type=str, help='The base caption to be used for all images.')
parser.add_argument('--caption_file_name', type=str, help='The name of the file to write the captions to.')

args = parser.parse_args()


with open(args.elements_file, 'r') as file:
    elements = [line.strip() for line in file]

image_dir = args.image_directory
image_paths = f"{image_dir}/*.jpg"
all_images = sorted(glob.glob(image_paths))
images = [Image.open(path) for path in all_images]

captions = []
captions_file_path = args.caption_file_name
index = 0

def write_caption_file(captions):
    """Writes the generated captions to a file."""
    with open(captions_file_path, 'w') as new_file:
            for caption in captions:
                new_file.write(caption)

def caption_image(elements):
    """Generates a caption for the next image based on the selected elements."""
    if(len(elements)>0):
        caption = f"{args.base_caption}, {', '.join(elements) }, masterpiece costume, very fine details, hbo, 8k detail, hq\n"
        captions.append(caption)
        write_caption_file(captions)
    else:
        caption = None

    image = get_next_image()
    return [image,caption]

def get_next_image():
    """Returns the next image to be captioned."""
    global index
    image = images[index]
    index+=1
    return image

with gr.Blocks() as demo:

  with gr.Row():
    with gr.Column():

      check_boxes = gr.CheckboxGroup(elements, label="Elements", info="Elements in the image")
      submit_button = gr.Button("Submit")

    with gr.Column():
      image = gr.Image(label="Image to be captioned")
      output_text = gr.Textbox()

  submit_button.click(caption_image,inputs=check_boxes,outputs=[image, output_text])

if __name__ == "__main__":
    demo.launch(share=True,debug=True)