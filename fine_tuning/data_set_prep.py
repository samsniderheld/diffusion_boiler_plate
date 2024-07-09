"""
data_set_prep.py:

This module provides a function to resize and rename images in a directory.

Functions:
    resize_and_rename_images(input_directory, output_directory): Resizes and renames images in the input directory and saves them in the output directory.

The script takes two command-line arguments: 'input_directory' and 'output_directory'. 'input_directory' is the directory containing the images to be resized and renamed. 'output_directory' is the directory where the resized and renamed images will be saved.

The resize_and_rename_images function goes through each image in the input directory, resizes it, renames it, and saves it in the output directory. The new name of each image is a four-digit number that represents its order in the sorted list of filenames in the input directory.
"""
import os
import argparse
from PIL import Image,ImageOps

def resize_and_rename_images(input_directory, output_directory):
    i = 1
    # Check if output_directory exists, create it if it doesn't
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    files = sorted(os.listdir(input_directory))

    for filename in files:
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = Image.open(os.path.join(input_directory, filename))
            img = ImageOps.exif_transpose(img)
            # img = img.resize((512, 1024),Image.BILINEAR)
            new_filename = str(i).zfill(4) + '.' + filename.split('.')[-1]
            img.save(os.path.join(output_directory, new_filename))
            i += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Resize and rename images.')
    parser.add_argument('--input_directory', type=str, help='Input directory containing the images.')
    parser.add_argument('--output_directory', type=str, help='Output directory to save the resized and renamed images.')
    args = parser.parse_args()

    resize_and_rename_images(args.input_directory, args.output_directory)