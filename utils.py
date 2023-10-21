''' This file consists of definitions of Helper Functions '''
import gc
import PIL
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def image_grid2(imgs, labels, rows, cols):
    assert len(imgs) == len(labels) == rows * cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h), color='white')
    draw = ImageDraw.Draw(grid)
    font = ImageFont.truetype("arial.ttf", 11)  # You can specify the font file and font size here
    
    for i, (img, label) in enumerate(zip(imgs, labels)):
        x, y = i % cols * w, i // cols * h
        grid.paste(img, box=(x, y))

        # Add label text below the image
        text_width, text_height = draw.textsize(label, font=font)
        text_x = x + (w - text_width) // 2
        text_y = y + h
        draw.text((text_x, text_y), label, fill="black", font=font)
    
    return grid


def generate_pil_images(img):
    image = (img / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images

def cache_cleanup():
    # Clear the CUDA cache
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()    