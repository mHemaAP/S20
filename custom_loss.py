''' Custom Loss Functions '''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import v2
from transformers import CLIPProcessor, CLIPVisionModelWithProjection, CLIPTextModelWithProjection

# Loss Function based on Edge Detection
def edge_detection(image):
    channels = image.shape[1]

    # Define the kernels for Edge Detection
    ed_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    ed_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Replicate the Edge detection kernels for each channel
    ed_x = ed_x.repeat(channels, 1, 1, 1).to(image.device)
    ed_y = ed_y.repeat(channels, 1, 1, 1).to(image.device)

    # ed_x = ed_x.to(torch.float16)
    # ed_y = ed_y.to(torch.float16)

    # Convolve the image with the Edge detection kernels
    conv_ed_x = F.conv2d(image, ed_x, padding=1, groups=channels)
    conv_ed_y = F.conv2d(image, ed_y, padding=1, groups=channels)

    # Combine the x and y gradients after convolution
    ed_value = torch.sqrt(conv_ed_x**2 + conv_ed_y**2)

    return ed_value

def edge_loss(image):
    ed_value = edge_detection(image)
    ed_capped = (ed_value > 0.5).to(torch.float32)
    return F.mse_loss(ed_value, ed_capped)

def compute_loss(original_image, loss_type, device, 
                 model="openai/clip-vit-large-patch14", 
                 prompt = "mountain background"):

    if loss_type == 'blue':
        # blue loss
        # [:,2] -> all images in batch, only the blue channel
        error = torch.abs(original_image[:,2] - 0.9).mean()
    elif loss_type == 'clip':
        # CLIP loss  
        error = cosine_loss(original_image, prompt, model, device)
    elif loss_type == 'edge':
        # edge loss
        error = edge_loss(original_image)        
    elif loss_type == 'contrast':
        # RGB to Gray loss
        transformed_image = T.functional.adjust_contrast(original_image, contrast_factor = 2)
        error = torch.abs(transformed_image - original_image).mean()
    elif loss_type == 'brightness':
        # brightnesss loss
        transformed_image = T.functional.adjust_brightness(original_image, brightness_factor = 2)
        error = torch.abs(transformed_image - original_image).mean()
    elif loss_type == 'sharpness':
        # sharpness loss
        transformed_image = T.functional.adjust_sharpness(original_image, sharpness_factor = 2)
        error = torch.abs(transformed_image - original_image).mean()
    elif loss_type == 'elastic':
        # elastic loss
        elastic_transformer = T.ElasticTransform(alpha=550.0,sigma=5.0)
        transformed_image = elastic_transformer(original_image)
        error = torch.abs(transformed_image - original_image).mean()
    elif loss_type == 'symmetry':
        transformed_image = torch.flip(original_image, [3])
        error = F.mse_loss(original_image, transformed_image)
    elif loss_type == 'saturation':
        # saturation loss
        transformed_image = T.functional.adjust_saturation(original_image, saturation_factor = 10)
        error = torch.abs(transformed_image - original_image).mean()
    else:
        print("error. Loss not defined")

    return error


# # additional textual prompt (on a mountain) (mountain background)
def get_text_embed(prompt, model, device):
    processor = CLIPProcessor.from_pretrained(model)
    inputs = processor(text=prompt,
                       return_tensors="pt",
                       padding=True)
    with torch.no_grad():
        text_embed = CLIPTextModelWithProjection.from_pretrained(
            model)(**inputs).text_embeds.to(device)
    return text_embed


def cosine_loss(gen_image, prompt, model, device):
    text_embed = get_text_embed(prompt, model, device)
    vision_encoder = CLIPVisionModelWithProjection.from_pretrained(model).to(device)
    gen_image_clamped = gen_image.clamp(0, 1).mul(255)
    resized_image = v2.Resize(224)(gen_image_clamped)
    image_embed = vision_encoder(resized_image).image_embeds
    similarity = F.cosine_similarity(text_embed, image_embed, dim=1)
    loss = 1 - similarity.mean()
    return loss
