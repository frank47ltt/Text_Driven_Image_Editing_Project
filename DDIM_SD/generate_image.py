import torch
import requests
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from matplotlib import pyplot as plt
from torchvision import transforms as tfms
from diffusers import StableDiffusionPipeline, DDIMScheduler

def load_image(file_path, size=None):
    """
    Load an image from a given file path and optionally resize it.
    Args:
        file_path (str): Path to the image file.
        size (tuple, optional): Desired size as (width, height). Default is None.
    Returns:
        PIL.Image.Image: The loaded and optionally resized image.
    """
    img = Image.open(file_path).convert('RGB')
    if size is not None:
        img = img.resize(size)
    return img

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize pipeline
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

@torch.no_grad()
def sample(prompt, start_step=0, start_latents=None, guidance_scale=3.5, 
           num_inference_steps=30, num_images_per_prompt=1, 
           do_classifier_free_guidance=True, negative_prompt='', device=device):
    
    # Encode prompt
    text_embeddings = pipe._encode_prompt(
        prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
    )
    
    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    
    # Create random starting point if none provided
    if start_latents is None:
        start_latents = torch.randn(1, 4, 64, 64, device=device)
        start_latents *= pipe.scheduler.init_noise_sigma
        
    latents = start_latents.clone()
    
    for i in tqdm(range(start_step, num_inference_steps)):
        t = pipe.scheduler.timesteps[i]
        
        # Expand latents for classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        
        # Predict noise residual
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        
        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # Manual update step
        prev_t = max(1, t.item() - (1000//num_inference_steps))
        alpha_t = pipe.scheduler.alphas_cumprod[t.item()]
        alpha_t_prev = pipe.scheduler.alphas_cumprod[prev_t]
        predicted_x0 = (latents - (1-alpha_t).sqrt()*noise_pred) / alpha_t.sqrt()
        direction_pointing_to_xt = (1-alpha_t_prev).sqrt()*noise_pred
        latents = alpha_t_prev.sqrt()*predicted_x0 + direction_pointing_to_xt
    
    # Post-processing
    images = pipe.decode_latents(latents)
    images = pipe.numpy_to_pil(images)
    
    return images

@torch.no_grad()
def invert(start_latents, prompt, guidance_scale=3.5, num_inference_steps=80,
           num_images_per_prompt=1, do_classifier_free_guidance=True,
           negative_prompt='', device=device):
    
    # Encode prompt
    text_embeddings = pipe._encode_prompt(
        prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
    )
    
    latents = start_latents.clone()
    intermediate_latents = []
    
    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    
    # Reversed timesteps
    timesteps = reversed(pipe.scheduler.timesteps)
    
    for i in tqdm(range(1, num_inference_steps), total=num_inference_steps-1):
        if i >= num_inference_steps - 1:
            continue
            
        t = timesteps[i]
        
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        current_t = max(0, t.item() - (1000//num_inference_steps))
        next_t = t
        alpha_t = pipe.scheduler.alphas_cumprod[current_t]
        alpha_t_next = pipe.scheduler.alphas_cumprod[next_t]
        
        latents = (latents - (1-alpha_t).sqrt()*noise_pred)*(alpha_t_next.sqrt()/alpha_t.sqrt()) + (1-alpha_t_next).sqrt()*noise_pred
        intermediate_latents.append(latents)
    
    return torch.cat(intermediate_latents)

def edit(input_image, input_image_prompt, edit_prompt, num_steps=100, start_step=30, guidance_scale=3.5):
    with torch.no_grad():
        latent = pipe.vae.encode(tfms.functional.to_tensor(input_image).unsqueeze(0).to(device)*2-1)
    l = 0.18215 * latent.latent_dist.sample()
    inverted_latents = invert(l, input_image_prompt, num_inference_steps=num_steps)
    final_im = sample(edit_prompt, start_latents=inverted_latents[-(start_step+1)][None],
                     start_step=start_step, num_inference_steps=num_steps, guidance_scale=guidance_scale)[0]
    return final_im
