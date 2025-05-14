"""
Sprite Generator - Core functionality for generating sprites from AI models.

This module provides functions to generate sprite images using Hugging Face models,
remove backgrounds, and resize/pixelize the resulting images.
"""

import os
import io
from pathlib import Path
from typing import Optional, Tuple, Union, List

import numpy as np
from PIL import Image
import torch
from diffusers import DiffusionPipeline, StableDiffusionPipeline, StableDiffusionXLPipeline
from rembg import remove

# Set Hugging Face cache directory if specified
def set_hf_cache_dir(cache_dir: Optional[str] = None):
    """Set the Hugging Face cache directory."""
    if cache_dir:
        os.environ["HF_HOME"] = cache_dir
        print(f"Hugging Face cache directory set to: {cache_dir}")
    elif "HF_HOME" in os.environ:
        print(f"Using existing Hugging Face cache directory: {os.environ['HF_HOME']}")
    else:
        print("No Hugging Face cache directory specified. Using default location.")

# Get default cache directory from environment if available
HF_CACHE_DIR = os.environ.get("HF_HOME", None)

# Available models
AVAILABLE_MODELS = {
    "sd3.5": "stabilityai/stable-diffusion-3-medium-diffusers",
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "saana": "segmind/SaanaV2",
}

def load_model(model_name: str) -> DiffusionPipeline:
    """
    Load a diffusion model from Hugging Face.
    
    Args:
        model_name: Name of the model to load (sd3.5, sdxl, saana)
        
    Returns:
        Loaded diffusion pipeline
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available models: {', '.join(AVAILABLE_MODELS.keys())}")
    
    model_id = AVAILABLE_MODELS[model_name]
    
    # Determine which pipeline class to use based on the model
    if model_name == "sdxl":
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            use_safetensors=True,
            variant="fp16"
        )
    else:
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        )
    
    # Move to GPU if available
    if torch.cuda.is_available():
        pipeline = pipeline.to("cuda")
    
    return pipeline

def generate_image(
    prompt: str,
    model_name: str = "sd3.5",
    negative_prompt: str = "ugly, blurry, low quality, distorted, deformed",
    width: int = 512,
    height: int = 512,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None
) -> Image.Image:
    """
    Generate an image using a diffusion model.
    
    Args:
        prompt: Text prompt for image generation
        model_name: Name of the model to use (sd3.5, sdxl, saana)
        negative_prompt: Negative prompt to guide generation away from certain attributes
        width: Width of the generated image
        height: Height of the generated image
        num_inference_steps: Number of denoising steps
        guidance_scale: Scale for classifier-free guidance
        seed: Random seed for reproducibility
        
    Returns:
        Generated PIL Image
    """
    # Load the model
    pipeline = load_model(model_name)
    
    # Set the seed if provided
    generator = None
    if seed is not None:
        generator = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
    
    # Generate the image
    result = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator
    )
    
    # Return the first image
    return result.images[0]

def remove_background(image: Image.Image) -> Image.Image:
    """
    Remove the background from an image and convert it to RGBA.
    The background will be transparent (alpha channel).
    
    Args:
        image: Input PIL Image
        
    Returns:
        PIL Image with transparent background
    """
    # Use rembg to remove the background
    return remove(image)

def resize_pixelize(image: Image.Image, sprite_size: Tuple[int, int]) -> Image.Image:
    """
    Resize and pixelize an image to create a sprite effect.
    
    Args:
        image: Input PIL Image
        sprite_size: Target size as (width, height)
        
    Returns:
        Resized and pixelized PIL Image
    """
    # Resize the image to the target sprite size
    # Use nearest neighbor for pixelated effect
    return image.resize(sprite_size, Image.NEAREST)

def generate_sprite(
    prompt: str,
    output_path: Optional[str] = None,
    model_name: str = "sd3.5",
    sprite_size: Tuple[int, int] = (64, 64),
    negative_prompt: str = "ugly, blurry, low quality, distorted, deformed",
    gen_width: int = 1024,
    gen_height: int = 1024,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None,
    hf_cache_dir: Optional[str] = None,
) -> Image.Image:
    """
    Generate a sprite image using an AI model, remove the background, and resize/pixelize.
    
    Args:
        prompt: Text prompt for image generation
        output_path: Path to save the generated sprite (optional)
        model_name: Name of the model to use (sd3.5, sdxl, saana)
        sprite_size: Target size of the sprite as (width, height)
        negative_prompt: Negative prompt to guide generation away from certain attributes
        gen_width: Width of the generated image before resizing
        gen_height: Height of the generated image before resizing
        num_inference_steps: Number of denoising steps
        guidance_scale: Scale for classifier-free guidance
        seed: Random seed for reproducibility
        
    Returns:
        Generated sprite as a PIL Image
    """
    # Set Hugging Face cache directory if provided
    if hf_cache_dir:
        set_hf_cache_dir(hf_cache_dir)
    elif HF_CACHE_DIR:
        set_hf_cache_dir(HF_CACHE_DIR)
        
    # Generate the base image
    print(f"Generating image with {model_name} using prompt: '{prompt}'")
    image = generate_image(
        prompt=prompt,
        model_name=model_name,
        negative_prompt=negative_prompt,
        width=gen_width,
        height=gen_height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed
    )
    
    # Remove the background
    print("Removing background...")
    image_no_bg = remove_background(image)
    
    # Resize and pixelize
    print(f"Resizing to sprite size: {sprite_size}...")
    sprite = resize_pixelize(image_no_bg, sprite_size)
    
    # Save the sprite if an output path is provided
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        sprite.save(output_path, format="PNG")
        print(f"Sprite saved to: {output_path}")
    
    return sprite
