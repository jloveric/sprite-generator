"""
Command-line interface for the Sprite Generator.

This module provides a Click-based CLI for the sprite generator functionality.
"""

import os
import random
from typing import Optional, Tuple

import click
from PIL import Image

from sprite_generator.sprite_gen import generate_sprite, AVAILABLE_MODELS

@click.group()
def cli():
    """Sprite Generator - Create 2D sprites using AI image generation models."""
    pass

@cli.command()
@click.option(
    "--prompt", "-p", 
    required=True, 
    help="Text prompt for image generation"
)
@click.option(
    "--output", "-o", 
    default="sprite.png", 
    help="Output path for the generated sprite"
)
@click.option(
    "--model", "-m",
    type=click.Choice(list(AVAILABLE_MODELS.keys())),
    default="sd3.5",
    help="Model to use for image generation"
)
@click.option(
    "--sprite-width", 
    type=int, 
    default=128, 
    help="Width of the output sprite"
)
@click.option(
    "--sprite-height", 
    type=int, 
    default=128, 
    help="Height of the output sprite"
)
@click.option(
    "--negative-prompt", 
    default="ugly, blurry, low quality, distorted, deformed",
    help="Negative prompt to guide generation away from certain attributes"
)
@click.option(
    "--gen-width", 
    type=int, 
    default=1024, 
    help="Width of the generated image before resizing"
)
@click.option(
    "--gen-height", 
    type=int, 
    default=1024, 
    help="Height of the generated image before resizing"
)
@click.option(
    "--steps", 
    type=int, 
    default=30, 
    help="Number of denoising steps"
)
@click.option(
    "--guidance-scale", 
    type=float, 
    default=7.5, 
    help="Scale for classifier-free guidance"
)
@click.option(
    "--seed", 
    type=int, 
    help="Random seed for reproducibility (optional)"
)
@click.option(
    "--batch-size",
    type=int,
    default=1,
    help="Number of sprites to generate"
)
@click.option(
    "--hf-cache-dir",
    type=str,
    help="Hugging Face cache directory (optional, defaults to HF_HOME environment variable)"
)
def generate(
    prompt: str,
    output: str,
    model: str = "sd3.5",
    sprite_width: int = 128,
    sprite_height: int = 128,
    negative_prompt: str = "ugly, blurry, low quality, distorted, deformed",
    gen_width: int = 1024,
    gen_height: int = 1024,
    steps: int = 30,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None,
    batch_size: int = 1,
    hf_cache_dir: Optional[str] = None,
):
    """Generate a sprite using AI image generation."""
    # Use random seed if not provided
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
        click.echo(f"Using random seed: {seed}")
    
    # Generate the sprite(s)
    sprites = generate_sprite(
        prompt=prompt,
        output_path=output,
        model_name=model,
        sprite_size=(sprite_width, sprite_height),
        negative_prompt=negative_prompt,
        gen_width=gen_width,
        gen_height=gen_height,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
        hf_cache_dir=hf_cache_dir,
        batch_size=batch_size
    )
    
    if batch_size > 1:
        click.echo(f"{batch_size} sprites generated!")
    else:
        click.echo(f"Sprite generation complete! Saved to: {output}")
    
    return sprites

@cli.command()
def list_models():
    """List available models for sprite generation."""
    click.echo("Available models:")
    for name, model_id in AVAILABLE_MODELS.items():
        click.echo(f"  - {name}: {model_id}")

if __name__ == "__main__":
    cli()
