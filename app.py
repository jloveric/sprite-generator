"""
Streamlit app for the Sprite Generator.

This module provides a web interface for the sprite generator functionality.
"""

import os
import random
import datetime
import io
from typing import List, Optional, Tuple
from pathlib import Path

import streamlit as st
from PIL import Image
import numpy as np

from sprite_generator.sprite_gen import generate_sprite, AVAILABLE_MODELS

# Set page config
st.set_page_config(
    page_title="Sprite Generator",
    page_icon="🎮",
    layout="wide",
)

# Add custom CSS for styling
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 3rem !important;
        margin-bottom: 1rem;
    }
    .sub-title {
        text-align: center;
        font-size: 1.5rem !important;
        margin-bottom: 2rem;
        color: #888;
    }
    .stImage {
        margin: 0 auto;
        display: block;
    }
    .sprite-container {
        background-color: #f0f0f0;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        text-align: center;
    }
    .sprite-container img {
        image-rendering: pixelated;
        image-rendering: -moz-crisp-edges;
        image-rendering: crisp-edges;
        /* Prevent upscaling */
        transform-origin: top left;
        transform: scale(1);
        width: auto !important;
        height: auto !important;
        max-width: none !important;
    }
    .actual-size-container {
        display: inline-block;
        margin: 0 auto;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-title">🎮 Sprite Generator</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Create pixel art sprites using AI</p>', unsafe_allow_html=True)

# Create output directory for saved sprites
def get_output_dir():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

# Sidebar for parameters
with st.sidebar:
    st.header("Generation Parameters")
    
    # Model selection
    model = st.selectbox(
        "Select Model",
        options=list(AVAILABLE_MODELS.keys()),
        format_func=lambda x: f"{x} ({AVAILABLE_MODELS[x]})",
        index=0
    )
    
    # Number of images to generate
    num_images = st.slider("Number of Images", min_value=1, max_value=8, value=1, step=1)
    
    # Sprite size
    st.subheader("Sprite Size")
    sprite_width = st.number_input("Width", min_value=16, max_value=256, value=128, step=16)
    sprite_height = st.number_input("Height", min_value=16, max_value=256, value=128, step=16)
    
    # Save options
    st.subheader("Save Options")
    save_sprites = st.checkbox("Save generated sprites", value=True)
    
    # Advanced options (collapsible)
    with st.expander("Advanced Options"):
        # Generation resolution
        st.subheader("Generation Resolution")
        gen_width = st.number_input("Gen Width", min_value=512, max_value=2048, value=1024, step=128)
        gen_height = st.number_input("Gen Height", min_value=512, max_value=2048, value=1024, step=128)
        
        # Generation parameters
        steps = st.slider("Inference Steps", min_value=10, max_value=50, value=30, step=5)
        guidance_scale = st.slider("Guidance Scale", min_value=1.0, max_value=15.0, value=7.5, step=0.5)
        
        # Negative prompt
        negative_prompt = st.text_area(
            "Negative Prompt", 
            value="ugly, blurry, low quality, distorted, deformed",
            height=100
        )
        
        # Random seed
        use_random_seed = st.checkbox("Use Random Seed", value=True)
        seed = None
        if not use_random_seed:
            seed = st.number_input("Seed", min_value=0, max_value=2**32-1, value=42, step=1)

# Main content area
st.header("Create Your Sprites")

# Prompt input
prompt = st.text_area("Enter your prompt", value="An alien creature", height=100)

# Generate button
if st.button("Generate Sprites", type="primary", use_container_width=True):
    if not prompt:
        st.error("Please enter a prompt")
    else:
        # Create a timestamped directory for this generation session
        if save_sprites:
            output_dir = get_output_dir()
            st.info(f"Sprites will be saved to: {output_dir}")
        
        # Create a container to display all sprites
        sprite_gallery = st.container()
        
        # Create a progress bar for the overall generation
        progress_bar = st.progress(0)
        
        # Create columns for the sprites based on the number of images
        with sprite_gallery:
            # Use 4 columns max to ensure sprites are visible
            num_cols = min(4, num_images)
            cols = st.columns(num_cols)
            
            # Status message
            status_container = st.empty()
            status_container.info(f"Generating {num_images} sprite{'s' if num_images > 1 else ''}...")
            
            # Store all generated sprites for display
            all_sprites = []
            all_seeds = []
            
            # Set a single seed for batch generation if not using random seeds
            current_seed = None
            if not use_random_seed and seed is not None:
                current_seed = seed
            elif use_random_seed:
                current_seed = random.randint(0, 2**32 - 1)
                
            # Record the seed
            all_seeds.append(current_seed)
                
            # Generate sprites in a single batch
            status_container.info(f"Generating {num_images} sprite{'s' if num_images > 1 else ''} in batch...")
            progress_bar.progress(0.1)  # Show some initial progress
            
            # Set output path for batch
            output_path = None
            if save_sprites and num_images == 1:
                output_path = str(output_dir / f"sprite_seed_{current_seed}.png")
            elif save_sprites:
                output_path = str(output_dir / f"sprite.png")  # Base name, will be modified for batch
                
            # Generate all sprites in a single batch call
            sprites = generate_sprite(
                prompt=prompt,
                output_path=output_path,
                model_name=model,
                sprite_size=(sprite_width, sprite_height),
                negative_prompt=negative_prompt,
                gen_width=gen_width,
                gen_height=gen_height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                seed=current_seed,
                batch_size=num_images
            )
            
            progress_bar.progress(0.8)  # Update progress after generation
            
            # Convert to list if single sprite was returned
            if num_images == 1:
                all_sprites = [sprites]  # Convert single sprite to list
                # For display purposes, we need the seed for each sprite
                all_seeds = [current_seed] * num_images
            else:
                all_sprites = sprites
                # For display purposes, we need the seed for each sprite
                # Since we used a single seed for the batch, we'll show the same seed for all
                all_seeds = [current_seed] * num_images
                
            progress_bar.progress(1.0)  # Complete the progress
            
            # Display all sprites after generation
            status_container.success(f"Generated {num_images} sprite{'s' if num_images > 1 else ''}!")
            
            # Save generation parameters to a text file
            if save_sprites:
                params_file = output_dir / "generation_params.txt"
                with open(params_file, "w") as f:
                    f.write(f"Prompt: {prompt}\n")
                    f.write(f"Model: {model}\n")
                    f.write(f"Negative Prompt: {negative_prompt}\n")
                    f.write(f"Sprite Size: {sprite_width}x{sprite_height}\n")
                    f.write(f"Generation Size: {gen_width}x{gen_height}\n")
                    f.write(f"Steps: {steps}\n")
                    f.write(f"Guidance Scale: {guidance_scale}\n")
                    f.write(f"Seeds: {all_seeds}\n")
                    f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # Display all sprites in the gallery
            for i, sprite in enumerate(all_sprites):
                col_idx = i % num_cols
                with cols[col_idx]:
                    # Display the sprite with pixelated rendering at actual size
                    st.markdown(f"<div class='sprite-container'>", unsafe_allow_html=True)
                    
                    # Create a container that will display the image at actual size
                    st.markdown(f"<div class='actual-size-container'>", unsafe_allow_html=True)
                    
                    # Convert to numpy array to get dimensions
                    sprite_array = np.array(sprite)
                    height, width = sprite_array.shape[:2]
                    
                    # Display the image without any scaling
                    st.image(sprite, caption=f"Sprite {i+1} (Seed: {all_seeds[i]}) - {width}x{height}px", use_container_width=False)
                    
                    # Add download button for each sprite
                    buf = sprite.convert("RGBA")
                    img_bytes = io.BytesIO()
                    sprite.save(img_bytes, format="PNG")
                    img_bytes.seek(0)
                    
                    st.download_button(
                        label="Download",
                        data=img_bytes,
                        file_name=f"sprite_{i+1}_seed_{all_seeds[i]}.png",
                        mime="image/png",
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
