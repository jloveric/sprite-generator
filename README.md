# Sprite Generator

A super simple AI sprite generator that creates 2D sprites using Hugging Face image generation models. The tool generates an image based on a text prompt, removes the background, and resizes/pixelizes the result to create a sprite with a transparent background.

## Features

- Generate sprites using various AI models (SD3.5, SDXL, SaanaV2)
- Automatic background removal with transparent alpha channel
- Resize and pixelize to create sprite-like images
- Command-line interface with Click
- Standalone core functionality for future service integration
- Configurable Hugging Face cache directory

## Installation

```bash
# Clone the repository
git clone https://github.com/jloveric/sprite-generator.git
cd sprite-generator

# Install with Poetry
poetry install
```

## Usage

### Command Line Interface

```bash
# Generate a sprite
poetry run sprite-gen generate --prompt "a red dragon" --output dragon.png

# List available models
poetry run sprite-gen list-models

# Get help
poetry run sprite-gen --help
poetry run sprite-gen generate --help
```

### CLI Options

- `--prompt, -p`: Text prompt for image generation (required)
- `--output, -o`: Output path for the generated sprite (required)
- `--model, -m`: Model to use (sd3.5, sdxl, saana) (default: sd3.5)
- `--sprite-width`: Width of the output sprite (default: 64)
- `--sprite-height`: Height of the output sprite (default: 64)
- `--negative-prompt`: Negative prompt to guide generation away from certain attributes
- `--gen-width`: Width of the generated image before resizing (default: 512)
- `--gen-height`: Height of the generated image before resizing (default: 512)
- `--steps`: Number of denoising steps (default: 30)
- `--guidance-scale`: Scale for classifier-free guidance (default: 7.5)
- `--seed`: Random seed for reproducibility (optional)

### Python API

```python
from sprite_generator import generate_sprite

# Generate a sprite
sprite = generate_sprite(
    prompt="a red dragon",
    output_path="dragon.png",
    model_name="sd3.5",
    sprite_size=(64, 64)
)

# Display the sprite
sprite.show()
```

## Configuration

The Hugging Face cache directory is set to `/home/john/700gb/huggingface` by default. You can modify this in the `sprite_generator/sprite_gen.py` file.

## Requirements

- Python 3.10+
- PyTorch
- Diffusers
- Transformers
- rembg (for background removal)
- Click (for CLI)
- PIL/Pillow (for image processing)

## License

MIT
