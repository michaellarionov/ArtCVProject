# Neural Style Transfer

A PyTorch implementation of neural style transfer optimized for Mac (including 8GB systems).

## Features

- **Classic Gatys Style Transfer**: Optimization-based approach for flexible, high-quality results
- **Fast Style Transfer**: Feed-forward network for real-time stylization
- **Multiple Interfaces**: CLI, Jupyter notebook, and Gradio web app
- **Mac Optimized**: MPS (Metal Performance Shaders) support for Apple Silicon

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Command Line

```bash
# Classic Gatys style transfer
python main.py --content data/content/photo.jpg --style data/style/starry_night.jpg --output output.jpg

# Fast style transfer
python main.py --content data/content/photo.jpg --style data/style/starry_night.jpg --output output.jpg --method fast
```

### Web Interface

```bash
python app/gradio_app.py
```

Then open http://localhost:7860 in your browser.

### Jupyter Notebook

```bash
jupyter notebook notebooks/style_transfer_demo.ipynb
```

## Project Structure

```
ArtCVProject/
├── configs/          # Hyperparameter configurations
├── data/             # Content, style, and output images
├── models/           # Neural network architectures
├── utils/            # Helper functions
├── pretrained/       # Pre-trained model weights
├── scripts/          # Training and utility scripts
├── notebooks/        # Jupyter notebooks
├── app/              # Web interface
└── main.py           # CLI entry point
```

## Memory Usage

Optimized for 8GB systems:
- Default image size: 512x512
- Automatic MPS device detection
- Gradient checkpointing for training

## References

- [A Neural Algorithm of Artistic Style (Gatys et al., 2015)](https://arxiv.org/abs/1508.06576)
- [Perceptual Losses for Real-Time Style Transfer (Johnson et al., 2016)](https://arxiv.org/abs/1603.08155)

## License

MIT License
