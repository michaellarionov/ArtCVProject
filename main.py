#!/usr/bin/env python3
"""
Neural Style Transfer CLI

Usage:
    # Classic Gatys style transfer
    python main.py --content image.jpg --style starry_night.jpg --output result.jpg
    
    # Fast style transfer (requires pre-trained model)
    python main.py --content image.jpg --output result.jpg --method fast --model pretrained/style.pth
"""

import argparse
import os
import sys
import yaml
from pathlib import Path


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / "configs" / "default.yaml"
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_gatys(args, config):
    """Run Gatys-style optimization-based transfer."""
    from models.gatys import GatysStyleTransfer
    
    # Get settings from config or args
    gatys_config = config.get("gatys", {})
    
    style_transfer = GatysStyleTransfer(
        content_weight=args.content_weight or gatys_config.get("content_weight", 1),
        style_weight=args.style_weight or gatys_config.get("style_weight", 1000000),
        num_steps=args.num_steps or gatys_config.get("num_steps", 300),
    )
    
    result = style_transfer.transfer_from_paths(
        content_path=args.content,
        style_path=args.style,
        output_path=args.output,
        max_size=args.max_size or config.get("image", {}).get("max_size", 512),
        init_image=args.init,
        optimizer_type=args.optimizer,
    )
    
    return result


def run_fast(args, config):
    """Run fast feed-forward style transfer."""
    from models.fast_style import FastStyleTransfer
    
    if not args.model:
        print("Error: --model is required for fast style transfer")
        print("Train a model first with: python scripts/train_fast_style.py")
        sys.exit(1)
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    style_transfer = FastStyleTransfer(model_path=args.model)
    
    result = style_transfer.transfer_from_path(
        image_path=args.content,
        output_path=args.output,
        max_size=args.max_size or config.get("image", {}).get("max_size", 512),
    )
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Neural Style Transfer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Gatys style transfer (slower, more flexible)
  python main.py -c photo.jpg -s starry_night.jpg -o output.jpg
  
  # Fast style transfer (real-time, requires pre-trained model)
  python main.py -c photo.jpg -o output.jpg --method fast --model pretrained/starry.pth
  
  # Custom settings
  python main.py -c photo.jpg -s style.jpg -o out.jpg --steps 500 --style-weight 500000
        """
    )
    
    # Required arguments
    parser.add_argument(
        "-c", "--content", type=str, required=True,
        help="Path to content image"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="output.jpg",
        help="Path for output image (default: output.jpg)"
    )
    
    # Method selection
    parser.add_argument(
        "--method", type=str, choices=["gatys", "fast"], default="gatys",
        help="Style transfer method (default: gatys)"
    )
    
    # Gatys-specific arguments
    parser.add_argument(
        "-s", "--style", type=str,
        help="Path to style image (required for gatys method)"
    )
    parser.add_argument(
        "--content-weight", type=float,
        help="Weight for content loss"
    )
    parser.add_argument(
        "--style-weight", type=float,
        help="Weight for style loss"
    )
    parser.add_argument(
        "--num-steps", type=int,
        help="Number of optimization steps"
    )
    parser.add_argument(
        "--init", type=str, choices=["content", "style", "random"], default="content",
        help="Initialize output as content, style, or random"
    )
    parser.add_argument(
        "--optimizer", type=str, choices=["lbfgs", "adam"], default="lbfgs",
        help="Optimizer for Gatys method (default: lbfgs)"
    )
    
    # Fast style arguments
    parser.add_argument(
        "--model", type=str,
        help="Path to pre-trained fast style model"
    )
    
    # Common arguments
    parser.add_argument(
        "--max-size", type=int,
        help="Maximum image dimension"
    )
    parser.add_argument(
        "--config", type=str,
        help="Path to config file"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.method == "gatys" and not args.style:
        parser.error("--style is required for gatys method")
    
    # Load config
    config = load_config(args.config)
    
    # Print settings
    print("=" * 50)
    print("Neural Style Transfer")
    print("=" * 50)
    print(f"Method: {args.method}")
    print(f"Content: {args.content}")
    if args.style:
        print(f"Style: {args.style}")
    print(f"Output: {args.output}")
    print("=" * 50)
    
    # Run style transfer
    if args.method == "gatys":
        result = run_gatys(args, config)
    else:
        result = run_fast(args, config)
    
    print("\nStyle transfer complete!")
    print(f"Result saved to: {args.output}")
    
    return result


if __name__ == "__main__":
    main()
