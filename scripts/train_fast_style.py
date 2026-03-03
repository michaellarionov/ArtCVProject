"""Training script for Fast Style Transfer."""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vgg import VGGFeatures
from models.fast_style import FastStyleNet
from utils.image_utils import load_image, preprocess, get_device
from utils.losses import ContentLoss, StyleLoss, TotalVariationLoss, gram_matrix


def train(args):
    """Train fast style transfer model."""
    device = get_device()
    print(f"Training on device: {device}")
    
    # Load style image
    style_image = load_image(args.style_image, args.style_size)
    style_tensor = preprocess(style_image, device)
    
    # Initialize models
    transformer = FastStyleNet(num_residual_blocks=args.num_residual).to(device)
    vgg = VGGFeatures(device=device)
    
    # Get style features
    with torch.no_grad():
        style_features = vgg.get_style_features(style_tensor)
        style_grams = {
            name: gram_matrix(feat) 
            for name, feat in style_features.items()
        }
    
    # Optimizer
    optimizer = optim.Adam(transformer.parameters(), lr=args.lr)
    
    # Data loader - expects a folder of images
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    dataset = datasets.ImageFolder(args.dataset, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    # Training loop
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        transformer.train()
        
        epoch_content_loss = 0
        epoch_style_loss = 0
        epoch_tv_loss = 0
        
        pbar = tqdm(dataloader, desc="Training")
        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(device)
            
            optimizer.zero_grad()
            
            # Generate styled images
            output = transformer(images)
            
            # Scale output from [-1, 1] to VGG range
            output_scaled = (output + 1) / 2
            output_vgg = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )(output_scaled)
            
            # Get features
            output_features = vgg(output_vgg)
            content_features = vgg.get_content_features(images)
            
            # Content loss
            content_loss = 0
            for name in vgg.content_layers:
                content_loss += nn.functional.mse_loss(
                    output_features[name],
                    content_features[name]
                )
            
            # Style loss
            style_loss = 0
            for name in vgg.style_layers:
                output_gram = gram_matrix(output_features[name])
                # Expand style gram to match batch size
                target_gram = style_grams[name].expand(images.size(0), -1, -1)
                style_loss += nn.functional.mse_loss(output_gram, target_gram)
            
            # Total variation loss
            tv_loss = (
                torch.mean(torch.abs(output[:, :, :, :-1] - output[:, :, :, 1:])) +
                torch.mean(torch.abs(output[:, :, :-1, :] - output[:, :, 1:, :]))
            )
            
            # Total loss
            loss = (
                args.content_weight * content_loss +
                args.style_weight * style_loss +
                args.tv_weight * tv_loss
            )
            
            loss.backward()
            optimizer.step()
            
            epoch_content_loss += content_loss.item()
            epoch_style_loss += style_loss.item()
            epoch_tv_loss += tv_loss.item()
            
            pbar.set_postfix({
                'content': f'{content_loss.item():.4f}',
                'style': f'{style_loss.item():.4f}',
                'tv': f'{tv_loss.item():.6f}'
            })
        
        # Print epoch summary
        n_batches = len(dataloader)
        print(f"  Content: {epoch_content_loss / n_batches:.4f}")
        print(f"  Style: {epoch_style_loss / n_batches:.4f}")
        print(f"  TV: {epoch_tv_loss / n_batches:.6f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(
            args.checkpoint_dir,
            f"fast_style_epoch_{epoch + 1}.pth"
        )
        torch.save(transformer.state_dict(), checkpoint_path)
        print(f"  Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(args.checkpoint_dir, "fast_style_final.pth")
    torch.save(transformer.state_dict(), final_path)
    print(f"\nTraining complete! Final model: {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Fast Style Transfer")
    
    # Required arguments
    parser.add_argument(
        "--style-image", type=str, required=True,
        help="Path to style image"
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Path to training dataset (ImageFolder format)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--epochs", type=int, default=2,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Training batch size"
    )
    parser.add_argument(
        "--image-size", type=int, default=256,
        help="Training image size"
    )
    parser.add_argument(
        "--style-size", type=int, default=512,
        help="Style image size"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--content-weight", type=float, default=1.0,
        help="Content loss weight"
    )
    parser.add_argument(
        "--style-weight", type=float, default=1e5,
        help="Style loss weight"
    )
    parser.add_argument(
        "--tv-weight", type=float, default=1e-5,
        help="Total variation loss weight"
    )
    parser.add_argument(
        "--num-residual", type=int, default=5,
        help="Number of residual blocks"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="pretrained",
        help="Directory for saving checkpoints"
    )
    
    args = parser.parse_args()
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    train(args)


if __name__ == "__main__":
    main()
