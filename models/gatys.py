"""Classic Gatys Style Transfer (Optimization-based)."""

import torch
import torch.optim as optim
from PIL import Image
from tqdm import tqdm
from typing import Callable, Optional

from .vgg import VGGFeatures
from utils.image_utils import load_image, preprocess, deprocess, save_image, get_device
from utils.losses import ContentLoss, StyleLoss


class GatysStyleTransfer:
    """
    Neural Style Transfer using the Gatys et al. optimization approach.
    
    This method directly optimizes the pixel values of the output image
    to minimize content and style losses.
    
    Reference: "A Neural Algorithm of Artistic Style" (Gatys et al., 2015)
    """
    
    def __init__(
        self,
        content_weight: float = 1.0,
        style_weight: float = 1000000.0,
        num_steps: int = 300,
        device: torch.device = None
    ):
        """
        Initialize Gatys style transfer.
        
        Args:
            content_weight: Weight for content loss
            style_weight: Weight for style loss
            num_steps: Number of optimization steps
            device: Target device (auto-detected if None)
        """
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.num_steps = num_steps
        self.device = device or get_device()
        
        # Initialize VGG feature extractor
        self.vgg = VGGFeatures(device=self.device)
    
    def transfer(
        self,
        content_image: Image.Image,
        style_image: Image.Image,
        init_image: str = "content",
        optimizer_type: str = "lbfgs",
        progress_callback: Optional[Callable[[int, float], None]] = None
    ) -> Image.Image:
        """
        Perform style transfer.
        
        Args:
            content_image: PIL Image for content
            style_image: PIL Image for style
            init_image: Initialize output as "content", "style", or "random"
            optimizer_type: "lbfgs" (recommended) or "adam"
            progress_callback: Optional callback(step, loss) for progress updates
            
        Returns:
            Stylized PIL Image
        """
        # Preprocess images
        content_tensor = preprocess(content_image, self.device)
        style_tensor = preprocess(style_image, self.device)
        
        # Resize style to match content
        if style_tensor.shape != content_tensor.shape:
            style_tensor = torch.nn.functional.interpolate(
                style_tensor,
                size=content_tensor.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        
        # Extract target features
        with torch.no_grad():
            content_features = self.vgg.get_content_features(content_tensor)
            style_features = self.vgg.get_style_features(style_tensor)
        
        # Initialize output image
        if init_image == "content":
            output = content_tensor.clone()
        elif init_image == "style":
            output = style_tensor.clone()
        else:  # random
            output = torch.randn_like(content_tensor)
        
        output.requires_grad_(True)
        
        # Create loss modules
        content_losses = {
            name: ContentLoss(features)
            for name, features in content_features.items()
        }
        style_losses = {
            name: StyleLoss(features)
            for name, features in style_features.items()
        }
        
        # Run optimization
        if optimizer_type == "lbfgs":
            output = self._optimize_lbfgs(
                output, content_losses, style_losses, progress_callback
            )
        else:
            output = self._optimize_adam(
                output, content_losses, style_losses, progress_callback
            )
        
        # Convert to PIL Image
        return deprocess(output)
    
    def _optimize_lbfgs(
        self,
        output: torch.Tensor,
        content_losses: dict,
        style_losses: dict,
        progress_callback: Optional[Callable] = None
    ) -> torch.Tensor:
        """Optimize using L-BFGS (recommended for style transfer)."""
        optimizer = optim.LBFGS([output], lr=1.0, max_iter=20)
        
        step = [0]  # Use list to modify in closure
        
        pbar = tqdm(total=self.num_steps, desc="Style Transfer")
        
        while step[0] < self.num_steps:
            def closure():
                optimizer.zero_grad()
                
                # Clamp to valid range
                with torch.no_grad():
                    output.clamp_(-2.5, 2.5)
                
                # Extract features from current output
                features = self.vgg(output)
                
                # Compute content loss
                c_loss = sum(
                    loss(features[name])
                    for name, loss in content_losses.items()
                )
                
                # Compute style loss
                s_loss = sum(
                    loss(features[name])
                    for name, loss in style_losses.items()
                )
                
                # Total loss
                total_loss = (
                    self.content_weight * c_loss +
                    self.style_weight * s_loss
                )
                
                total_loss.backward()
                
                step[0] += 1
                pbar.update(1)
                pbar.set_postfix({
                    'content': f'{c_loss.item():.4f}',
                    'style': f'{s_loss.item():.6f}',
                    'total': f'{total_loss.item():.4f}'
                })
                
                if progress_callback:
                    progress_callback(step[0], total_loss.item())
                
                return total_loss
            
            optimizer.step(closure)
        
        pbar.close()
        
        # Final clamp
        with torch.no_grad():
            output.clamp_(-2.5, 2.5)
        
        return output
    
    def _optimize_adam(
        self,
        output: torch.Tensor,
        content_losses: dict,
        style_losses: dict,
        progress_callback: Optional[Callable] = None
    ) -> torch.Tensor:
        """Optimize using Adam (slower but more stable)."""
        optimizer = optim.Adam([output], lr=0.01)
        
        pbar = tqdm(range(self.num_steps), desc="Style Transfer")
        
        for step in pbar:
            optimizer.zero_grad()
            
            # Extract features
            features = self.vgg(output)
            
            # Compute losses
            c_loss = sum(
                loss(features[name])
                for name, loss in content_losses.items()
            )
            s_loss = sum(
                loss(features[name])
                for name, loss in style_losses.items()
            )
            
            total_loss = (
                self.content_weight * c_loss +
                self.style_weight * s_loss
            )
            
            total_loss.backward()
            optimizer.step()
            
            # Clamp
            with torch.no_grad():
                output.clamp_(-2.5, 2.5)
            
            pbar.set_postfix({
                'content': f'{c_loss.item():.4f}',
                'style': f'{s_loss.item():.6f}'
            })
            
            if progress_callback:
                progress_callback(step, total_loss.item())
        
        return output
    
    def transfer_from_paths(
        self,
        content_path: str,
        style_path: str,
        output_path: str = None,
        max_size: int = 512,
        **kwargs
    ) -> Image.Image:
        """
        Convenience method to transfer from file paths.
        
        Args:
            content_path: Path to content image
            style_path: Path to style image
            output_path: Optional path to save result
            max_size: Maximum image dimension
            **kwargs: Additional arguments for transfer()
            
        Returns:
            Stylized PIL Image
        """
        content_image = load_image(content_path, max_size)
        style_image = load_image(style_path, max_size)
        
        print(f"Content image size: {content_image.size}")
        print(f"Style image size: {style_image.size}")
        
        result = self.transfer(content_image, style_image, **kwargs)
        
        if output_path:
            result.save(output_path)
            print(f"Saved result to {output_path}")
        
        return result
