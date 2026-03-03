"""Loss functions for neural style transfer."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def gram_matrix(features: torch.Tensor) -> torch.Tensor:
    """
    Compute the Gram matrix of feature maps.
    
    The Gram matrix captures style information by computing correlations
    between feature channels.
    
    Args:
        features: Tensor of shape (batch, channels, height, width)
        
    Returns:
        Gram matrix of shape (batch, channels, channels)
    """
    batch, channels, height, width = features.size()
    
    # Reshape to (batch, channels, height * width)
    features = features.view(batch, channels, height * width)
    
    # Compute Gram matrix: G = F * F^T
    gram = torch.bmm(features, features.transpose(1, 2))
    
    # Normalize by number of elements
    gram = gram / (channels * height * width)
    
    return gram


class ContentLoss(nn.Module):
    """
    Content loss measures the difference between content representations.
    
    Uses MSE between feature maps at a specific layer.
    """
    
    def __init__(self, target_features: torch.Tensor):
        """
        Args:
            target_features: Feature maps from the content image
        """
        super().__init__()
        self.target = target_features.detach()
    
    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """Compute content loss."""
        return F.mse_loss(input_features, self.target)


class StyleLoss(nn.Module):
    """
    Style loss measures the difference between style representations.
    
    Uses MSE between Gram matrices of feature maps.
    """
    
    def __init__(self, target_features: torch.Tensor):
        """
        Args:
            target_features: Feature maps from the style image
        """
        super().__init__()
        self.target_gram = gram_matrix(target_features).detach()
    
    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """Compute style loss."""
        input_gram = gram_matrix(input_features)
        return F.mse_loss(input_gram, self.target_gram)


class TotalVariationLoss(nn.Module):
    """
    Total variation loss for spatial smoothness.
    
    Encourages neighboring pixels to have similar values,
    reducing noise and artifacts.
    """
    
    def __init__(self, weight: float = 1.0):
        """
        Args:
            weight: Weight for the TV loss
        """
        super().__init__()
        self.weight = weight
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Compute total variation loss.
        
        Args:
            image: Tensor of shape (batch, channels, height, width)
            
        Returns:
            Scalar TV loss
        """
        # Horizontal variation
        h_var = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]))
        
        # Vertical variation
        v_var = torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
        
        return self.weight * (h_var + v_var)


class PerceptualLoss(nn.Module):
    """
    Combined perceptual loss for style transfer.
    
    Combines content loss, style loss, and optionally TV loss.
    """
    
    def __init__(
        self,
        content_weight: float = 1.0,
        style_weight: float = 1000000.0,
        tv_weight: float = 0.0
    ):
        """
        Args:
            content_weight: Weight for content loss
            style_weight: Weight for style loss
            tv_weight: Weight for total variation loss
        """
        super().__init__()
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight
        self.tv_loss = TotalVariationLoss() if tv_weight > 0 else None
    
    def forward(
        self,
        content_losses: list,
        style_losses: list,
        generated_image: torch.Tensor = None
    ) -> tuple:
        """
        Compute combined perceptual loss.
        
        Args:
            content_losses: List of content loss values
            style_losses: List of style loss values
            generated_image: Generated image tensor (for TV loss)
            
        Returns:
            Tuple of (total_loss, content_loss, style_loss, tv_loss)
        """
        content_loss = sum(content_losses)
        style_loss = sum(style_losses)
        
        total_loss = (
            self.content_weight * content_loss +
            self.style_weight * style_loss
        )
        
        tv_loss = torch.tensor(0.0)
        if self.tv_loss is not None and generated_image is not None:
            tv_loss = self.tv_loss(generated_image)
            total_loss += self.tv_weight * tv_loss
        
        return total_loss, content_loss, style_loss, tv_loss
