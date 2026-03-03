"""VGG19 Feature Extractor for Neural Style Transfer."""

import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, List


class VGGFeatures(nn.Module):
    """
    VGG19 feature extractor for style transfer.
    
    Extracts feature maps from specified layers of a pre-trained VGG19 network.
    The network is frozen and used only for feature extraction.
    """
    
    # Mapping from layer names to VGG19 layer indices
    LAYER_MAPPING = {
        'conv1_1': 0,   # 64 filters
        'conv1_2': 2,
        'conv2_1': 5,   # 128 filters
        'conv2_2': 7,
        'conv3_1': 10,  # 256 filters
        'conv3_2': 12,
        'conv3_3': 14,
        'conv3_4': 16,
        'conv4_1': 19,  # 512 filters
        'conv4_2': 21,
        'conv4_3': 23,
        'conv4_4': 25,
        'conv5_1': 28,  # 512 filters
        'conv5_2': 30,
        'conv5_3': 32,
        'conv5_4': 34,
    }
    
    # Default layers for style transfer
    DEFAULT_CONTENT_LAYERS = ['conv4_2']
    DEFAULT_STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    
    def __init__(
        self,
        content_layers: List[str] = None,
        style_layers: List[str] = None,
        device: torch.device = None
    ):
        """
        Initialize VGG feature extractor.
        
        Args:
            content_layers: Layer names for content features
            style_layers: Layer names for style features
            device: Target device (auto-detected if None)
        """
        super().__init__()
        
        self.content_layers = content_layers or self.DEFAULT_CONTENT_LAYERS
        self.style_layers = style_layers or self.DEFAULT_STYLE_LAYERS
        
        # Determine device
        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        self.device = device
        
        # Load pre-trained VGG19
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        
        # Extract only the feature layers we need
        self.features = self._build_feature_extractor(vgg.features)
        
        # Freeze all parameters
        for param in self.features.parameters():
            param.requires_grad = False
        
        self.features.to(device)
        self.features.eval()
        
        print(f"VGG19 feature extractor initialized on {device}")
        print(f"  Content layers: {self.content_layers}")
        print(f"  Style layers: {self.style_layers}")
    
    def _build_feature_extractor(self, vgg_features: nn.Sequential) -> nn.Sequential:
        """
        Build a sequential model that stops at the last needed layer.
        
        Args:
            vgg_features: VGG19 feature layers
            
        Returns:
            Truncated sequential model
        """
        all_layers = self.content_layers + self.style_layers
        max_layer_idx = max(self.LAYER_MAPPING[name] for name in all_layers)
        
        # Only keep layers up to the last needed one
        layers = list(vgg_features.children())[:max_layer_idx + 1]
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features from specified layers.
        
        Args:
            x: Input tensor of shape (batch, 3, H, W)
            
        Returns:
            Dictionary mapping layer names to feature tensors
        """
        features = {}
        all_layers = set(self.content_layers + self.style_layers)
        
        # Get layer indices we care about
        layer_indices = {
            self.LAYER_MAPPING[name]: name 
            for name in all_layers
        }
        
        # Forward through each layer
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in layer_indices:
                features[layer_indices[idx]] = x
        
        return features
    
    def get_content_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract only content features."""
        all_features = self.forward(x)
        return {k: v for k, v in all_features.items() if k in self.content_layers}
    
    def get_style_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract only style features."""
        all_features = self.forward(x)
        return {k: v for k, v in all_features.items() if k in self.style_layers}


def get_vgg_features(
    content_layers: List[str] = None,
    style_layers: List[str] = None,
    device: torch.device = None
) -> VGGFeatures:
    """
    Factory function to create a VGG feature extractor.
    
    Args:
        content_layers: Layer names for content features
        style_layers: Layer names for style features
        device: Target device
        
    Returns:
        VGGFeatures instance
    """
    return VGGFeatures(
        content_layers=content_layers,
        style_layers=style_layers,
        device=device
    )
