"""Fast Style Transfer Network (Johnson et al.)."""

import torch
import torch.nn as nn
from PIL import Image
from typing import Optional

from utils.image_utils import load_image, preprocess, deprocess, get_device


class ConvBlock(nn.Module):
    """Convolutional block with instance normalization and ReLU."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        upsample: bool = False
    ):
        super().__init__()
        
        self.upsample = upsample
        if upsample:
            self.upsample_layer = nn.Upsample(
                scale_factor=2, 
                mode='nearest'
            )
        
        # Reflection padding to avoid border artifacts
        padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding)
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=kernel_size,
            stride=stride,
            padding=0
        )
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.upsample:
            x = self.upsample_layer(x)
        x = self.reflection_pad(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block with two conv layers."""
    
    def __init__(self, channels: int):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(channels, affine=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class FastStyleNet(nn.Module):
    """
    Fast Style Transfer Network.
    
    A feed-forward convolutional network that transforms images
    in a single forward pass.
    
    Architecture:
    - 3 downsampling conv layers
    - N residual blocks
    - 3 upsampling conv layers
    
    Reference: "Perceptual Losses for Real-Time Style Transfer" (Johnson et al., 2016)
    """
    
    def __init__(self, num_residual_blocks: int = 5):
        """
        Args:
            num_residual_blocks: Number of residual blocks (default: 5)
        """
        super().__init__()
        
        # Initial convolution
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(4),
            nn.Conv2d(3, 32, kernel_size=9, stride=1, padding=0),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
        )
        
        # Downsampling
        self.down1 = ConvBlock(32, 64, kernel_size=3, stride=2)
        self.down2 = ConvBlock(64, 128, kernel_size=3, stride=2)
        
        # Residual blocks
        self.residual = nn.Sequential(
            *[ResidualBlock(128) for _ in range(num_residual_blocks)]
        )
        
        # Upsampling
        self.up1 = ConvBlock(128, 64, kernel_size=3, upsample=True)
        self.up2 = ConvBlock(64, 32, kernel_size=3, upsample=True)
        
        # Final convolution (no activation, outputs in [-1, 1] range)
        self.final = nn.Sequential(
            nn.ReflectionPad2d(4),
            nn.Conv2d(32, 3, kernel_size=9, stride=1, padding=0),
            nn.Tanh(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform input image.
        
        Args:
            x: Input tensor of shape (batch, 3, H, W)
            
        Returns:
            Stylized tensor of shape (batch, 3, H, W)
        """
        x = self.initial(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.residual(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.final(x)
        return x


class FastStyleTransfer:
    """
    Fast Style Transfer wrapper for inference.
    
    Loads a pre-trained FastStyleNet and performs style transfer
    in a single forward pass.
    """
    
    def __init__(
        self,
        model_path: str = None,
        device: torch.device = None
    ):
        """
        Initialize fast style transfer.
        
        Args:
            model_path: Path to pre-trained model weights
            device: Target device (auto-detected if None)
        """
        self.device = device or get_device()
        self.model = FastStyleNet()
        
        if model_path:
            self.load_model(model_path)
        
        self.model.to(self.device)
        self.model.eval()
    
    def load_model(self, model_path: str) -> None:
        """Load pre-trained model weights."""
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        print(f"Loaded model from {model_path}")
    
    def save_model(self, model_path: str) -> None:
        """Save model weights."""
        torch.save(self.model.state_dict(), model_path)
        print(f"Saved model to {model_path}")
    
    @torch.no_grad()
    def transfer(self, image: Image.Image) -> Image.Image:
        """
        Perform style transfer on a single image.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Stylized PIL Image
        """
        # Preprocess
        tensor = preprocess(image, self.device)
        
        # Transform
        output = self.model(tensor)
        
        # The output is in [-1, 1] from tanh, scale to [0, 1]
        output = (output + 1) / 2
        
        # Denormalize (undo VGG normalization that preprocess applied)
        # Since we scaled output to [0, 1], we need to match that
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        
        # Undo preprocess normalization
        tensor_unnorm = tensor * std + mean
        
        # Blend or just use output directly
        # For simplicity, we'll transform the output properly
        output = output.clamp(0, 1)
        
        # Convert to PIL
        output = output.squeeze(0).cpu()
        from torchvision.transforms import ToPILImage
        to_pil = ToPILImage()
        return to_pil(output)
    
    def transfer_from_path(
        self,
        image_path: str,
        output_path: str = None,
        max_size: int = 512
    ) -> Image.Image:
        """
        Transfer style from file path.
        
        Args:
            image_path: Path to input image
            output_path: Optional path to save result
            max_size: Maximum image dimension
            
        Returns:
            Stylized PIL Image
        """
        image = load_image(image_path, max_size)
        result = self.transfer(image)
        
        if output_path:
            result.save(output_path)
            print(f"Saved result to {output_path}")
        
        return result
