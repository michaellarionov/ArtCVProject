"""Image loading, preprocessing, and utility functions."""

import torch
from PIL import Image
from torchvision import transforms


def get_device() -> torch.device:
    """Get the best available device (MPS for Mac, CUDA, or CPU)."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_image(path: str, max_size: int = 512) -> Image.Image:
    """
    Load an image and resize it to fit within max_size.
    
    Args:
        path: Path to the image file
        max_size: Maximum dimension (preserves aspect ratio)
        
    Returns:
        PIL Image in RGB format
    """
    image = Image.open(path).convert("RGB")
    
    # Resize if larger than max_size
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = tuple(int(dim * ratio) for dim in image.size)
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    return image


def preprocess(image: Image.Image, device: torch.device = None) -> torch.Tensor:
    """
    Convert PIL image to normalized tensor for VGG.
    
    Args:
        image: PIL Image
        device: Target device (auto-detected if None)
        
    Returns:
        Tensor of shape (1, 3, H, W), normalized for VGG
    """
    if device is None:
        device = get_device()
    
    # VGG normalization values (ImageNet statistics)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    tensor = transform(image).unsqueeze(0)
    return tensor.to(device)


def deprocess(tensor: torch.Tensor) -> Image.Image:
    """
    Convert normalized tensor back to PIL image.
    
    Args:
        tensor: Tensor of shape (1, 3, H, W) or (3, H, W)
        
    Returns:
        PIL Image
    """
    # Remove batch dimension if present
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Move to CPU and clone
    tensor = tensor.cpu().clone()
    
    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor * std + mean
    
    # Clamp to valid range
    tensor = tensor.clamp(0, 1)
    
    # Convert to PIL
    to_pil = transforms.ToPILImage()
    return to_pil(tensor)


def save_image(tensor: torch.Tensor, path: str) -> None:
    """
    Save a tensor as an image file.
    
    Args:
        tensor: Tensor of shape (1, 3, H, W) or (3, H, W)
        path: Output file path
    """
    image = deprocess(tensor)
    image.save(path)
    print(f"Saved image to {path}")


def resize_like(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Resize source tensor to match target tensor's spatial dimensions."""
    return torch.nn.functional.interpolate(
        source,
        size=target.shape[2:],
        mode="bilinear",
        align_corners=False
    )
