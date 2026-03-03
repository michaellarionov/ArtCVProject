"""Download pre-trained model weights."""

import os
import torch
from torchvision import models


def download_vgg19():
    """
    Download VGG19 weights.
    
    The weights are automatically downloaded by torchvision
    when VGGFeatures is first instantiated.
    """
    print("Downloading VGG19 pre-trained weights...")
    
    # This triggers the download
    _ = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    
    print("VGG19 weights downloaded successfully!")
    print("Weights are cached in the torch hub directory.")


def main():
    """Download all required weights."""
    download_vgg19()
    
    print("\nAll weights downloaded!")
    print("\nNote: Fast Style Transfer models need to be trained separately.")
    print("Use scripts/train_fast_style.py to train a model on your style image.")


if __name__ == "__main__":
    main()
