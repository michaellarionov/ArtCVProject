from setuptools import setup, find_packages

setup(
    name="neural-style-transfer",
    version="1.0.0",
    description="Neural Style Transfer with PyTorch, optimized for Mac",
    author="Your Name",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "Pillow>=9.0.0",
        "PyYAML>=6.0",
        "tqdm>=4.65.0",
        "gradio>=4.0.0",
        "numpy>=1.24.0",
    ],
    entry_points={
        "console_scripts": [
            "style-transfer=main:main",
        ],
    },
)
