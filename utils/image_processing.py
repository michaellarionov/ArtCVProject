from PIL import Image
import numpy as np


def load_image(image_path, size=(512, 512)):

    image = Image.open(image_path).convert("RGB")
    image = image.resize(size)

    return image


def save_image(image, path):

    image.save(path)


def image_to_numpy(image):

    return np.array(image) / 255.0