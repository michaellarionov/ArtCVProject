import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from skimage import color


def extract_palette_lab(image_path, num_colors=5):

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Resize for faster computation
    image = image.resize((256, 256))

    # Convert to numpy array and normalize
    rgb_image = np.array(image) / 255.0

    # Convert RGB → LAB color space
    lab_image = color.rgb2lab(rgb_image)

    # Flatten pixel data
    pixels = lab_image.reshape(-1, 3)

    # Run KMeans clustering
    kmeans = KMeans(
        n_clusters=num_colors,
        random_state=42,
        n_init=10
    )

    kmeans.fit(pixels)

    # Get cluster centers (palette in LAB space)
    lab_palette = kmeans.cluster_centers_

    # Convert LAB palette → RGB
    rgb_palette = color.lab2rgb(lab_palette.reshape(1, num_colors, 3))[0]

    # Convert to 0-255 RGB integers
    rgb_palette = (rgb_palette * 255).astype(int)

    return [tuple(color) for color in rgb_palette]


if __name__ == "__main__":
    palette = extract_palette_lab("test.jpg")
    print(palette)