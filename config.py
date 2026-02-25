# =============================================================================
# ART PERIOD CONFIGURATIONS
# =============================================================================
# Each period has a prompt_style (used for generation) and example artists

ART_PERIODS = {
    "Renaissance": {
        "prompt_style": "Renaissance oil painting, classical composition, chiaroscuro lighting",
        "artists": ["Leonardo da Vinci", "Michelangelo", "Raphael"],
    },
    "Baroque": {
        "prompt_style": "Baroque painting, dramatic lighting, rich colors, emotional intensity",
        "artists": ["Caravaggio", "Rembrandt", "Vermeer"],
    },
    "Impressionism": {
        "prompt_style": "Impressionist painting, visible brushstrokes, light and movement",
        "artists": ["Monet", "Renoir", "Degas"],
    },
    "Post-Impressionism": {
        "prompt_style": "Post-Impressionist painting, bold colors, geometric forms",
        "artists": ["Van Gogh", "Cézanne", "Gauguin"],
    },
    "Expressionism": {
        "prompt_style": "Expressionist painting, distorted forms, vivid colors",
        "artists": ["Munch", "Kandinsky", "Kirchner"],
    },
    "Surrealism": {
        "prompt_style": "Surrealist painting, dreamlike imagery, symbolic elements",
        "artists": ["Dalí", "Magritte", "Ernst"],
    },
}


# =============================================================================
# COLOR PALETTE PRESETS
# =============================================================================
# Each palette has RGB colors and a text description for prompts

COLOR_PALETTES = {
    "Warm Sunset": {
        "colors": [(255, 140, 50), (255, 200, 100), (180, 80, 60), (255, 220, 180), (120, 60, 40)],
        "description": "warm golden orange and amber tones",
    },
    "Cool Ocean": {
        "colors": [(30, 80, 120), (70, 150, 180), (200, 230, 240), (40, 60, 90), (100, 180, 200)],
        "description": "cool blue and teal ocean tones",
    },
    "Forest Green": {
        "colors": [(34, 85, 51), (85, 128, 68), (170, 190, 130), (51, 51, 34), (200, 200, 170)],
        "description": "natural green and earthy forest tones",
    },
    "Royal Purple": {
        "colors": [(80, 40, 120), (140, 80, 160), (200, 150, 200), (40, 20, 60), (220, 200, 230)],
        "description": "rich purple and violet royal tones",
    },
    "Monochrome": {
        "colors": [(20, 20, 20), (80, 80, 80), (140, 140, 140), (200, 200, 200), (245, 245, 245)],
        "description": "black and white monochromatic tones",
    },
}


# =============================================================================
# GENERATION SETTINGS
# =============================================================================

DEFAULT_STEPS = 50
DEFAULT_GUIDANCE = 7.5
MODEL_ID = "runwayml/stable-diffusion-v1-5"
