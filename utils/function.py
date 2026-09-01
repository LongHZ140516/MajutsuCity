import numpy as np
from PIL import Image

_PALETTES = {
    "majutsu": {
        "empty": [0, 0, 0],
        "traffic_roads": [255, 0, 0],
        "building": [255, 255, 0],
        "vegetation": [34, 139, 34],
        "water": [0, 0, 255],
        "ground": [128, 128, 128],
    },
    "citydreamer": {
        "empty": [0, 0, 0],
        "traffic_roads": [96, 0, 0],
        "building": [96, 96, 0],
        "vegetation": [0, 96, 0],
        "water": [0, 0, 96],
        "ground": [128, 128, 128],
    },
}


def get_palette_dict(palette_name: str = "majutsu") -> dict:
    """
    Get a supported layout palette.

    The label order is intentionally identical across palettes because it defines
    the channel order returned by ``image_to_mask``.
    """
    if palette_name not in _PALETTES:
        supported = ", ".join(sorted(_PALETTES))
        raise ValueError(f"Unknown palette '{palette_name}'. Supported palettes: {supported}")
    return _PALETTES[palette_name]


def _detect_palette(image: np.ndarray) -> str:
    """Select the palette that exactly matches the most pixels."""
    scores = {}
    for palette_name, palette_dict in _PALETTES.items():
        matched = np.zeros(image.shape[:2], dtype=bool)
        for color in palette_dict.values():
            matched |= np.all(image == color, axis=-1)
        scores[palette_name] = int(matched.sum())
    return max(scores, key=scores.get)


def image_to_mask(image: Image.Image, palette_name: str = "auto") -> np.ndarray:
    """
    Convert a color-coded layout image to one binary channel per semantic label.
    """
    image = np.array(image.convert("RGB"), dtype=np.uint8)
    if palette_name == "auto":
        palette_name = _detect_palette(image)
    palette_dict = get_palette_dict(palette_name)
    palette = list(palette_dict.values())

    mask = np.zeros((image.shape[0], image.shape[1], len(palette)), dtype=bool)

    for i, color in enumerate(palette):
        mask[:, :, i] = np.all(image == color, axis=-1)

    return mask


def extract_lmask(label_name: str, mask: np.ndarray) -> np.ndarray:
    """
    extract the mask of the label
    """
    palette_dict = get_palette_dict()
    idx = list(palette_dict.keys()).index(label_name)
    return mask[:, :, idx]
