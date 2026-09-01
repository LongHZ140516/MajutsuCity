"""Small deterministic edge blending helpers for generated texture maps."""

from __future__ import annotations


def blend_wrapped_edges(
    image,
    *,
    left_right: bool = True,
    top_bottom: bool = True,
    max_band: int = 96,
):
    """Blend opposing edge bands so a generated RGB image wraps cleanly."""
    import numpy as np
    from PIL import Image

    array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    height, width = array.shape[:2]
    band = max(2, min(max_band, width // 6, height // 6))
    if left_right:
        for offset in range(band):
            alpha = 1.0 - offset / (band - 1)
            left = array[:, offset, :].copy()
            right = array[:, width - 1 - offset, :].copy()
            average = (left + right) * 0.5
            array[:, offset, :] = left * (1.0 - alpha) + average * alpha
            array[:, width - 1 - offset, :] = (
                right * (1.0 - alpha) + average * alpha
            )
    if top_bottom:
        for offset in range(band):
            alpha = 1.0 - offset / (band - 1)
            top = array[offset, :, :].copy()
            bottom = array[height - 1 - offset, :, :].copy()
            average = (top + bottom) * 0.5
            array[offset, :, :] = top * (1.0 - alpha) + average * alpha
            array[height - 1 - offset, :, :] = (
                bottom * (1.0 - alpha) + average * alpha
            )
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
