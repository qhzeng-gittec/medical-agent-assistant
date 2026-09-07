from pathlib import Path

from PIL import Image, ImageOps


def load_model_image(path: str | Path, max_edge: int | None = None) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
    if max_edge is not None:
        if max_edge < 32:
            raise ValueError("max_edge must be at least 32 pixels")
        image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    # Qwen's 16-pixel patches merge 2x2; pad to 32, keeping every source pixel unchanged.
    return ImageOps.expand(image, border=(0, 0, (-image.width) % 32, (-image.height) % 32), fill=0)
