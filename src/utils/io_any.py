# src/utils/io_any.py
from pathlib import Path
import numpy as np

def imread_any(path):
    """Robust image reader: tries PIL first (handles webp), fallbacks to cv2."""
    p = Path(path)
    # PIL
    try:
        from PIL import Image
        with Image.open(p) as im:
            return np.array(im.convert("RGB"))
    except Exception:
        pass
    # OpenCV fallback
    try:
        import cv2
        a = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if a is not None:
            return a[:, :, ::-1]  # BGR->RGB
    except Exception:
        pass
    raise RuntimeError(f"imread_any failed: {p}")

def imwrite_png(path, rgb):
    """Saves RGB ndarray as PNG via PIL (no cv2 dependency for webp)."""
    from PIL import Image
    Image.fromarray(rgb).save(Path(path), format="PNG")
