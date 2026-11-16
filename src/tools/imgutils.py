import numpy as np


def tensor_to_uint8(x) -> np.ndarray:
    """Tenseur torch (3, H, W) dans [0, 1] -> image uint8 (H, W, 3)."""
    return (x.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
