"""Mesures de qualité d'image (agent Prétraitement)."""
import cv2
import numpy as np

_K = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)


def specular_mask(img_rgb: np.ndarray, v_min: int = 230, s_max: int = 50) -> np.ndarray:
    """Pixels très clairs et peu saturés : reflets de la lumière de l'endoscope."""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    return (hsv[..., 2] > v_min) & (hsv[..., 1] < s_max)


def image_quality(img_rgb: np.ndarray) -> dict:
    """img_rgb : uint8 (H, W, 3). Retourne 5 mesures simples."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    g = gray.astype(np.float32)
    h, w = g.shape
    # estimation du bruit (Immerkaer, 1996)
    conv = cv2.filter2D(g, -1, _K)[1:-1, 1:-1]
    noise = float(np.sqrt(np.pi / 2) * np.abs(conv).sum() / (6.0 * (w - 2) * (h - 2)))
    return {
        "noise": noise,
        "contrast": float(g.std()),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "specular": float(specular_mask(img_rgb).mean()),
        "brightness": float(g.mean()),
    }
