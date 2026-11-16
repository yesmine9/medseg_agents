"""Méthodes classiques de segmentation, avec l'interface commune
segment_classical(img, method) -> (masque bool, proba float).

Ce sont des experts faibles : leur intérêt est surtout l'accord (ou le
désaccord) avec le U-Net, utilisé ensuite par l'agent Critique.
"""
import cv2
import numpy as np
from scipy import ndimage as ndi


def _largest_cc(mask):
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = ndi.sum(mask.astype(np.uint8), lab, range(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def _post(mask):
    mask = ndi.binary_opening(mask, iterations=2)
    mask = _largest_cc(mask)
    return ndi.binary_fill_holes(mask)


def _soft(mask, sigma=3.0):
    return np.clip(ndi.gaussian_filter(mask.astype(np.float32), sigma), 0.0, 1.0)


def otsu(img):
    """Seuillage d'Otsu sur la luminance ; le premier plan est la classe minoritaire."""
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (7, 7), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = th > 0
    if m.mean() > 0.5:
        m = ~m
    return _post(m)


def kmeans(img, k=3, seed=0):
    """K-means sur les couleurs Lab. Fond = cluster dominant sur le bord ;
    premier plan = autre cluster le plus présent au centre."""
    cv2.setRNGSeed(seed)
    blur = cv2.GaussianBlur(img, (7, 7), 0)
    lab = cv2.cvtColor(blur, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, _ = cv2.kmeans(lab, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(img.shape[:2])
    h, w = labels.shape
    b = max(2, h // 16)
    border = np.zeros((h, w), bool)
    border[:b] = border[-b:] = True
    border[:, :b] = border[:, -b:] = True
    center = np.zeros((h, w), bool)
    center[h // 4:3 * h // 4, w // 4:3 * w // 4] = True
    bg = np.bincount(labels[border], minlength=k).argmax()
    counts = np.bincount(labels[center], minlength=k).astype(float)
    counts[bg] = -1
    return _post(labels == counts.argmax())


def watershed(img):
    """Watershed avec marqueurs issus d'Otsu (érosion = premier plan sûr)."""
    base = otsu(img)
    if base.sum() == 0:
        return base
    sure_fg = ndi.binary_erosion(base, iterations=3)
    sure_bg = ~ndi.binary_dilation(base, iterations=5)
    if not sure_fg.any():
        return base
    markers = np.zeros(base.shape, np.int32)
    markers[sure_bg] = 1
    markers[sure_fg] = 2
    cv2.watershed(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), markers)
    return _post(markers == 2)


METHODS = {"otsu": otsu, "kmeans": kmeans, "watershed": watershed}


def segment_classical(img: np.ndarray, method: str):
    m = METHODS[method](img)
    return m, _soft(m)
