"""Prétraitement : règles de choix (adaptatif) et opérations."""
import json
from pathlib import Path

import cv2
import numpy as np

from src.tools.quality import specular_mask

THRESH_FILE = Path("configs/quality_thresholds.json")
DEFAULT_THRESH = {"noise_hi": 3.0, "contrast_lo": 35.0, "specular_hi": 0.01}
FIXED_PIPELINE = ["denoise", "clahe"]  # prétraitement fixe (baseline B2)


def load_thresholds(path=THRESH_FILE) -> dict:
    th = dict(DEFAULT_THRESH)
    p = Path(path)
    if p.exists():
        th.update(json.loads(p.read_text()))
    return th


def choose_pipeline(q: dict, th: dict | None = None) -> list:
    """Règle : ne traiter que ce qui est dégradé."""
    th = th or load_thresholds()
    steps = []
    if q["specular"] > th["specular_hi"]:
        steps.append("inpaint_specular")
    if q["noise"] > th["noise_hi"]:
        steps.append("denoise")
    if q["contrast"] < th["contrast_lo"]:
        steps.append("clahe")
    return steps


def _inpaint_specular(img):
    m = specular_mask(img).astype(np.uint8)
    if m.sum() == 0 or m.mean() > 0.2:
        return img
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(cv2.inpaint(bgr, m, 3, cv2.INPAINT_TELEA), cv2.COLOR_BGR2RGB)


def _denoise(img):
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    out = cv2.fastNlMeansDenoisingColored(bgr, None, 5, 5, 7, 21)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def _clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


STEPS = {"inpaint_specular": _inpaint_specular, "denoise": _denoise, "clahe": _clahe}


def apply_pipeline(img: np.ndarray, steps: list) -> np.ndarray:
    for s in steps:
        img = STEPS[s](img)
    return img
