"""Critic : transforme des mesures d'(in)certitude en score de risque.
Fonctions pures (numpy) : faciles à tester et à calibrer sur la validation.
Les poids et seuils par défaut sont PROVISOIRES : à calibrer sur val (phase 5)."""
import numpy as np
from scipy.ndimage import label

DEFAULT_CFG = {
    "mc_T": 10,              # passes MC-Dropout
    "std_thr": 0.10,         # un pixel est "incertain" si son écart-type MC dépasse ce seuil
    "ratio_cap": 1.0,        # plafonne le ratio incertain/aire prédite (évite l'explosion si masque ~vide)
    "tiny_area_frac": 0.002, # masque < 0.2 % de l'image => suspect
    "max_components": 3,     # au-delà, la fragmentation pèse à fond
    "w_unc": 0.5, "w_flip": 0.3, "w_shape": 0.2,
    "tau_risk": 0.15,        # risque >= tau => on ne fait pas confiance
    "max_retries": 2,
}


def dice_bool(a: np.ndarray, b: np.ndarray) -> float:
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else float(2.0 * np.logical_and(a, b).sum() / s)


def critic_features(mask: np.ndarray, std: np.ndarray, flip_dice: float, cfg: dict) -> dict:
    h, w = mask.shape
    area = int(mask.sum())
    unc_px = int((std > cfg["std_thr"]).sum())
    return {
        "area_frac": area / (h * w),
        "unc_ratio": unc_px / max(area, 1),   # relatif à la taille du masque prédit
        "unc_img": unc_px / (h * w),          # relatif à l'image (information)
        "n_components": int(label(mask)[1]),
        "flip_dice": float(flip_dice),        # cohérence entre prédiction normale et retournée
    }


def risk_score(f: dict, cfg: dict) -> float:
    r_unc = min(f["unc_ratio"], cfg["ratio_cap"]) / cfg["ratio_cap"]
    r_flip = 1.0 - f["flip_dice"]
    if f["area_frac"] < cfg["tiny_area_frac"]:
        r_shape = 1.0
    else:
        r_shape = min(1.0, max(0, f["n_components"] - 1) / cfg["max_components"])
    return float(cfg["w_unc"] * r_unc + cfg["w_flip"] * r_flip + cfg["w_shape"] * r_shape)
