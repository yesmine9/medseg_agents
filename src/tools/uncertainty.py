"""Incertitude par MC-Dropout (utilisée par l'agent Critique)."""
import numpy as np
import torch
import torch.nn as nn


def enable_mc_dropout(model: nn.Module) -> None:
    """Modèle en eval (BatchNorm figée) mais Dropout actif."""
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()


@torch.no_grad()
def mc_dropout_predict(model: nn.Module, x: torch.Tensor, T: int = 10):
    """x : (B, 3, H, W). Retourne (moyenne, écart-type) des probabilités, (B, 1, H, W)."""
    enable_mc_dropout(model)
    ps = torch.stack([torch.sigmoid(model(x)) for _ in range(T)])
    model.eval()
    return ps.mean(0), ps.std(0)


def uncertainty_summary(mean_p: np.ndarray, std_p: np.ndarray) -> dict:
    """Résume la carte d'incertitude en quelques scalaires (H, W numpy)."""
    fg = mean_p > 0.5
    area = int(fg.sum())
    if area == 0:
        return {"mc_std_fg": float("nan"), "entropy_fg": float("nan"),
                "uncertain_ratio": float("nan"), "empty": True}
    p = np.clip(mean_p, 1e-6, 1 - 1e-6)
    ent = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    return {
        "mc_std_fg": float(std_p[fg].mean()),
        "entropy_fg": float(ent[fg].mean()),
        "uncertain_ratio": float(((mean_p > 0.2) & (mean_p < 0.8)).sum() / area),
        "empty": False,
    }
