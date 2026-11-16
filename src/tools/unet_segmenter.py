"""Enveloppe du U-Net avec l'interface commune segment(img) -> (masque, proba)."""
import cv2
import numpy as np
import torch

from src.tools.uncertainty import mc_dropout_predict
from src.tools.unet import UNet


class UNetSegmenter:
    def __init__(self, ckpt: str = "checkpoints/unet_b1.pt", size: int = 256, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.size = size
        self.model = UNet().to(self.device)
        self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
        self.model.eval()

    def _prep(self, img: np.ndarray):
        h, w = img.shape[:2]
        x = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).float().div(255.0)[None]
        return x.to(self.device), (h, w)

    @torch.no_grad()
    def segment(self, img: np.ndarray):
        """img : uint8 (H, W, 3). Retourne (masque bool, proba float) à la taille de img."""
        x, (h, w) = self._prep(img)
        self.model.eval()
        p = torch.sigmoid(self.model(x))[0, 0].cpu().numpy()
        p = cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)
        return p > 0.5, p

    def segment_mc(self, img: np.ndarray, T: int = 10):
        """Retourne (masque, proba moyenne, écart-type) via MC-Dropout."""
        x, (h, w) = self._prep(img)
        mean_p, std_p = mc_dropout_predict(self.model, x, T)
        mp = cv2.resize(mean_p[0, 0].cpu().numpy(), (w, h), interpolation=cv2.INTER_LINEAR)
        sp = cv2.resize(std_p[0, 0].cpu().numpy(), (w, h), interpolation=cv2.INTER_LINEAR)
        return mp > 0.5, mp, sp
