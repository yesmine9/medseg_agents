import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

EXT = {".jpg", ".jpeg", ".png"}
ROOT = Path("data/raw/kvasir-seg")


def _index(folder: Path) -> dict:
    return {p.stem: p for p in folder.iterdir() if p.suffix.lower() in EXT}


class KvasirSeg(Dataset):
    def __init__(self, split, size=256, augment=False,
                 root=ROOT, splits_file="data/splits.json"):
        stems = json.loads(Path(splits_file).read_text())[split]
        imgs, masks = _index(Path(root) / "images"), _index(Path(root) / "masks")
        self.items = [(imgs[s], masks[s]) for s in stems]
        self.size, self.augment = size, augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ip, mp = self.items[i]
        img = cv2.cvtColor(cv2.imread(str(ip)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        if self.augment:
            if np.random.rand() < 0.5:
                img, mask = img[:, ::-1], mask[:, ::-1]
            if np.random.rand() < 0.5:
                img, mask = img[::-1], mask[::-1]
        img = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy((np.ascontiguousarray(mask) > 127).astype(np.float32))[None]
        return img, mask, ip.stem
