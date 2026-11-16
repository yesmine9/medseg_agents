"""Calcule les seuils du prétraitement adaptatif à partir du split TRAIN."""
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.tools.dataset import KvasirSeg
from src.tools.imgutils import tensor_to_uint8
from src.tools.quality import image_quality


def main() -> None:
    ds = KvasirSeg("train", 256)
    rows = [image_quality(tensor_to_uint8(ds[i][0])) for i in tqdm(range(len(ds)))]
    arr = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    for k, v in arr.items():
        pct = {p: round(float(np.percentile(v, p)), 3) for p in (5, 15, 50, 85, 95)}
        print(f"{k:12s}", pct)
    th = {
        "noise_hi": float(np.percentile(arr["noise"], 85)),
        "contrast_lo": float(np.percentile(arr["contrast"], 15)),
        "specular_hi": float(max(np.percentile(arr["specular"], 85), 0.005)),
    }
    Path("configs").mkdir(exist_ok=True)
    Path("configs/quality_thresholds.json").write_text(json.dumps(th, indent=2))
    print("seuils :", th)


if __name__ == "__main__":
    main()
