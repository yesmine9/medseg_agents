"""Évalue les outils de la semaine 2 sur le split val (jamais sur test).

Produit :
  - Dice moyen : U-Net brut (B1), U-Net + prétraitement fixe (B2),
    U-Net + prétraitement adaptatif, et chaque méthode classique
  - corrélation (Spearman) entre les signaux de confiance (accord entre
    méthodes, incertitude MC-Dropout) et le vrai Dice : c'est la base de
    l'agent Critique de la semaine 3
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm

from src.eval.metrics import dice
from src.seed import set_seed
from src.tools.classical import METHODS, segment_classical
from src.tools.dataset import KvasirSeg
from src.tools.imgutils import tensor_to_uint8
from src.tools.preprocess import FIXED_PIPELINE, apply_pipeline, choose_pipeline, load_thresholds
from src.tools.quality import image_quality
from src.tools.uncertainty import uncertainty_summary
from src.tools.unet_segmenter import UNetSegmenter


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--T", type=int, default=10, help="passes MC-Dropout")
    args = ap.parse_args()

    set_seed(42)
    ds = KvasirSeg(args.split, 256)
    seg = UNetSegmenter()
    th = load_thresholds()

    rows = []
    for i in tqdm(range(len(ds))):
        x, y, stem = ds[i]
        img = tensor_to_uint8(x)
        gt = y[0].numpy() > 0.5
        steps = choose_pipeline(image_quality(img), th)

        m_raw, _ = seg.segment(img)
        m_fix, _ = seg.segment(apply_pipeline(img, FIXED_PIPELINE))
        m_ada, _ = seg.segment(apply_pipeline(img, steps))
        r = {
            "stem": stem,
            "steps": "+".join(steps) or "none",
            "dice_unet_raw": dice(m_raw, gt),
            "dice_unet_fixed": dice(m_fix, gt),
            "dice_unet_adaptive": dice(m_ada, gt),
        }
        agree = []
        for name in METHODS:
            m_c, _ = segment_classical(img, name)
            r[f"dice_{name}"] = dice(m_c, gt)
            agree.append(dice(m_raw, m_c))
        r["agreement"] = float(np.mean(agree))

        _, mean_p, std_p = seg.segment_mc(img, args.T)
        r.update(uncertainty_summary(mean_p, std_p))
        rows.append(r)

    df = pd.DataFrame(rows)
    d = df["dice_unet_raw"]

    def sp(col):
        return float(spearmanr(df[col], d, nan_policy="omit")[0])

    summary = {
        "split": args.split,
        "n": len(df),
        "mean_dice": {c: float(df[c].mean()) for c in df.columns if c.startswith("dice_")},
        "spearman_vs_unet_dice": {c: sp(c) for c in ("agreement", "mc_std_fg", "entropy_fg", "uncertain_ratio")},
        "n_empty_predictions": int(df["empty"].sum()),
        "adaptive_steps_counts": df["steps"].value_counts().to_dict(),
    }
    Path("reports").mkdir(exist_ok=True)
    df.to_csv(f"reports/s2_per_image_{args.split}.csv", index=False)
    Path(f"reports/s2_tools_{args.split}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
