"""Démo : fait tourner le graphe sur quelques images de validation (4 pires + 4 meilleures du U-Net seul)
et compare le Dice final à la vérité terrain. La vérité terrain n'est utilisée QU'ICI, jamais dans le graphe.
Usage : python -m scripts.demo_graph [n]"""
import json
import os
import sys

import pandas as pd

from src.agents.graph import build_graph, run_one
from src.eval.metrics import dice
from src.tools.dataset import KvasirSeg
from src.tools.imgutils import tensor_to_uint8
from src.tools.unet_segmenter import UNetSegmenter


def main(n: int = 4):
    cfg = json.load(open("configs/critic.json")) if os.path.exists("configs/critic.json") else {}
    ds = KvasirSeg("val", 256)
    index = {ip.stem: i for i, (ip, _) in enumerate(ds.items)}
    graph = build_graph(UNetSegmenter(), cfg)

    csv = "reports/s2_per_image_val.csv"
    if os.path.exists(csv):
        df = pd.read_csv(csv).sort_values("dice_unet_raw")
        stems = list(df["stem"].head(n)) + list(df["stem"].tail(n))
        base = dict(zip(df["stem"], df["dice_unet_raw"]))
    else:
        stems, base = list(index)[: 2 * n], {}

    for stem in stems:
        x, y, _ = ds[index[stem]]
        out = run_one(graph, tensor_to_uint8(x), stem)
        d = dice(out["final_mask"], y[0].numpy() > 0.5)
        b = f" (U-Net seul : {base[stem]:.3f})" if stem in base else ""
        print(f"\n=== {stem[:14]} | statut={out['status']} | Dice final={d:.3f}{b}")
        for h in out["history"]:
            print("   ", h)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
