"""Évaluation du système sur la VALIDATION (jamais le test avant la semaine 5).

Chaque image passe par les 3 tentatives (tau forcé à -1) ; on enregistre risque et Dice de chaque
tentative, puis on SIMULE hors ligne n'importe quel seuil tau : B3 (sans critic) et B4 (avec critic).
La vérité terrain ne sert qu'ici, pour mesurer.
Sorties : reports/s3_attempts_val.csv, reports/s3_system_val.json
Usage : python -m scripts.eval_system"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.agents.graph import build_graph, run_one
from src.eval.metrics import dice
from src.tools.dataset import KvasirSeg
from src.tools.imgutils import tensor_to_uint8
from src.tools.unet_segmenter import UNetSegmenter

K = 3  # tentatives par image


def collect():
    cfg = json.load(open("configs/critic.json"))
    graph = build_graph(UNetSegmenter(), {**cfg, "tau_risk": -1.0, "max_retries": K - 1})
    ds = KvasirSeg("val", 256)
    n, rows = len(ds.items), []
    for i in range(n):
        x, y, stem = ds[i]
        gt = y[0].numpy() > 0.5
        out = run_one(graph, tensor_to_uint8(x), stem)
        row = {"stem": stem}
        for a in out["attempts"]:
            row[f"dice_a{a['attempt']}"] = dice(a["mask"], gt)
            row[f"risk_a{a['attempt']}"] = a["risk"]
        f0 = out["attempts"][0]["features"]
        row.update(unc_ratio=f0["unc_ratio"], flip_dice=f0["flip_dice"],
                   n_components=f0["n_components"], area_frac=f0["area_frac"])
        rows.append(row)
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{n}", flush=True)
    return pd.DataFrame(rows), cfg


def simulate(df, tau):
    R = df[[f"risk_a{k}" for k in range(K)]].to_numpy()
    D = df[[f"dice_a{k}" for k in range(K)]].to_numpy()
    acc, final_d, retried = [], [], []
    for r, d in zip(R, D):
        ok = np.where(r < tau)[0]
        if len(ok):
            acc.append(True); final_d.append(d[ok[0]]); retried.append(ok[0] > 0)
        else:
            acc.append(False); final_d.append(d[int(np.argmin(r))]); retried.append(False)
    acc, final_d = np.array(acc), np.array(final_d)
    return {"tau": tau, "couverture": float(acc.mean()),
            "dice_acceptes": float(final_d[acc].mean()) if acc.any() else float("nan"),
            "dice_tous_B4": float(final_d.mean()),
            "escalades": int((~acc).sum()), "acceptes_apres_retry": int(np.sum(retried))}


def main():
    df, cfg = collect()
    os.makedirs("reports", exist_ok=True)
    df.to_csv("reports/s3_attempts_val.csv", index=False)

    fail = (df["dice_a0"] < 0.7).astype(int)
    auroc = {"risque_critic": roc_auc_score(fail, df["risk_a0"]),
             "incertitude_seule": roc_auc_score(fail, df["unc_ratio"]),
             "incoherence_flip_seule": roc_auc_score(fail, 1 - df["flip_dice"]),
             "nb_composantes_seul": roc_auc_score(fail, df["n_components"])}
    best = df[[f"dice_a{k}" for k in range(K)]].max(axis=1)
    summary = {
        "n": len(df), "n_echecs_dice<0.7": int(fail.sum()),
        "B1_U-Net_seul_(tentative_0)": float(df["dice_a0"].mean()),
        "B3_ref_attendue_B1": 0.8355,
        "oracle_meilleure_tentative": float(best.mean()),
        "AUROC_echec": {k: float(v) for k, v in auroc.items()},
        "balayage_tau": [simulate(df, t) for t in
                         [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.60]],
        "tau_config": cfg["tau_risk"], "simulation_au_tau_config": simulate(df, cfg["tau_risk"]),
    }
    json.dump(summary, open("reports/s3_system_val.json", "w"), indent=2)

    print(f"\nB1 / B3 (tentative 0, sans critic) : Dice {summary['B1_U-Net_seul_(tentative_0)']:.4f}  (attendu ~0.8355)")
    print(f"Oracle (meilleure des 3 tentatives, vérité terrain connue) : {summary['oracle_meilleure_tentative']:.4f}")
    print(f"Échecs (Dice<0.7) : {summary['n_echecs_dice<0.7']}/{summary['n']}")
    print("AUROC détection d'échec :", {k: round(v, 3) for k, v in summary["AUROC_echec"].items()})
    print("\nBalayage du seuil tau (B4) :")
    print(pd.DataFrame(summary["balayage_tau"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
