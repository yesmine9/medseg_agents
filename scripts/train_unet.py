import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.eval.metrics import dice, hd95, iou
from src.seed import set_seed
from src.tools.dataset import KvasirSeg
from src.tools.unet import UNet


def dice_loss(logits, target, eps=1e-6):
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    return 1 - ((2 * inter + eps) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + eps)).mean()


@torch.no_grad()
def evaluate(model, loader, device, full=False):
    model.eval()
    d, i, h = [], [], []
    for x, y, _ in loader:
        pred = (torch.sigmoid(model(x.to(device)).float()) > 0.5).cpu().numpy()[:, 0]
        gt = y.numpy()[:, 0] > 0.5
        for p, g in zip(pred, gt):
            d.append(dice(p, g))
            if full:
                i.append(iou(p, g))
                h.append(hd95(p, g))
    res = {"dice": float(np.mean(d))}
    if full:
        res["iou"] = float(np.mean(i))
        res["hd95"] = float(np.nanmean(h))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    tr = DataLoader(KvasirSeg("train", args.size, augment=True), batch_size=args.bs,
                    shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    va = DataLoader(KvasirSeg("val", args.size), batch_size=args.bs, num_workers=2)

    model = UNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=4)
    bce = nn.BCEWithLogitsLoss()
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    Path("checkpoints").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    best, bad = -1.0, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y, _ in tqdm(tr, desc=f"epoch {ep}", leave=False):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                out = model(x)
            out = out.float()
            loss = bce(out, y) + dice_loss(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        m = evaluate(model, va, device)["dice"]
        sched.step(m)
        print(f"epoch {ep:02d} | loss {np.mean(losses):.4f} | val dice {m:.4f}")
        if m > best:
            best, bad = m, 0
            torch.save(model.state_dict(), "checkpoints/unet_b1.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print("early stopping")
                break

    model.load_state_dict(torch.load("checkpoints/unet_b1.pt", map_location=device))
    res = evaluate(model, va, device, full=True)
    Path("reports/b1_val.json").write_text(json.dumps(res, indent=2))
    print("B1 (val):", res)


if __name__ == "__main__":
    main()
