import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def dice(pred, gt, eps=1e-7):
    inter = (pred & gt).sum()
    return float((2 * inter + eps) / (pred.sum() + gt.sum() + eps))


def iou(pred, gt, eps=1e-7):
    inter = (pred & gt).sum()
    return float((inter + eps) / ((pred | gt).sum() + eps))


def _surface(m):
    return m & ~binary_erosion(m)


def hd95(pred, gt):
    if not pred.any() or not gt.any():
        return float("nan")
    sp, sg = _surface(pred), _surface(gt)
    d1 = distance_transform_edt(~sg)[sp]
    d2 = distance_transform_edt(~sp)[sg]
    return float(np.percentile(np.hstack([d1, d2]), 95))
