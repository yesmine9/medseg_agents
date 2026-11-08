import numpy as np

from src.eval.metrics import dice, hd95, iou


def test_perfect_overlap():
    m = np.zeros((32, 32), bool)
    m[8:20, 8:20] = True
    assert dice(m, m) > 0.999 and iou(m, m) > 0.999 and hd95(m, m) == 0


def test_disjoint():
    a = np.zeros((32, 32), bool)
    a[:5, :5] = True
    b = np.zeros((32, 32), bool)
    b[20:25, 20:25] = True
    assert dice(a, b) < 1e-3
