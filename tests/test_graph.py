"""Tests du graphe avec un faux segmenteur (sans torch ni GPU)."""
import numpy as np
from src.agents.graph import build_graph, run_one

H = W = 64


class FakeSeg:
    """Bon si l'image est assez contrastée (std > 5), sinon masque minuscule + forte incertitude."""
    def _ok(self, img):
        return float(img.std()) > 5

    def segment(self, img):
        p = np.zeros((H, W), np.float32)
        if self._ok(img):
            p[20:44, 20:44] = 0.95
        else:
            p[2:4, 2:4] = 0.9
        return p > 0.5, p

    def segment_mc(self, img, T):
        std = np.zeros((H, W), np.float32)
        if not self._ok(img):
            std[:] = 0.3
        return (None, None, std)


def _contrasted():
    img = np.zeros((H, W, 3), np.uint8); img[::2] = 255      # std ~ 127
    return img


def _flat():
    # image plate : CLAHE (tentative 2) la contraste suffisamment pour passer le seuil du faux segmenteur
    rng = np.random.default_rng(0)
    return (120 + rng.integers(0, 6, (H, W, 3))).astype(np.uint8)


def test_accept_directement():
    out = run_one(build_graph(FakeSeg()), _contrasted(), "a")
    assert out["status"] == "accepted" and out["attempt"] == 0
    assert out["final_mask"].sum() > 0


def test_escalade_apres_deux_retries():
    class AlwaysBad(FakeSeg):
        def _ok(self, img): return False
    out = run_one(build_graph(AlwaysBad()), _contrasted(), "b")
    assert out["status"] == "needs_human"
    assert out["attempt"] == 2                                   # max 2 re-segmentations
    assert sum("[critic]" in h for h in out["history"]) == 3
    assert any("[humain]" in h for h in out["history"])
    assert out["final_mask"] is not None                         # l'humain reçoit quand même le meilleur masque


def test_retry_puis_accept_via_clahe():
    out = run_one(build_graph(FakeSeg()), _flat(), "c")
    strategies = [h for h in out["history"] if "[orchestrateur]" in h]
    assert len(strategies) == 3 and "'clahe'" in strategies[-1]
    assert out["status"] == "accepted_after_retry" and out["attempt"] == 2
    assert out["risk"] < 0.15


def test_verite_terrain_absente_de_l_etat():
    out = run_one(build_graph(FakeSeg()), _contrasted(), "d")
    assert not any(k in out for k in ("gt", "mask_gt", "ground_truth"))


def test_base_egale_passage_simple_et_tentatives_enregistrees():
    seg = FakeSeg()
    out = run_one(build_graph(seg), _contrasted(), "e")
    _, p = seg.segment(_contrasted())
    assert np.array_equal(out["attempts"][0]["mask"], p > 0.5)     # base == U-Net seul
    assert len(out["attempts"]) == 1
    bad = run_one(build_graph(seg, {"tau_risk": -1.0}), _contrasted(), "f")
    assert [a["attempt"] for a in bad["attempts"]] == [0, 1, 2]
