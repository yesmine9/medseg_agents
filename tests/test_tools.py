import numpy as np
import pytest

from src.eval.metrics import dice
from src.tools.classical import METHODS, segment_classical
from src.tools.preprocess import FIXED_PIPELINE, apply_pipeline, choose_pipeline
from src.tools.quality import image_quality


def synthetic(seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((128, 128, 3), (120, 60, 60), np.uint8)
    yy, xx = np.ogrid[:128, :128]
    disc = (yy - 64) ** 2 + (xx - 64) ** 2 < 25 ** 2
    img[disc] = (230, 140, 130)
    img = np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
    return img, disc


def test_quality_keys_and_ranges():
    img, _ = synthetic()
    q = image_quality(img)
    assert set(q) == {"noise", "contrast", "sharpness", "specular", "brightness"}
    assert q["noise"] >= 0 and 0 <= q["specular"] <= 1


def test_pipeline_preserves_shape_and_dtype():
    img, _ = synthetic()
    out = apply_pipeline(img, FIXED_PIPELINE + ["inpaint_specular"])
    assert out.shape == img.shape and out.dtype == np.uint8


def test_choose_pipeline_rules():
    th = {"noise_hi": 3.0, "contrast_lo": 35.0, "specular_hi": 0.01}
    q = {"noise": 5.0, "contrast": 20.0, "specular": 0.05}
    assert choose_pipeline(q, th) == ["inpaint_specular", "denoise", "clahe"]
    q = {"noise": 1.0, "contrast": 60.0, "specular": 0.0}
    assert choose_pipeline(q, th) == []


def test_otsu_finds_disc():
    img, disc = synthetic()
    mask, proba = segment_classical(img, "otsu")
    assert mask.dtype == bool and mask.shape == disc.shape
    assert dice(mask, disc) > 0.8
    assert proba.min() >= 0 and proba.max() <= 1


@pytest.mark.parametrize("name", list(METHODS))
def test_classical_interface(name):
    img, disc = synthetic()
    mask, proba = segment_classical(img, name)
    assert mask.shape == disc.shape and mask.dtype == bool
    assert proba.shape == disc.shape


def test_mc_dropout_has_variance():
    torch = pytest.importorskip("torch")
    from src.tools.uncertainty import mc_dropout_predict
    from src.tools.unet import UNet

    model = UNet()
    x = torch.rand(1, 3, 64, 64)
    mean_p, std_p = mc_dropout_predict(model, x, T=5)
    assert mean_p.shape == (1, 1, 64, 64)
    assert float(std_p.max()) > 0
    assert not model.training
