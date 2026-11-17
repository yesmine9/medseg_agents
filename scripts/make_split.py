import json
import random
from pathlib import Path

RAW = Path("data/raw/kvasir-seg")
OUT = Path("data/splits.json")
EXT = {".jpg", ".jpeg", ".png"}


def stems(folder: Path) -> set:
    return {p.stem for p in folder.iterdir() if p.suffix.lower() in EXT}


def main(seed: int = 42) -> None:
    imgs, masks = stems(RAW / "images"), stems(RAW / "masks")
    missing = imgs - masks
    assert not missing, f"Masques manquants : {sorted(missing)[:5]}"
    names = sorted(imgs)
    random.Random(seed).shuffle(names)
    n = len(names)
    n_tr, n_va = int(0.70 * n), int(0.15 * n)
    splits = {
        "train": names[:n_tr],
        "val": names[n_tr:n_tr + n_va],
        "test": names[n_tr + n_va:],
    }
    OUT.write_text(json.dumps(splits, indent=2))
    print({k: len(v) for k, v in splits.items()})


if __name__ == "__main__":
    main()
