#!/usr/bin/env python3
"""
Do the validation images share welds/specimens with the training images more than the test images do?
CPU only, a few minutes. Two independent checks:

1. File names: original names of the train / val / test images (the authors' split lists). Candidate "specimen"
   keys are built from the name (first k parts, or the name without its trailing number); for each key the script
   counts how many val and how many test images have a training image with the same key.
2. Image content: a 256-bit difference hash (dHash) of every image (high-res train/val/test and the low-res training
   images, all hashed at the same small size, so resolution does not matter). For every val and test image, the
   distance to its nearest training image. Near-identical photos of the same weld have small distances.

If val images sit much closer to the training images than test images do, the validation split rewards
"recognising" known welds, which would explain why wider context wins on validation but not on test.
"""
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

DH = Path("/data/det/high")
INDEX = Path("/data/datasets/lohi-weld/crops_index.csv")


def read_list(p):
    return [l.strip() for l in open(p) if l.strip()]


def dhash(path, n=16):
    with Image.open(path) as im:
        im.draft("L", (n * 8, n * 8))       # fast JPEG decode at reduced size
        g = np.asarray(im.convert("L").resize((n + 1, n), Image.BILINEAR), dtype=np.int16)
    return (g[:, 1:] > g[:, :-1]).ravel()


def name_keys(stem):
    parts = [p for p in re.split(r"[_\-\. ]+", stem) if p]
    keys = {f"first {k} part(s)": "_".join(parts[:k]) for k in range(1, len(parts))}
    keys["name without trailing number"] = re.sub(r"[_\-\. ]*\d+$", "", stem)
    return keys


def main():
    split = {s: read_list(DH / f"{s}.txt") for s in ("train", "val", "test")}
    low_train = []
    if INDEX.exists():
        low_train = sorted({r["image"] for r in csv.DictReader(open(INDEX)) if r["res"] == "low" and r["split"] == "train"})
    print("images: " + ", ".join(f"{s} {len(v)}" for s, v in split.items()) + f", low-res train {len(low_train)}")
    for s, v in split.items():
        print(f"  example {s} names: " + ", ".join(Path(p).name for p in v[:4]))

    # ---- 1. names
    print("\n1. Shared name-based keys with the training images (share of images whose key also occurs in train)")
    stems = {s: [Path(p).stem for p in v] for s, v in split.items()}
    kinds = sorted({k for st in stems["train"][:50] for k in name_keys(st)})
    print(f"  {'key':32} {'groups in train':>16} {'val shares':>11} {'test shares':>12}")
    for kind in kinds:
        tr = {name_keys(st).get(kind) for st in stems["train"]}
        tr.discard(None)
        if len(tr) <= 1:
            continue
        sh = {s: np.mean([name_keys(st).get(kind) in tr for st in stems[s]]) for s in ("val", "test")}
        print(f"  {kind:32} {len(tr):>16} {sh['val']:>11.0%} {sh['test']:>12.0%}")

    # ---- 2. image content
    print("\n2. Nearest training image by content (dHash, 256 bits; 0 = identical, ~128 = unrelated)")
    train_paths = split["train"] + low_train
    H = {}
    for i, p in enumerate(train_paths + split["val"] + split["test"]):
        try:
            H[p] = dhash(p)
        except Exception as e:
            print(f"  cannot read {p}: {e}", file=sys.stderr)
        if i % 300 == 0:
            print(f"  hashed {i} images", flush=True)
    T = np.array([H[p] for p in train_paths if p in H], dtype=np.uint8)
    for s in ("val", "test"):
        Q = np.array([H[p] for p in split[s] if p in H], dtype=np.uint8)
        d = (Q[:, None, :] != T[None, :, :]).sum(2).min(1)
        print(f"  {s:4}: median nearest distance {np.median(d):5.1f} | share with a near-duplicate in train: "
              f"<=10 bits {np.mean(d <= 10):5.1%}, <=25 bits {np.mean(d <= 25):5.1%}, <=40 bits {np.mean(d <= 40):5.1%}")
    print("\nReading: if val has clearly more near-duplicates / shared keys than test, validation is easier for")
    print("models that recognise known welds (supports the leakage hypothesis). Similar numbers: hypothesis unlikely.")


if __name__ == "__main__":
    main()
