#!/usr/bin/env python3
"""
Phase 4 data: the label-hidden folder for AutoResearch on two-stage detection.

Creates <root>/data/ (visible to the agent's code)
  images/imgNNNNN.jpg   high-res images (train, val, test) + low-res training images, hard links,
                        renamed in random order
  crops.csv             ground-truth training boxes WITH labels (high + low res):
                        id, res, image, x1, y1, x2, y2, img_w, img_h, cls
  dets.csv              class-agnostic detector (D2) boxes: det_id, split, image, x1, y1, x2, y2, img_w, img_h, conf
                        split "train" = training images (labels can be matched from crops.csv),
                        split "eval"  = validation + test images mixed, no labels anywhere
and <root>/private/ (harness only)
  gt.csv                true boxes of the eval images (image, cls, x1, y1, x2, y2)
  images.csv            which eval image is val and which is test
  reference.json        reference numbers on VALIDATION (shown to the agent)

Usage:  python ar2_prepare.py
"""
import argparse
import csv
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

from det_metric import iou, map50

IMG_SIZE = 1280  # size the detectors ran at


def read_list(p):
    return [Path(l.strip()) for l in open(p) if l.strip()]


def yolo_x_factor(W0, H0, img_size=1280):
    """YOLOv7 saves predicted x coordinates stretched by (w/w0)/(h/h0) (it rescales with the height ratio only);
    multiplying by this factor puts them back on the image (see ts_score.py)."""
    r = img_size / max(H0, W0)
    if r == 1:
        return 1.0
    return (int(H0 * r) / H0) / (int(W0 * r) / W0)


def yolo_boxes(path, W, H, with_conf):
    """with_conf=True: YOLOv7 predictions (x stretch undone); False: label files (exact)."""
    k = yolo_x_factor(W, H, IMG_SIZE) if with_conf else 1.0
    out = []
    if path.exists():
        for line in open(path):
            p = line.split()
            if len(p) != (6 if with_conf else 5):
                continue
            c, cx, cy, w, h = int(float(p[0])), *map(float, p[1:5])
            cx, w = cx * k, w * k
            box = ((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H)
            out.append((c, box, float(p[5]) if with_conf else 1.0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="/data/datasets/lohi-weld/crops_index.csv")
    ap.add_argument("--det-high", default="/data/det/high")
    ap.add_argument("--preds", default="/data/det/preds")
    ap.add_argument("--root", default="/data/ar2")
    ap.add_argument("--min-conf", type=float, default=0.005)
    ap.add_argument("--top", type=int, default=100, help="max detections kept per image")
    ap.add_argument("--seed", type=int, default=2027)
    args = ap.parse_args()
    root, dh, pr = Path(args.root), Path(args.det_high), Path(args.preds)
    data, private = root / "data", root / "private"
    if (data / "dets.csv").exists():
        raise SystemExit(f"{data / 'dets.csv'} already exists. Delete {data} and {private} to rebuild.")
    (data / "images").mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    os.chmod(private, 0o700)
    rng = random.Random(args.seed)

    split_imgs = {s: read_list(dh / f"{s}.txt") for s in ("train", "val", "test")}
    for s in ("train", "val", "test"):
        if not (pr / f"d2-agnostic-{s}" / "labels").exists():
            raise SystemExit(f"missing D2 detections for {s}: {pr / f'd2-agnostic-{s}'} (ar2_setup.sh creates them)")
    rows = list(csv.DictReader(open(args.index)))
    low_train = sorted({r["image"] for r in rows if r["res"] == "low" and r["split"] == "train"})
    all_imgs = [str(p) for s in split_imgs.values() for p in s] + low_train
    rng.shuffle(all_imgs)
    new = {}
    for i, src in enumerate(all_imgs):
        dst = data / "images" / f"img{i:05d}.jpg"
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        new[src] = dst.name
    size = {}
    for src in all_imgs:
        with Image.open(src) as im:
            size[src] = im.size
    # high-res images in the detector folder are hard links named like the originals: map by stem
    by_stem_high = {Path(p).stem: str(p) for s in split_imgs.values() for p in s}

    # crops.csv: ground-truth training boxes (high from the detector labels, low from the crop index)
    crops = []
    for p in split_imgs["train"]:
        W, H = size[str(p)]
        for c, b, _ in yolo_boxes(dh / "labels" / f"{p.stem}.txt", W, H, False):
            crops.append(("high", new[str(p)], *b, W, H, c))
    for r in rows:
        if r["res"] == "low" and r["split"] == "train":
            W, H = size[r["image"]]
            crops.append(("low", new[r["image"]], float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"]), W, H,
                          int(r["cls"])))
    rng.shuffle(crops)
    with open(data / "crops.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "res", "image", "x1", "y1", "x2", "y2", "img_w", "img_h", "cls"])
        for i, c in enumerate(crops):
            w.writerow([f"g{i:05d}", c[0], c[1], *(round(v, 1) for v in c[2:6]), c[6], c[7], c[8]])

    # dets.csv + private ground truth
    dets, gt_rows, img_split = [], [], []
    for s in ("train", "val", "test"):
        for p in split_imgs[s]:
            W, H = size[str(p)]
            ds = yolo_boxes(pr / f"d2-agnostic-{s}" / "labels" / f"{p.stem}.txt", W, H, True)
            ds = sorted([d for d in ds if d[2] >= args.min_conf], key=lambda d: -d[2])[:args.top]
            for _, b, cf in ds:
                dets.append(("train" if s == "train" else "eval", new[str(p)], *b, W, H, cf, s))
            if s != "train":
                img_split.append((new[str(p)], s))
                for c, b, _ in yolo_boxes(dh / "labels" / f"{p.stem}.txt", W, H, False):
                    gt_rows.append((new[str(p)], c, *b))
    rng.shuffle(dets)
    with open(data / "dets.csv", "w", newline="") as f, open(private / "dets_split.csv", "w", newline="") as g:
        w, w2 = csv.writer(f), csv.writer(g)
        w.writerow(["det_id", "split", "image", "x1", "y1", "x2", "y2", "img_w", "img_h", "conf"])
        w2.writerow(["det_id", "split"])
        for i, d in enumerate(dets):
            did = f"d{i:06d}"
            w.writerow([did, d[0], d[1], *(round(v, 2) for v in d[2:6]), d[6], d[7], f"{d[8]:.5g}"])
            w2.writerow([did, d[9]])
    with open(private / "gt.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "cls", "x1", "y1", "x2", "y2"])
        w.writerows((g[0], g[1], *(round(v, 2) for v in g[2:])) for g in gt_rows)
    with open(private / "images.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "split"])
        w.writerows(img_split)
    for p in private.iterdir():
        os.chmod(p, 0o600)

    # reference numbers on validation
    gts = defaultdict(list)
    for g in gt_rows:
        gts[g[0]].append((g[1], tuple(g[2:])))
    val_imgs = [im for im, s in img_split if s == "val"]
    G = {im: gts.get(im, []) for im in val_imgs}
    d1 = defaultdict(list)
    for p in split_imgs["val"]:
        W, H = size[str(p)]
        for c, b, cf in yolo_boxes(pr / "d1-fine-val" / "labels" / f"{p.stem}.txt", W, H, True):
            d1[new[str(p)]].append((c, cf, b))
    oracle, d2only = defaultdict(list), defaultdict(list)
    for d in dets:
        if d[9] != "val":
            continue
        box, g = tuple(d[2:6]), G.get(d[1], [])
        k = 0
        if g:
            m = iou([box], [b for _, b in g])[0]
            if m.max() > 0.5:
                k = g[int(m.argmax())][0]
        oracle[d[1]].append((k, d[8], box))
        d2only[d[1]].append((0, d[8], box))
    ref = {"d1_val_map50": map50(d1, G)[0] if d1 else None,
           "d2_boxes_perfect_classes_val_map50": map50(oracle, G)[0],
           "d2_class_agnostic_val_map50": map50(d2only, G, single_cls=True)[0]}
    (private / "reference.json").write_text(json.dumps(ref, indent=1))
    os.chmod(private / "reference.json", 0o600)
    n_tr = sum(1 for d in dets if d[0] == "train"); n_ev = len(dets) - n_tr
    print(f"images: {len(all_imgs)} -> {data / 'images'}")
    print(f"crops.csv: {len(crops)} labelled training boxes (high {sum(c[0] == 'high' for c in crops)}, "
          f"low {sum(c[0] == 'low' for c in crops)})")
    print(f"dets.csv: {n_tr} detections on training images, {n_ev} on eval images (val+test, no labels); "
          f"conf >= {args.min_conf}, at most {args.top} per image")
    print("reference (validation): " + ", ".join(f"{k} {v:.3f}" for k, v in ref.items() if v is not None))


if __name__ == "__main__":
    main()
