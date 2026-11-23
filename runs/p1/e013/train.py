#!/usr/bin/env python3
"""
Two-stage weld defect detection, stage 2: the file the research agent edits.

Stage 1 (fixed, already done): a class-agnostic detector (YOLOv7-tiny) found candidate defect boxes,
each with a confidence ("conf"). Stage 2 (this file): decide the defect type of each box and give
each prediction a score: 0 pore, 1 deposit, 2 discontinuity, 3 stain.
The result is scored like a normal detector: mAP at IoU 0.5 over the 4 classes.

Contract with the harness (do not break it):
  - inputs in <data>:
      crops.csv  labelled training boxes: id, res ("high"/"low"), image, x1, y1, x2, y2, img_w, img_h, cls
      dets.csv   detector boxes: det_id, split ("train" or "eval"), image, x1, y1, x2, y2, img_w, img_h, conf
                 "train" boxes are on training images (their true class can be found by overlap with crops.csv);
                 "eval" boxes are on hidden evaluation images: make predictions for these
      images/    the images
  - write <out>/preds.csv with columns det_id, cls, score: one or more rows per eval box
    (several rows with different classes for the same box are allowed; leaving a box out drops it).
    Higher score = more confident. Per image at most the 300 highest-scoring rows count.
  - use --seed for all randomness
  - finish within --seconds of wall time (loading, training, prediction); the harness kills it at the limit.

Starting point: the "two-crop" classifier (tight crop 1.5x + wide crop 2.5x of the box through the same
EfficientNet-B0, logits averaged), trained on the true training boxes; each eval box becomes one
prediction per class with score = detector conf x class probability.
"""
import time

T0 = time.time()  # the time budget counts from here

import argparse
import csv
import math
import os
import random

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image

# ---------------------------------------------------------------- settings
MODEL = "efficientnet_b0"
PRETRAINED = True
IMG_SIZE = 160
SCALES = (1.25, 1.75, 2.5, 3.5)  # multi-scale crop pyramid
MIN_CROP = 16
TRAIN_ON = ("high",)    # resolutions of crops.csv used for training
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 0.05
RESERVE_S = 85          # seconds kept free for prediction and writing (4 crops ≈ 2x pred time)
MIN_SCORE = 0.001       # rows below this score are not written
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def read_csv(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k in ("x1", "y1", "x2", "y2", "conf"):
            if k in r:
                r[k] = float(r[k])
        if "cls" in r:
            r["cls"] = int(r["cls"])
    return rows


def make_crop(img, r, context, train):
    h, w = img.shape[:2]
    cx, cy = (r["x1"] + r["x2"]) / 2, (r["y1"] + r["y2"]) / 2
    side = max(max(r["x2"] - r["x1"], r["y2"] - r["y1"]) * context, MIN_CROP)
    if train:  # light geometric jitter
        cx += random.uniform(-0.1, 0.1) * side
        cy += random.uniform(-0.1, 0.1) * side
        side *= random.uniform(0.85, 1.2)
    s = max(2, int(round(side)))
    x0, y0 = int(round(cx - s / 2)), int(round(cy - s / 2))
    canvas = np.full((s, s, 3), 114, dtype=np.uint8)  # gray padding outside the image
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(w, x0 + s), min(h, y0 + s)
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return np.asarray(Image.fromarray(canvas).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR), dtype=np.float32) / 255.0


class MultiScaleClassifier(nn.Module):
    """Shared backbone over 4 scale crops, per-scale 1x1 projection, concat, MLP head."""

    def __init__(self, backbone, num_scales=4, proj_dim=256, hidden=512, num_classes=4):
        super().__init__()
        self.backbone = backbone
        C = backbone.num_features
        self.scale_projs = nn.ModuleList([nn.Conv2d(C, proj_dim, 1) for _ in range(num_scales)])
        self.head = nn.Sequential(
            nn.Linear(num_scales * proj_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, views):
        feats = []
        for i, x in enumerate(views):
            f = self.backbone.forward_features(x)   # [B, C, H, W]
            f = self.scale_projs[i](f)              # [B, proj_dim, H, W]
            f = f.mean(dim=[2, 3])                  # [B, proj_dim]
            feats.append(f)
        f = torch.cat(feats, dim=1)                 # [B, num_scales * proj_dim]
        return self.head(f)


class BoxDataset(torch.utils.data.Dataset):
    def __init__(self, rows, images, train):
        self.rows, self.images, self.train = rows, images, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = self.images[r["image"]]
        fh = fv = False
        bright = 1.0
        if self.train:  # same flips and brightness for all views
            fh, fv, bright = random.random() < 0.5, random.random() < 0.5, random.uniform(0.8, 1.2)
        views = []
        for ctx in SCALES:
            x = make_crop(img, r, ctx, self.train)
            if fh:
                x = x[:, ::-1]
            if fv:
                x = x[::-1, :]
            if self.train:
                x = np.clip(x * bright, 0, 1)
            x = (x - MEAN) / STD
            views.append(torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))))
        return views, r.get("cls", -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/work/data")
    ap.add_argument("--out", default="/work/out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=180)
    args = ap.parse_args()
    deadline = T0 + args.seconds
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    crops = [r for r in read_csv(os.path.join(args.data, "crops.csv")) if r["res"] in TRAIN_ON]
    dets = read_csv(os.path.join(args.data, "dets.csv"))
    ev = [d for d in dets if d["split"] == "eval"]

    # Stage-1 detector boxes on the train split, labelled by IoU > 0.5 overlap with a GT crop.
    gt_by_image = {}
    for r in crops:
        gt_by_image.setdefault(r["image"], []).append(r)

    def box_iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        ih = max(0.0, min(ay2, by2) - max(ay1, by1))
        inter = iw * ih
        union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return inter / union if union > 0 else 0.0

    extra = []
    for d in dets:
        if d["split"] != "train":
            continue
        best_iou, best_cls = 0.0, -1
        for g in gt_by_image.get(d["image"], ()):
            v = box_iou((d["x1"], d["y1"], d["x2"], d["y2"]), (g["x1"], g["y1"], g["x2"], g["y2"]))
            if v > best_iou:
                best_iou, best_cls = v, g["cls"]
        if best_iou > 0.5:
            row = dict(d)
            row["cls"] = best_cls
            extra.append(row)
    crops = crops + extra

    needed = {r["image"] for r in crops} | {d["image"] for d in ev}
    images = {p: np.asarray(Image.open(os.path.join(args.data, "images", p)).convert("RGB")) for p in needed}
    print(f"train {len(crops)} boxes ({len(extra)} det boxes), eval {len(ev)} detections, {len(images)} images ({time.time() - T0:.0f}s)", flush=True)

    workers = min(8, os.cpu_count() or 2)

    def worker_init(_):
        s = torch.initial_seed() % 2**32
        random.seed(s); np.random.seed(s)

    g = torch.Generator().manual_seed(args.seed)
    dl_tr = torch.utils.data.DataLoader(BoxDataset(crops, images, True), batch_size=BATCH_SIZE, shuffle=True,
                                        num_workers=workers, pin_memory=True, drop_last=True,
                                        persistent_workers=True, generator=g, worker_init_fn=worker_init)
    dl_ev = torch.utils.data.DataLoader(BoxDataset(ev, images, False), batch_size=256, shuffle=False,
                                        num_workers=workers, pin_memory=True)

    backbone = timm.create_model(MODEL, pretrained=PRETRAINED, num_classes=None)
    model = MultiScaleClassifier(backbone, num_scales=len(SCALES))
    model = model.to(device).to(memory_format=torch.channels_last)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    amp = dict(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")

    # Train until the budget minus the reserve is used up: 5% warm-up, then cosine decay.
    t_start = time.time()
    t_end = deadline - RESERVE_S
    budget = max(1.0, t_end - t_start)
    steps, epoch, done = 0, 0, False
    model.train()
    while not done:
        tot, nb = 0.0, 0
        for views, y in dl_tr:
            frac = min(1.0, (time.time() - t_start) / budget)
            f = frac / 0.05 if frac < 0.05 else 0.5 * (1 + math.cos(math.pi * (frac - 0.05) / 0.95))
            for pg in opt.param_groups:
                pg["lr"] = LR * max(f, 1e-3)
            views = [v.to(device, non_blocking=True).to(memory_format=torch.channels_last) for v in views]
            y = y.to(device, non_blocking=True)
            with torch.autocast(**amp):
                loss = crit(model(views), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item(); nb += 1; steps += 1
            if time.time() >= t_end:
                done = True
                break
        print(f"epoch {epoch}: loss {tot / max(nb, 1):.4f}, {steps} steps, {time.time() - T0:.0f}s", flush=True)
        epoch += 1

    # Classify every eval detection; one prediction per class with score = detector conf x class probability.
    model.eval()
    probs = []
    with torch.no_grad(), torch.autocast(**amp):
        for views, _ in dl_ev:
            views = [v.to(device).to(memory_format=torch.channels_last) for v in views]
            probs.append(torch.softmax(model(views).float(), 1).cpu())
    probs = torch.cat(probs).numpy()
    os.makedirs(args.out, exist_ok=True)
    n = 0
    with open(os.path.join(args.out, "preds.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["det_id", "cls", "score"])
        for d, p in zip(ev, probs):
            for c in range(4):
                s = d["conf"] * float(p[c])
                if s > MIN_SCORE:
                    w.writerow([d["det_id"], c, f"{s:.6f}"]); n += 1
    print(f"done: {epoch} epochs, {steps} steps, {n} prediction rows, {time.time() - T0:.0f}s total", flush=True)


if __name__ == "__main__":
    main()
