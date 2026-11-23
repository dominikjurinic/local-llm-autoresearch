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
CONTEXT = 2.5           # wide crop: side = CONTEXT * long side of the box
TIGHT_CONTEXT = 1.5     # tight crop
MIN_CROP = 16
TRAIN_ON = ("high",)    # resolutions of crops.csv used for training
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 0.05
RESERVE_S = 55          # seconds kept free for prediction and writing
MIN_SCORE = 0.001       # rows below this score are not written
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def read_csv(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k in ("x1", "y1", "x2", "y2", "conf", "img_w", "img_h"):
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


class BoxDataset(torch.utils.data.Dataset):
    def __init__(self, rows, images, train, meta_mean=None, meta_std=None):
        self.rows, self.images, self.train = rows, images, train
        # Compute box metadata: log relative area, log aspect ratio
        self.meta = []
        for r in rows:
            w = max(1.0, r["x2"] - r["x1"])
            h = max(1.0, r["y2"] - r["y1"])
            img_area = r["img_w"] * r["img_h"]
            box_area = w * h
            rel_area = box_area / img_area
            aspect = w / h
            self.meta.append([math.log(rel_area + 1e-8), math.log(aspect)])
        self.meta = np.array(self.meta, dtype=np.float32)
        if train:
            self.meta_mean = self.meta.mean(0)
            self.meta_std = self.meta.std(0) + 1e-6
        else:
            self.meta_mean = meta_mean
            self.meta_std = meta_std
        self.meta = (self.meta - self.meta_mean) / self.meta_std

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = self.images[r["image"]]
        fh = fv = False
        bright = 1.0
        if self.train:  # same flips and brightness for both views
            fh, fv, bright = random.random() < 0.5, random.random() < 0.5, random.uniform(0.8, 1.2)
        views = []
        for ctx in (TIGHT_CONTEXT, CONTEXT):
            x = make_crop(img, r, ctx, self.train)
            if fh:
                x = x[:, ::-1]
            if fv:
                x = x[::-1, :]
            if self.train:
                x = np.clip(x * bright, 0, 1)
            x = (x - MEAN) / STD
            views.append(torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))))
        meta = torch.from_numpy(self.meta[i])
        return views[0], views[1], meta, r.get("cls", -1)


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
    # Add detector boxes from train split matched to GT crops (IoU > 0.5)
    det_train = [d for d in dets if d["split"] == "train"]
    # Build spatial index of crops per image
    crops_by_img = {}
    for c in crops:
        crops_by_img.setdefault(c["image"], []).append(c)
    def iou(a, b):
        ax1, ay1, ax2, ay2 = a["x1"], a["y1"], a["x2"], a["y2"]
        bx1, by1, bx2, by2 = b["x1"], b["y1"], b["x2"], b["y2"]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        aa = (ax2 - ax1) * (ay2 - ay1)
        bb = (bx2 - bx1) * (by2 - by1)
        return inter / (aa + bb - inter)
    for d in det_train:
        if d["image"] not in crops_by_img:
            continue
        best_iou, best_cls = 0.0, -1
        for c in crops_by_img[d["image"]]:
            v = iou(d, c)
            if v > best_iou:
                best_iou, best_cls = v, c["cls"]
        if best_iou > 0.5 and best_cls >= 0:
            d["cls"] = best_cls
            crops.append(d)
    needed = {r["image"] for r in crops} | {d["image"] for d in ev}
    images = {p: np.asarray(Image.open(os.path.join(args.data, "images", p)).convert("RGB")) for p in needed}
    print(f"train {len(crops)} boxes (incl. matched det), eval {len(ev)} detections, {len(images)} images ({time.time() - T0:.0f}s)", flush=True)

    workers = min(8, os.cpu_count() or 2)

    def worker_init(_):
        s = torch.initial_seed() % 2**32
        random.seed(s); np.random.seed(s)

    g = torch.Generator().manual_seed(args.seed)
    ds_tr = BoxDataset(crops, images, True)
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,
                                        num_workers=workers, pin_memory=True, drop_last=True,
                                        persistent_workers=True, generator=g, worker_init_fn=worker_init)
    ds_ev = BoxDataset(ev, images, False, meta_mean=ds_tr.meta_mean, meta_std=ds_tr.meta_std)
    dl_ev = torch.utils.data.DataLoader(ds_ev, batch_size=256, shuffle=False,
                                        num_workers=workers, pin_memory=True)

    # Build model with metadata MLP head
    backbone = timm.create_model(MODEL, pretrained=PRETRAINED, num_classes=0)
    feat_dim = backbone.num_features  # 1280 for efficientnet_b0
    meta_dim = 2  # log relative area, log aspect ratio
    hidden_dim = 64
    
    class MetaHead(nn.Module):
        def __init__(self, feat_dim, meta_dim, hidden_dim, num_classes):
            super().__init__()
            self.meta_mlp = nn.Sequential(
                nn.Linear(meta_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.classifier = nn.Linear(feat_dim + hidden_dim, num_classes)
        
        def forward(self, feat, meta):
            meta_feat = self.meta_mlp(meta)
            x = torch.cat([feat, meta_feat], dim=1)
            return self.classifier(x)
    
    model = nn.Sequential(backbone, MetaHead(feat_dim, meta_dim, hidden_dim, 4))
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
        for xt, xw, meta, y in dl_tr:
            frac = min(1.0, (time.time() - t_start) / budget)
            f = frac / 0.05 if frac < 0.05 else 0.5 * (1 + math.cos(math.pi * (frac - 0.05) / 0.95))
            for pg in opt.param_groups:
                pg["lr"] = LR * max(f, 1e-3)
            xt = xt.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            xw = xw.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            meta = meta.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(**amp):
                logits_t = model[1](model[0](xt), meta)
                logits_w = model[1](model[0](xw), meta)
                loss = crit((logits_t + logits_w) / 2, y)
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
        for xt, xw, meta, _ in dl_ev:
            xt = xt.to(device).to(memory_format=torch.channels_last)
            xw = xw.to(device).to(memory_format=torch.channels_last)
            meta = meta.to(device)
            logits_t = model[1](model[0](xt), meta)
            logits_w = model[1](model[0](xw), meta)
            probs.append(torch.softmax(((logits_t + logits_w) / 2).float(), 1).cpu())
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
