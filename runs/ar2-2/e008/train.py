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
from collections import defaultdict

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
TRAIN_ON = ("high", "low")    # resolutions of crops.csv used for training
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
        if self.train:
            fh, fv, bright = random.random() < 0.5, random.random() < 0.5, random.uniform(0.8, 1.2)
        x = make_crop(img, r, CONTEXT, self.train)
        if fh:
            x = x[:, ::-1]
        if fv:
            x = x[::-1, :]
        if self.train:
            x = np.clip(x * bright, 0, 1)
        x = (x - MEAN) / STD
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))), r.get("cls", -1)


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

    all_crops = read_csv(os.path.join(args.data, "crops.csv"))
    crops = [r for r in all_crops if r["res"] in TRAIN_ON]
    dets = read_csv(os.path.join(args.data, "dets.csv"))
    train_dets = [d for d in dets if d["split"] == "train"]
    ev = [d for d in dets if d["split"] == "eval"]
    crops_by_img = defaultdict(list)
    for r in all_crops:
        crops_by_img[r["image"]].append(r)
    matched_dets = []
    for d in train_dets:
        best_iou, best_cls = 0.0, -1
        for c in crops_by_img.get(d["image"], []):
            x1, y1 = max(d["x1"], c["x1"]), max(d["y1"], c["y1"])
            x2, y2 = min(d["x2"], c["x2"]), min(d["y2"], c["y2"])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            union = (d["x2"]-d["x1"])*(d["y2"]-d["y1"]) + (c["x2"]-c["x1"])*(c["y2"]-c["y1"]) - inter
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou, best_cls = iou, c["cls"]
        if best_iou >= 0.5:
            d["cls"] = best_cls
            matched_dets.append(d)
    train_rows = crops + matched_dets
    needed = {r["image"] for r in train_rows} | {d["image"] for d in ev}
    images = {p: np.asarray(Image.open(os.path.join(args.data, "images", p)).convert("RGB")) for p in needed}
    print(f"train {len(train_rows)} boxes ({len(crops)} gt + {len(matched_dets)} det), eval {len(ev)} detections, {len(images)} images ({time.time() - T0:.0f}s)", flush=True)

    workers = min(8, os.cpu_count() or 2)

    def worker_init(_):
        s = torch.initial_seed() % 2**32
        random.seed(s); np.random.seed(s)

    g = torch.Generator().manual_seed(args.seed)
    dl_tr = torch.utils.data.DataLoader(BoxDataset(train_rows, images, True), batch_size=BATCH_SIZE, shuffle=True,
                                        num_workers=workers, pin_memory=True, drop_last=True,
                                        persistent_workers=True, generator=g, worker_init_fn=worker_init)
    dl_ev = torch.utils.data.DataLoader(BoxDataset(ev, images, False), batch_size=256, shuffle=False,
                                        num_workers=workers, pin_memory=True)

    model = timm.create_model(MODEL, pretrained=PRETRAINED, num_classes=4)
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
        for x, y in dl_tr:
            frac = min(1.0, (time.time() - t_start) / budget)
            f = frac / 0.05 if frac < 0.05 else 0.5 * (1 + math.cos(math.pi * (frac - 0.05) / 0.95))
            for pg in opt.param_groups:
                pg["lr"] = LR * max(f, 1e-3)
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast(**amp):
                loss = crit(model(x), y)
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
        for x, _ in dl_ev:
            x = x.to(device).to(memory_format=torch.channels_last)
            probs.append(torch.softmax(model(x).float(), 1).cpu())
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
