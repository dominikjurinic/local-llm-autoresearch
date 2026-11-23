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
import sys

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
REG_SCALE = 1.25                  # single tight crop the offset head reads
MIN_CROP = 16
REG_ALPHA = 1.0                   # weight of the offset L1 term in the joint loss
REG_CLIP = 0.5                    # clamp predicted offset (in box-size units)
TRAIN_ON = ("high",)    # resolutions of crops.csv used for training
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 0.05
RESERVE_S = 105         # seconds kept free for prediction and writing (offset pass + 4-crop pass ≈ 1.25x pred time)
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


def make_crop(img, r, context, train, cx=None, cy=None):
    h, w = img.shape[:2]
    if cx is None:
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
    """Shared backbone over 4 scale crops, per-scale 1x1 projection, concat, MLP head,
    plus a regression head that predicts a 2D re-centering offset from the tight crop."""

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
        self.reg_hidden = nn.Linear(proj_dim, 128)
        self.reg_head = nn.Linear(128, 2)

    def forward(self, views, reg_view=None):
        feats = []
        for i, x in enumerate(views):
            f = self.backbone.forward_features(x)   # [B, C, H, W]
            f = self.scale_projs[i](f)              # [B, proj_dim, H, W]
            f = f.mean(dim=[2, 3])                  # [B, proj_dim]
            feats.append(f)
        f = torch.cat(feats, dim=1)                 # [B, num_scales * proj_dim]
        logits = self.head(f)
        offset = None
        if reg_view is not None:
            rf = self.backbone.forward_features(reg_view)
            rf = self.scale_projs[0](rf)
            rf = rf.mean(dim=[2, 3])
            offset = self.reg_head(torch.nn.functional.gelu(self.reg_hidden(rf)))
        return logits, offset


class BoxDataset(torch.utils.data.Dataset):
    def __init__(self, rows, images, train, det_by_img=None):
        self.rows, self.images, self.train = rows, images, train
        self.det_by_img = det_by_img if det_by_img is not None else {}
        self.det = None
        self.reg = None
        if self.train:
            self._match_det()

    def _match_det(self):
        # For each GT box, find the best-overlapping train-split detector box in the same image.
        det = []
        reg = []
        for r in self.rows:
            cands = self.det_by_img.get(r["image"], [])
            gx1, gy1, gx2, gy2 = r["x1"], r["y1"], r["x2"], r["y2"]
            gw, gh = max(gx2 - gx1, 1e-6), max(gy2 - gy1, 1e-6)
            gx, gy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
            best_iou, bx = -1.0, None
            for d in cands:
                ix1, iy1 = max(gx1, d["x1"]), max(gy1, d["y1"])
                ix2, iy2 = min(gx2, d["x2"]), min(gy2, d["y2"])
                iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
                inter = iw * ih
                union = gw * gh + (d["x2"] - d["x1"]) * (d["y2"] - d["y1"]) - inter
                iou = inter / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou, bx = iou, d
            if bx is not None:
                det.append(bx)
                dw = max(bx["x2"] - bx["x1"], 1e-6)
                dh = max(bx["y2"] - bx["y1"], 1e-6)
                dx = (gx - (bx["x1"] + bx["x2"]) / 2) / dw
                dy = (gy - (bx["y1"] + bx["y2"]) / 2) / dh
                reg.append(np.array([dx, dy], dtype=np.float32))
            else:  # no overlapping detector box: keep the GT box, zero offset
                det.append(r)
                reg.append(np.zeros(2, dtype=np.float32))
        self.det = det
        self.reg = reg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = self.images[r["image"]]
        fh = fv = False
        bright = 1.0
        if self.train:  # same flips and brightness for all views
            fh, fv, bright = random.random() < 0.5, random.random() < 0.5, random.uniform(0.8, 1.2)
            d = self.det[i]
            cx, cy = (d["x1"] + d["x2"]) / 2, (d["y1"] + d["y2"]) / 2
            reg = self.reg[i].copy()
            if fh:
                reg[0] = -reg[0]
            if fv:
                reg[1] = -reg[1]
        else:
            cx, cy = (r["x1"] + r["x2"]) / 2, (r["y1"] + r["y2"]) / 2
            reg = np.zeros(2, dtype=np.float32)
        views = []
        for ctx in SCALES:
            x = make_crop(img, r, ctx, self.train, cx, cy)
            if fh:
                x = x[:, ::-1]
            if fv:
                x = x[::-1, :]
            if self.train:
                x = np.clip(x * bright, 0, 1)
            x = (x - MEAN) / STD
            views.append(torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))))
        reg_view = make_crop(img, r, REG_SCALE, self.train, cx, cy)
        if fh:
            reg_view = reg_view[:, ::-1]
        if fv:
            reg_view = reg_view[::-1, :]
        if self.train:
            reg_view = np.clip(reg_view * bright, 0, 1)
        reg_view = (reg_view - MEAN) / STD
        reg_view = torch.from_numpy(np.ascontiguousarray(reg_view.transpose(2, 0, 1)))
        return views, r.get("cls", -1), reg_view, torch.from_numpy(reg)


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
    det_by_img = {}
    for d in dets:
        if d["split"] == "train":
            det_by_img.setdefault(d["image"], []).append(d)
    needed = {r["image"] for r in crops} | {d["image"] for d in ev}
    images = {p: np.asarray(Image.open(os.path.join(args.data, "images", p)).convert("RGB")) for p in needed}
    print(f"train {len(crops)} boxes, eval {len(ev)} detections, {len(images)} images ({time.time() - T0:.0f}s)", flush=True)

    workers = min(8, os.cpu_count() or 2)

    def worker_init(_):
        s = torch.initial_seed() % 2**32
        random.seed(s); np.random.seed(s)

    g = torch.Generator().manual_seed(args.seed)
    dl_tr = torch.utils.data.DataLoader(BoxDataset(crops, images, True, det_by_img), batch_size=BATCH_SIZE,
                                        shuffle=True, num_workers=workers, pin_memory=True, drop_last=True,
                                        persistent_workers=True, generator=g, worker_init_fn=worker_init)
    dl_ev = torch.utils.data.DataLoader(BoxDataset(ev, images, False), batch_size=256, shuffle=False,
                                        num_workers=workers, pin_memory=True)

    backbone = timm.create_model(MODEL, pretrained=PRETRAINED, num_classes=None)
    model = MultiScaleClassifier(backbone, num_scales=len(SCALES))
    model = model.to(device).to(memory_format=torch.channels_last)
    crit = nn.CrossEntropyLoss()
    reg_crit = nn.SmoothL1Loss()
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
        for views, y, reg_view, reg in dl_tr:
            frac = min(1.0, (time.time() - t_start) / budget)
            f = frac / 0.05 if frac < 0.05 else 0.5 * (1 + math.cos(math.pi * (frac - 0.05) / 0.95))
            for pg in opt.param_groups:
                pg["lr"] = LR * max(f, 1e-3)
            views = [v.to(device, non_blocking=True).to(memory_format=torch.channels_last) for v in views]
            reg_view = reg_view.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            reg = reg.to(device, non_blocking=True)
            with torch.autocast(**amp):
                logits, offset = model(views, reg_view)
                loss = crit(logits, y) + REG_ALPHA * reg_crit(offset, reg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item(); nb += 1; steps += 1
            if time.time() >= t_end:
                done = True
                break
        print(f"epoch {epoch}: loss {tot / max(nb, 1):.4f}, {steps} steps, {time.time() - T0:.0f}s", flush=True)
        epoch += 1

    # Re-center every eval detection with the predicted offset, then classify.
    model.eval()
    # pass 1: predict the re-centering offset from the detector-centered tight crop
    offsets = []
    with torch.no_grad(), torch.autocast(**amp):
        for views, _, reg_view, _ in dl_ev:
            reg_view = reg_view.to(device).to(memory_format=torch.channels_last)
            _, off = model(None, reg_view)
            offsets.append(off.float().cpu())
    offsets = torch.cat(offsets).numpy()
    for d, off in zip(ev, offsets):  # shift the box by the predicted offset (in box-size units)
        bw = max(d["x2"] - d["x1"], 1e-6)
        bh = max(d["y2"] - d["y1"], 1e-6)
        dx = float(np.clip(off[0], -REG_CLIP, REG_CLIP)) * bw
        dy = float(np.clip(off[1], -REG_CLIP, REG_CLIP)) * bh
        d["x1"] += dx; d["x2"] += dx; d["y1"] += dy; d["y2"] += dy
    # pass 2: classify the re-centered crops
    dl_ev2 = torch.utils.data.DataLoader(BoxDataset(ev, images, False), batch_size=256, shuffle=False,
                                          num_workers=workers, pin_memory=True)
    probs = []
    with torch.no_grad(), torch.autocast(**amp):
        for views, _, reg_view, _ in dl_ev2:
            views = [v.to(device).to(memory_format=torch.channels_last) for v in views]
            logits, _ = model(views, None)
            probs.append(torch.softmax(logits.float(), 1).cpu())
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
