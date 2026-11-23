#!/usr/bin/env python3
"""
FIXED detection metric (the agent may never change this file): mAP@0.5 exactly as YOLOv7's test.py
computes it — per image, predictions sorted by score (at most 300), greedy matching per class with
IoU > 0.5, each true box matched at most once; per class 101-point interpolated AP; mean over the
classes present in the ground truth. Verified against YOLOv7's own code (identical on 30 random cases).
"""
import numpy as np

MAX_DET = 300
trapz = getattr(np, "trapezoid", None) or np.trapz


def iou(a, b):
    """a: n x 4, b: m x 4 (x1, y1, x2, y2) -> n x m."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)


def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [recall[-1] + 0.01]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return trapz(np.interp(x, mrec, mpre), x)


def map50(preds, gts, single_cls=False):
    """preds: image -> list of (cls, score, (x1,y1,x2,y2)); gts: image -> list of (cls, box).
    Every image to be scored must be a key of gts (with an empty list if it has no defects).
    Returns (mAP, {class: AP})."""
    tp, conf, pcls, tcls = [], [], [], []
    for img, g in gts.items():
        gc = [0 if single_cls else c for c, _ in g]
        tcls += gc
        P = sorted(preds.get(img, []), key=lambda p: -p[1])[:MAX_DET]
        if not P:
            continue
        pc = np.array([0 if single_cls else p[0] for p in P])
        correct = np.zeros(len(P), bool)
        if g:
            gb = np.array([b for _, b in g]); pb = np.array([p[2] for p in P]); gca = np.array(gc)
            detected = 0
            for c in np.unique(gca):
                ti = np.nonzero(gca == c)[0]
                pi = np.nonzero(pc == c)[0]
                if len(pi):
                    m = iou(pb[pi], gb[ti])
                    best, arg = m.max(1), m.argmax(1)
                    taken = set()
                    for j in np.nonzero(best > 0.5)[0]:
                        d = ti[arg[j]]
                        if d not in taken:
                            taken.add(d); detected += 1; correct[pi[j]] = True
                            if detected == len(g):
                                break
        tp.append(correct); conf.append([p[1] for p in P]); pcls.append(pc)
    tp = np.concatenate(tp) if tp else np.zeros(0, bool)
    conf = np.concatenate(conf) if conf else np.zeros(0)
    pcls = np.concatenate(pcls) if pcls else np.zeros(0)
    tcls = np.array(tcls)
    i = np.argsort(-conf)
    tp, conf, pcls = tp[i], conf[i], pcls[i]
    aps = {}
    for c in np.unique(tcls):
        sel = pcls == c
        n_l, n_p = (tcls == c).sum(), sel.sum()
        if n_p == 0 or n_l == 0:
            aps[int(c)] = 0.0
            continue
        tpc = tp[sel].astype(float).cumsum()
        fpc = (1 - tp[sel]).astype(float).cumsum()
        aps[int(c)] = float(compute_ap(tpc / (n_l + 1e-16), tpc / (tpc + fpc)))
    return (float(np.mean(list(aps.values()))) if aps else 0.0), aps
