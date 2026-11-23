#!/usr/bin/env python3
"""
Image-level uncertainty for mAP@0.5 (harness v3.1). The metric itself stays in det_metric.py (fixed); this file
splits the same computation into per-image matching (done once) and a weighted AP (done per bootstrap sample), so
"resample the images with replacement" costs only weighted cumulative sums.

paired_image_se(new_preds, best_preds, gts): standard error, over images, of
    mean_s mAP(new seed s) - mean_s mAP(best seed s)
with the SAME resampled images for both versions (paired). This is the noise that seed-based standard errors do not
see: a gain that comes from a handful of validation images will not hold on other images.
"""
import numpy as np
from det_metric import iou, compute_ap, MAX_DET


class Parts:
    """Per-row arrays of one prediction set, matched exactly as det_metric.map50 does."""

    def __init__(self, preds, gts):
        self.images = list(gts)
        tp, conf, pcls, img_of = [], [], [], []
        ngt = np.zeros((len(self.images), 4))
        for k, (img, g) in enumerate(gts.items()):
            gc = [c for c, _ in g]
            for c in gc:
                ngt[k, c] += 1
            P = sorted(preds.get(img, []), key=lambda p: -p[1])[:MAX_DET]
            if not P:
                continue
            pc = np.array([p[0] for p in P])
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
            tp.append(correct); conf.append([p[1] for p in P]); pcls.append(pc); img_of.append(np.full(len(P), k))
        tp = np.concatenate(tp) if tp else np.zeros(0, bool)
        conf = np.concatenate(conf) if conf else np.zeros(0)
        pcls = np.concatenate(pcls) if pcls else np.zeros(0, int)
        img_of = np.concatenate(img_of) if img_of else np.zeros(0, int)
        i = np.argsort(-conf)                       # the same call as det_metric.map50 -> the same order
        self.tp, self.pcls, self.img_of = tp[i].astype(float), pcls[i], img_of[i]
        self.ngt = ngt                              # images x 4 true boxes per class
        self.by_cls = {c: np.nonzero(self.pcls == c)[0] for c in range(4)}

    def map(self, w=None):
        """mAP@0.5 with image weights w (counts of a bootstrap sample); w=None -> every image once (= map50)."""
        if w is None:
            w = np.ones(len(self.images))
        n_l = w @ self.ngt
        aps = []
        for c in range(4):
            if n_l[c] <= 0:
                continue                             # class absent from this sample: not part of the mean
            rows = self.by_cls[c]
            wr = w[self.img_of[rows]]
            keep = wr > 0
            if not keep.any():
                aps.append(0.0)
                continue
            t, wr = self.tp[rows][keep], wr[keep]
            tpc = np.cumsum(wr * t)
            fpc = np.cumsum(wr * (1 - t))
            aps.append(float(compute_ap(tpc / (n_l[c] + 1e-16), tpc / (tpc + fpc))))
        return float(np.mean(aps)) if aps else 0.0


def paired_image_se(new_parts, best_parts, n_boot=400, seed=0):
    """Bootstrap SE over images of the gain mean(new) - mean(best); every Parts must share the same image list."""
    n = len(new_parts[0].images)
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    for b in range(n_boot):
        w = rng.multinomial(n, np.full(n, 1.0 / n)).astype(float)
        d[b] = np.mean([p.map(w) for p in new_parts]) - np.mean([p.map(w) for p in best_parts])
    return float(d.std(ddof=1))
