# AutoResearch report: b1

- Task: det (mAP@0.5); LLM: **qwen3.8-27b-D_gpu5_mtp3** (thinking: xhigh), training budget 180.0 s wall time per run, 5 seeds per confirmed result
- Started 2026-11-02 10:37:26, report written 2026-11-02 15:32:10
- Harness: brief ar2_program_v21.md; acceptance gain > 1.0 SE; second seed for close calls; lessons table in prompt
- Ideas tried: 18 | edits unusable: 1 | crashed/timeout/bad output: 3 | likely bugs (score far below best): 1 | ran but worse: 7 | not confirmed: 5 | **confirmed improvements: 1**
- LLM time: 2.5 h in 20 calls (454 s per call); training time: 2.4 h

| category | ideas | ran | confirmed |
|---|---|---|---|
| data | 6 | 4 | 1 |
| architecture | 4 | 3 | 0 |
| scoring | 3 | 2 | 0 |
| augmentation | 2 | 2 | 0 |
| inference | 1 | 1 | 0 |
| optimisation | 1 | 1 | 0 |
| sampling | 1 | 1 | 0 |

## Final evaluation (hidden test set, read only here)

| version | seeds | val mAP@0.5 | test mAP@0.5 | test gain vs start |
|---|---|---|---|---|
| start file | 8 | 0.6283 ± 0.0018 | 0.6871 ± 0.0015 | - |
| final best (e002) | 8 | 0.6324 ± 0.0018 | 0.6898 ± 0.0014 | +0.0027 ± 0.0020 (+1.3 SE) |

Values are mean ± standard error over seeds. A gain above about 2 SE is unlikely to be seed noise.

Per-class val score, start file: pore 0.504, deposit 0.630, discontinuity 0.780, stain 0.607
Per-class val score, final best e002: pore 0.515, deposit 0.627, discontinuity 0.781, stain 0.611

Chain of accepted changes: e000 -> e001 -> e002

## What changed from the start file to the final best

```diff
--- start/train.py
+++ e002/train.py
@@ -128,7 +128,37 @@
     dets = read_csv(os.path.join(args.data, "dets.csv"))
     ev = [d for d in dets if d["split"] == "eval"]
+    # Label detector boxes on training images by overlap (IoU>0.5) with GT crops, so stage 2 trains on
+    # the exact box distribution it will score (detector boxes are slightly misplaced vs GT boxes).
+    det_by_img, crop_by_img = {}, {}
+    for d in dets:
+        if d["split"] == "train":
+            det_by_img.setdefault(d["image"], []).append(d)
+    for r in crops:
+        crop_by_img.setdefault(r["image"], []).append(r)
+    det_boxes = 0
+    for img_name, gs in crop_by_img.items():
+        ds = det_by_img.get(img_name)
+        if not ds:
+            continue
+        g = np.array([[r["x1"], r["y1"], r["x2"], r["y2"]] for r in gs])
+        c = np.array([r["cls"] for r in gs])
+        d = np.array([[x["x1"], x["y1"], x["x2"], x["y2"]] for x in ds])
+        inter_w = np.maximum(0.0, np.minimum(g[:, None, 2], d[None, :, 2]) - np.maximum(g[:, None, 0], d[None, :, 0]))
+        inter_h = np.maximum(0.0, np.minimum(g[:, None, 3], d[None, :, 3]) - np.maximum(g[:, None, 1], d[None, :, 1]))
+        inter = inter_w * inter_h
+        area_g = (g[:, None, 2] - g[:, None, 0]) * (g[:, None, 3] - g[:, None, 1])
+        area_d = (d[None, :, 2] - d[None, :, 0]) * (d[None, :, 3] - d[None, :, 1])
+        uni = area_g + area_d - inter
+        iou = inter / np.maximum(uni, 1e-9)
+        best = iou.argmax(0)
+        for j in range(len(ds)):
+            if iou[best[j], j] > 0.5:
+                ds[j]["cls"] = int(c[best[j]])
+                det_boxes += 1
+    crops = crops + [d for d in det_by_img.values() for d in d if "cls" in d]
     needed = {r["image"] for r in crops} | {d["image"] for d in ev}
     images = {p: np.asarray(Image.open(os.path.join(args.data, "images", p)).convert("RGB")) for p in needed}
-    print(f"train {len(crops)} boxes, eval {len(ev)} detections, {len(images)} images ({time.time() - T0:.0f}s)", flush=True)
+    print(f"train {len(crops)} boxes ({det_boxes} detector boxes), eval {len(ev)} detections, "
+          f"{len(images)} images ({time.time() - T0:.0f}s)", flush=True)
 
     workers = min(8, os.cpu_count() or 2)
```

## All experiments

| id | base | category | hypothesis | seed-0 | 5-seed mean | outcome |
|---|---|---|---|---|---|---|
| e000 | - | - | starting file | 0.6374 | 0.6302 | baseline |
| e001 | e000 | data | The remaining gap to the 0.758 ceiling is class-decision error (per-class APs 0.50-0.78 vs the ceiling), and the classifier is trained on perfectly placed GT crops while at test time it sees slightly  | - | - | crashed |
| e002 | e001 | data | The previous run crashed on a numpy broadcast error in the IoU computation (it built (n_g, n_d, 2) arrays instead of (n_g, n_d)), so the data idea was never actually tested. I fix the IoU to the corre | 0.6382 | 0.6337 | new best |
| e003 | e002 | architecture | Pore (AP 0.515) is the weakest class because pores are tiny dark holes that are barely resolved when the 1.5x tight crop is downscaled to 160px; rendering the tight view at 224px (EfficientNet-B0's na | - | - | crashed |
| e004 | e003 | architecture | e003 crashed on a trivial NameError (`ctx` is not a parameter of `make_crop`; the parameter is `context`), so the higher-resolution tight crop was never actually tested. I fix the name so the tight vi | 0.6325 | - | rejected |
| e005 | e002 | data | The entire gap to the 0.758 ceiling (0.758 - 0.634 = 0.124) is class-decision error, and the weak classes (pore 0.515, stain 0.611) are the ones whose look depends on surrounding texture that a 160px  | 0.6366 | 0.6315 | not confirmed |
| e006 | e002 | augmentation | The whole 0.124 gap to the 0.758 ceiling is class-decision error, concentrated in the visually similar pore/deposit/stain classes (pore 0.515, stain 0.611, deposit 0.627), and the current recipe only  | 0.6241 | - | rejected |
| e007 | e002 | scoring | The whole 0.124 gap to the 0.758 ceiling is class-decision error, and within each class AP depends only on the rank order of that class's rows, which is currently set by detector conf x raw softmax pr | - | - | timeout |
| e008 | e002 | scoring | Within each class the AP depends only on the rank order, currently set by detector-conf × softmax. The stage-1 conf is class-agnostic and is high for false alarms too, so multiplying it in lets overco | 0.1692 | - | suspicious (likely bug) |
| e009 | e002 | data | Add a 5th "background" training class made of stage-1 detector boxes that do NOT overlap any GT crop (the exact false alarms we see at test time), so the classifier learns to suppress them and the 5-c | 0.5984 | - | rejected |
| e010 | e002 | data | The weak classes (pore 0.515, stain 0.611, deposit 0.627) are limited by training-data diversity: we currently only use "high"-resolution GT crops. Adding the "low"-resolution crops roughly doubles th | 0.6331 | - | rejected |
| e011 | e002 | architecture | The whole 0.124 gap to the 0.758 ceiling is class-decision error (macro-F1 ~0.87 on crops), and the backbone is still the small EfficientNet-B0; swapping to tf_efficientnetv2_b0 (same input size, ~1.3 | 0.6365 | 0.6361 | not confirmed |
| e012 | e002 | scoring | The whole 0.124 gap to the 0.758 ceiling is class-decision error, and within each class AP depends only on rank order, currently set by detector-conf × softmax. The detector conf is class-agnostic and | 0.5708 | - | rejected |
| e013 | e002 | augmentation | The remaining 0.124 gap to the 0.758 ceiling is class-decision error concentrated in the visually similar pore/deposit/stain classes (APs 0.515/0.627/0.611 vs discontinuity 0.781). Mixup (λ~Beta(0.2,0 | 0.6340 | 0.6309 | not confirmed |
| e014 | e002 | architecture | The wide crop (2.5× context) dilutes the signal for small defects like pores (the weakest class, AP 0.515) because the 2.5× crop of a tiny pore box is mostly background, and the equal-weight average w | 0.6243 | - | rejected |
| e015 | e002 | inference | At test time, averaging the class-probability vector over the original and horizontally-flipped crop (2-view TTA) reduces variance in the class decision, which directly targets the class-decision erro | 0.6387 | 0.6310 | not confirmed |
| e016 | e002 | optimisation | The model is undertrained: only ~424 steps / 1.5 epochs fit in the budget and the loss is still falling fast (0.4582 -> 0.2468), so the classifier has not converged and the weak pore/deposit/stain cla | 0.6347 | 0.6299 | not confirmed |
| e017 | e002 | sampling | The 0.124 gap to the 0.758 ceiling is class-decision error concentrated in the weak classes (pore 0.515, stain 0.611, deposit 0.627), and the current recipe trains on every box uniformly, so the optim | 0.6146 | - | rejected |
| e018 | e017 | data | The 0.124 gap to the 0.758 ceiling is class-decision error, and both views are normalized to 160px so the model loses the absolute size of the box, a strong class prior (pores are tiny boxes, deposits | - | - | edit failed |

Each experiment's folder (/data/ar2/runs/b1/eNNN/) holds train.py, diff.patch, the prompt, the LLM's reply and thinking, and the run logs.
