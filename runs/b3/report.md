# AutoResearch report: b3

- Task: det (mAP@0.5); LLM: **qwen3.8-27b-D_gpu5_mtp3** (thinking: xhigh), training budget 180.0 s wall time per run, 5 seeds per confirmed result
- Started 2026-11-02 20:23:23, report written 2026-11-03 01:08:08
- Harness: brief ar2_program_v21.md; acceptance gain > 1.0 SE; second seed for close calls; lessons table in prompt
- Ideas tried: 18 | edits unusable: 0 | crashed/timeout/bad output: 1 | likely bugs (score far below best): 1 | ran but worse: 11 | not confirmed: 4 | **confirmed improvements: 1**
- LLM time: 2.3 h in 19 calls (445 s per call); training time: 2.4 h

| category | ideas | ran | confirmed |
|---|---|---|---|
| data | 9 | 8 | 1 |
| scoring | 3 | 3 | 0 |
| architecture | 2 | 2 | 0 |
| inference | 1 | 1 | 0 |
| optimisation | 1 | 1 | 0 |
| sampling | 1 | 1 | 0 |
| loss | 1 | 1 | 0 |

## Final evaluation (hidden test set, read only here)

| version | seeds | val mAP@0.5 | test mAP@0.5 | test gain vs start |
|---|---|---|---|---|
| start file | 8 | 0.6288 ± 0.0020 | 0.6868 ± 0.0016 | - |
| final best (e004) | 8 | 0.6325 ± 0.0019 | 0.6880 ± 0.0023 | +0.0012 ± 0.0028 (+0.4 SE) |

Values are mean ± standard error over seeds. A gain above about 2 SE is unlikely to be seed noise.

Per-class val score, start file: pore 0.502, deposit 0.628, discontinuity 0.781, stain 0.610
Per-class val score, final best e004: pore 0.522, deposit 0.624, discontinuity 0.780, stain 0.611

Chain of accepted changes: e000 -> e004

## What changed from the start file to the final best

```diff
--- start/train.py
+++ e004/train.py
@@ -125,7 +125,34 @@
     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
-    crops = [r for r in read_csv(os.path.join(args.data, "crops.csv")) if r["res"] in TRAIN_ON]
+    crops_all = read_csv(os.path.join(args.data, "crops.csv"))
+    crops = [r for r in crops_all if r["res"] in TRAIN_ON]
     dets = read_csv(os.path.join(args.data, "dets.csv"))
     ev = [d for d in dets if d["split"] == "eval"]
+    # Pseudo-label train-split detector boxes that overlap a true box (IoU > 0.5) with that box's
+    # class and train on them too: inference sees detector boxes, so this matches the training
+    # distribution to it (and adds labelled data).
+    by_img = {}
+    for r in crops_all:
+        by_img.setdefault(r["image"], []).append(r)
+    pseudo = []
+    for d in dets:
+        if d["split"] != "train":
+            continue
+        cs = by_img.get(d["image"], [])
+        if not cs:
+            continue
+        a = np.array([[c["x1"], c["y1"], c["x2"], c["y2"]] for c in cs])
+        ix1 = np.maximum(a[:, 0], d["x1"]); iy1 = np.maximum(a[:, 1], d["y1"])
+        ix2 = np.minimum(a[:, 2], d["x2"]); iy2 = np.minimum(a[:, 3], d["y2"])
+        inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
+        ad = (d["x2"] - d["x1"]) * (d["y2"] - d["y1"])
+        ac = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
+        iou = inter / np.clip(ad + ac - inter, 1e-9, None)
+        i = int(np.argmax(iou))
+        if iou[i] > 0.5:
+            d2 = dict(d)
+            d2["cls"] = cs[i]["cls"]
+            pseudo.append(d2)
+    crops = crops + pseudo
     needed = {r["image"] for r in crops} | {d["image"] for d in ev}
     images = {p: np.asarray(Image.open(os.path.join(args.data, "images", p)).convert("RGB")) for p in needed}
```

## All experiments

| id | base | category | hypothesis | seed-0 | 5-seed mean | outcome |
|---|---|---|---|---|---|---|
| e000 | - | - | starting file | 0.6231 | 0.6302 | baseline |
| e001 | e000 | scoring | Currently every eval box emits up to four rows (one per class, score=conf×p(c)), so each true detection also appears as a false positive in the other three classes, polluting every class's precision-r | 0.5749 | - | rejected |
| e002 | e000 | data | Currently only high-res crops are used for training (TRAIN_ON=("high",)), but the eval images may include low-res ones (the one-step baseline 0.638 is below the paper's 0.69 high-res figure, hinting t | 0.6383 | 0.6293 | not confirmed |
| e003 | e000 | architecture | The gap from 0.630 to the 0.758 perfect-class ceiling is dominated by the crop classifier mis-assigning types on boxes that do overlap real defects (pore 0.502 is weakest; stain/deposit confuse), so I | 0.6389 | 0.6313 | not confirmed |
| e004 | e000 | data | The classifier is trained on ground-truth boxes but at inference it sees YOLO detector boxes (slightly misplaced, different size distribution), and the 0.630 vs 0.755 class-agnostic gap shows the loss | 0.6326 | 0.6342 | new best |
| e005 | e004 | scoring | The current score = conf × p(c) lets high-conf detector false alarms (conf≈0.9) outrank lower-conf true positives when the classifier's p(c) is only moderately high, because the multiplicative couplin | 0.1554 | - | suspicious (likely bug) |
| e006 | e004 | data | e004 showed that training on detector boxes matched to true boxes (IoU>0.5) helps, and the 0.634 vs 0.758 ceiling gap is dominated by the classifier failing on exactly the boxes the detector emits (mi | 0.6323 | - | rejected |
| e007 | e004 | inference | The 0.634 vs 0.758-perfect-class ceiling gap is almost entirely class mis-assignment (pore 0.522, stain 0.611 weakest), and the classifier's probabilities are computed from a single forward pass per v | 0.6277 | - | rejected |
| e008 | e004 | data | Pore (0.522) is the weakest class, and pores are the smallest boxes: with MIN_CROP=16 a 10px pore box becomes a ~15px crop that is upscaled ~10x to 160px, so both views are dominated by interpolation  | 0.6387 | 0.6338 | not confirmed |
| e009 | e004 | optimisation | The classifier is a pretrained EfficientNet-B0 fine-tuned for only ~2 epochs (loss still falling 0.44→0.24 at the cut), so it is under-trained; the 0.634→0.758 gap is class mis-assignment (pore 0.522, | 0.6330 | - | rejected |
| e010 | e004 | data | Pore (0.522) and stain (0.611) are the weak classes, and both are small, low-contrast defects whose 160px crops are dominated by interpolation blur; e008 only capped the crop side (MIN_CROP) and was f | - | - | crashed |
| e011 | e004 | data | The 0.634 vs 0.758-perfect-class gap is class mis-assignment, and the per-class curve (pore 0.522, stain 0.611, deposit 0.624) shows the classifier is weakest on small, low-contrast defects whose crop | 0.6357 | 0.6328 | not confirmed |
| e012 | e004 | scoring | The 0.634→0.758-perfect-class gap is cross-class pollution: each true box also emits rows in the other 3 classes with score conf×p(c), so a true pore with p_deposit=0.2 ranks in the deposit list and p | 0.6159 | - | rejected |
| e013 | e004 | data | The crop pixels carry weak class signal for small pores (crop is mostly background) and elongated discontinuities (shape is the key cue), yet the model never sees the box's own size/aspect ratio. I wi | 0.6287 | - | rejected |
| e014 | e004 | architecture | The 0.634→0.758 gap is class mis-assignment, and single-model fine-tuning has plateaued (larger model e003, TTA e007, longer training e009 all flat), so I will train a second, architecturally differen | 0.6004 | - | rejected |
| e015 | e004 | sampling | Pore (0.522) and stain (0.611) are the two weakest classes and likely under-represented in the training distribution relative to their equal 1/4 weight in mAP. By oversampling pore and stain 2× via a  | 0.6227 | - | rejected |
| e016 | e004 | data | The model has never seen a non-defect crop: it is trained only on GT boxes and IoU>0.5 detector boxes, all of which contain a real defect, so at inference false-alarm boxes get a full softmax over the | 0.6215 | - | rejected |
| e017 | e004 | loss | The 0.6342→0.758 gap is almost entirely class mis-assignment (pore 0.522, stain 0.611, deposit 0.624; discontinuity 0.780), i.e. the classifier is confidently wrong on borderline boxes that look like  | 0.6098 | - | rejected |
| e018 | e004 | data | The 0.634→0.758 gap is class mis-assignment (pore 0.522 weakest), and the model currently sees only 1.5x and 2.5x crops of each box; I add a third view, a 4.0x wide crop, so stain/deposit (the known c | 0.6334 | - | rejected |

Each experiment's folder (/data/ar2/runs/b3/eNNN/) holds train.py, diff.patch, the prompt, the LLM's reply and thinking, and the run logs.
