# AutoResearch report: ar2-2

- Task: det (mAP@0.5); LLM: **qwen3.8-27b-D_gpu5_mtp3** (thinking: xhigh), training budget 180.0 s wall time per run, 5 seeds per confirmed result
- Started 2026-11-01 00:14:34, report written 2026-11-01 09:30:35
- Ideas tried: 39 | edits unusable: 0 | crashed/timeout/bad output: 1 | likely bugs (score far below best): 1 | ran but worse: 29 | not confirmed: 3 | **confirmed improvements: 5**
- LLM time: 5.5 h in 49 calls (405 s per call); training time: 3.7 h

| category | ideas | ran | confirmed |
|---|---|---|---|
| architecture | 9 | 8 | 2 |
| data | 5 | 5 | 2 |
| scoring | 5 | 5 | 0 |
| optimisation | 5 | 5 | 0 |
| loss | 4 | 4 | 0 |
| augmentation | 4 | 4 | 0 |
| inference | 4 | 4 | 1 |
| sampling | 3 | 3 | 0 |

## Final evaluation (hidden test set, read only here)

| version | seeds | val mAP@0.5 | test mAP@0.5 | test gain vs start |
|---|---|---|---|---|
| start file | 8 | 0.6267 ± 0.0017 | 0.6864 ± 0.0020 | - |
| final best (e016) | 8 | 0.6397 ± 0.0011 | 0.6949 ± 0.0011 | +0.0085 ± 0.0023 (+3.8 SE) |

Values are mean ± standard error over seeds. A gain above about 2 SE is unlikely to be seed noise.

Per-class val score, start file: pore 0.497, deposit 0.628, discontinuity 0.780, stain 0.607
Per-class val score, final best e016: pore 0.537, deposit 0.635, discontinuity 0.780, stain 0.612

Chain of accepted changes: e000 -> e002 -> e003 -> e008 -> e009 -> e016

## What changed from the start file to the final best

```diff
--- start/train.py
+++ e016/train.py
@@ -34,4 +34,5 @@
 import os
 import random
+from collections import defaultdict
 
 import numpy as np
@@ -44,9 +45,9 @@
 MODEL = "efficientnet_b0"
 PRETRAINED = True
-IMG_SIZE = 160
+IMG_SIZE = 224
 CONTEXT = 2.5           # wide crop: side = CONTEXT * long side of the box
 TIGHT_CONTEXT = 1.5     # tight crop
 MIN_CROP = 16
-TRAIN_ON = ("high",)    # resolutions of crops.csv used for training
+TRAIN_ON = ("high", "low")    # resolutions of crops.csv used for training
 BATCH_SIZE = 64
 LR = 1e-3
@@ -98,18 +99,15 @@
         fh = fv = False
         bright = 1.0
-        if self.train:  # same flips and brightness for both views
+        if self.train:
             fh, fv, bright = random.random() < 0.5, random.random() < 0.5, random.uniform(0.8, 1.2)
-        views = []
-        for ctx in (TIGHT_CONTEXT, CONTEXT):
-            x = make_crop(img, r, ctx, self.train)
-            if fh:
-                x = x[:, ::-1]
-            if fv:
-                x = x[::-1, :]
-            if self.train:
-                x = np.clip(x * bright, 0, 1)
-            x = (x - MEAN) / STD
-            views.append(torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))))
-        return views[0], views[1], r.get("cls", -1)
+        x = make_crop(img, r, CONTEXT, self.train)
+        if fh:
+            x = x[:, ::-1]
+        if fv:
+            x = x[::-1, :]
+        if self.train:
+            x = np.clip(x * bright, 0, 1)
+        x = (x - MEAN) / STD
+        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))), r.get("cls", -1)
 
 
@@ -125,10 +123,30 @@
     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
-    crops = [r for r in read_csv(os.path.join(args.data, "crops.csv")) if r["res"] in TRAIN_ON]
+    all_crops = read_csv(os.path.join(args.data, "crops.csv"))
+    crops = [r for r in all_crops if r["res"] in TRAIN_ON]
     dets = read_csv(os.path.join(args.data, "dets.csv"))
+    train_dets = [d for d in dets if d["split"] == "train"]
     ev = [d for d in dets if d["split"] == "eval"]
-    needed = {r["image"] for r in crops} | {d["image"] for d in ev}
+    crops_by_img = defaultdict(list)
+    for r in all_crops:
+        crops_by_img[r["image"]].append(r)
+    matched_dets = []
+    for d in train_dets:
+        best_iou, best_cls = 0.0, -1
+        for c in crops_by_img.get(d["image"], []):
+            x1, y1 = max(d["x1"], c["x1"]), max(d["y1"], c["y1"])
+            x2, y2 = min(d["x2"], c["x2"]), min(d["y2"], c["y2"])
+            inter = max(0, x2 - x1) * max(0, y2 - y1)
+            union = (d["x2"]-d["x1"])*(d["y2"]-d["y1"]) + (c["x2"]-c["x1"])*(c["y2"]-c["y1"]) - inter
+            iou = inter / union if union > 0 else 0.0
+            if iou > best_iou:
+                best_iou, best_cls = iou, c["cls"]
+        if best_iou >= 0.5:
+            d["cls"] = best_cls
+            matched_dets.append(d)
+    train_rows = crops + matched_dets
+    needed = {r["image"] for r in train_rows} | {d["image"] for d in ev}
     images = {p: np.asarray(Image.open(os.path.join(args.data, "images", p)).convert("RGB")) for p in needed}
-    print(f"train {len(crops)} boxes, eval {len(ev)} detections, {len(images)} images ({time.time() - T0:.0f}s)", flush=True)
+    print(f"train {len(train_rows)} boxes ({len(crops)} gt + {len(matched_dets)} det), eval {len(ev)} detections, {len(images)} images ({time.time() - T0:.0f}s)", flush=True)
 
     workers = min(8, os.cpu_count() or 2)
@@ -139,5 +157,5 @@
 
     g = torch.Generator().manual_seed(args.seed)
-    dl_tr = torch.utils.data.DataLoader(BoxDataset(crops, images, True), batch_size=BATCH_SIZE, shuffle=True,
+    dl_tr = torch.utils.data.DataLoader(BoxDataset(train_rows, images, True), batch_size=BATCH_SIZE, shuffle=True,
                                         num_workers=workers, pin_memory=True, drop_last=True,
                                         persistent_workers=True, generator=g, worker_init_fn=worker_init)
@@ -159,14 +177,13 @@
     while not done:
         tot, nb = 0.0, 0
-        for xt, xw, y in dl_tr:
+        for x, y in dl_tr:
             frac = min(1.0, (time.time() - t_start) / budget)
             f = frac / 0.05 if frac < 0.05 else 0.5 * (1 + math.cos(math.pi * (frac - 0.05) / 0.95))
             for pg in opt.param_groups:
                 pg["lr"] = LR * max(f, 1e-3)
-            xt = xt.to(device, non_blocking=True).to(memory_format=torch.channels_last)
-            xw = xw.to(device, non_blocking=True).to(memory_format=torch.channels_last)
+            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
             y = y.to(device, non_blocking=True)
             with torch.autocast(**amp):
-                loss = crit((model(xt) + model(xw)) / 2, y)
+                loss = crit(model(x), y)
             opt.zero_grad(set_to_none=True)
             loss.backward()
@@ -180,11 +197,13 @@
 
     # Classify every eval detection; one prediction per class with score = detector conf x class probability.
+    # TTA: average predictions from original and horizontal-flip views.
     model.eval()
     probs = []
     with torch.no_grad(), torch.autocast(**amp):
-        for xt, xw, _ in dl_ev:
-            xt = xt.to(device).to(memory_format=torch.channels_last)
-            xw = xw.to(device).to(memory_format=torch.channels_last)
-            probs.append(torch.softmax(((model(xt) + model(xw)) / 2).float(), 1).cpu())
+        for x, _ in dl_ev:
+            x = x.to(device).to(memory_format=torch.channels_last)
+            p1 = torch.softmax(model(x).float(), 1)
+            p2 = torch.softmax(model(x.flip(-1)).float(), 1)
+            probs.append(((p1 + p2) / 2).cpu())
     probs = torch.cat(probs).numpy()
     os.makedirs(args.out, exist_ok=True)
```

## All experiments

| id | base | category | hypothesis | seed-0 | 5-seed mean | outcome |
|---|---|---|---|---|---|---|
| e000 | - | - | starting file | 0.6280 | 0.6280 | baseline |
| e001 | e000 | data | Training on detector boxes (labelled by IoU≥0.5 overlap with ground truth, plus a 5th "background" class for false alarms) instead of perfect crops will improve mAP because the classifier becomes robu | 0.5691 | - | rejected |
| e002 | e000 | data | Adding detector boxes that match ground truth (IoU ≥ 0.5) to the training set alongside perfect crops will improve mAP because the classifier becomes robust to the detector's box misplacement while re | 0.6368 | 0.6293 | new best |
| e003 | e002 | data | Adding low-resolution ground-truth crops to training will improve mAP because the classifier sees more diverse examples at the resolution the detector actually operates on, improving robustness for th | 0.6296 | 0.6317 | new best |
| e004 | e003 | scoring | Squaring the class probability in the score (conf × prob²) will improve mAP by better separating true detections (high prob for correct class) from false alarms and cross-class rows (lower prob), impr | 0.6211 | - | rejected |
| e005 | e003 | data | Adding a 5th "background" class (from unmatched detector boxes, capped to 1.5× matched count) while keeping GT crops will let the classifier learn "not a defect"; multiplying the score by (1 − prob_bg | 0.6039 | - | rejected |
| e006 | e003 | loss | Focal loss (γ=2, no class weights) will improve mAP by upweighting hard-to-classify examples—especially pore and stain, the two weakest classes—during training, without the negative effects seen when  | 0.6123 | - | rejected |
| e007 | e003 | augmentation | Mixup augmentation (blending two training samples with a Beta-distributed weight and soft-label cross-entropy) will improve mAP by creating virtual examples that smooth the decision boundary between t | 0.6233 | - | rejected |
| e008 | e003 | architecture | Switching from two-crop (tight 1.5× + wide 2.5×) to single wide-crop (2.5×) halves the forward-pass cost per sample, allowing ~2.3× more training epochs (from ~1 to ~2.3) within the same time budget.  | 0.6409 | 0.6386 | new best |
| e009 | e008 | inference | Test-time augmentation with horizontal flip at inference (averaging predictions from original and H-flipped crops) will improve classification accuracy—especially for pore and stain, the weakest class | 0.6396 | 0.6387 | new best |
| e010 | e009 | optimisation | Reducing batch size from 64 to 32 (with proportionally scaled LR) doubles the number of optimizer updates within the same wall-clock budget, improving convergence and classification accuracy—especiall | 0.6229 | - | rejected |
| e011 | e009 | sampling | Oversampling pore (AP 0.528) and stain (AP 0.611) by 1.5× via WeightedRandomSampler will improve their classification accuracy by giving the model more exposure to these weak classes during training,  | 0.6305 | - | rejected |
| e012 | e009 | inference | Adding vertical flip (and 180° rotation) to the existing horizontal-flip TTA will improve classification accuracy by giving the model four symmetric views of each crop, helping disambiguate pore from  | 0.6370 | - | rejected |
| e013 | e009 | architecture | EfficientNet-B2 (6.6M params vs B0's 4M) has greater capacity to learn the fine-grained visual distinctions between pore, stain, and deposit, and even with ~1.3 epochs instead of 2 (due to ~1.5× per-s | 0.6340 | - | rejected |
| e014 | e009 | scoring | Multiplying the per-class score by the classifier's max probability (max over 4 classes) will improve mAP by boosting confident true detections and suppressing uncertain false alarms, improving the ra | 0.6320 | - | rejected |
| e015 | e009 | architecture | ConvNeXt-Nano (5M params, sequential depthwise conv + LayerNorm blocks) captures fine-grained texture differences between pore/stain/deposit better than EfficientNet-B0's compound-scaling architecture | 0.5833 | - | rejected |
| e016 | e009 | architecture | Increasing IMG_SIZE from 160 to 224 aligns the input resolution with the pretrained EfficientNet-B0 weights (trained at 224×224) and gives ~40% more pixel detail for small defects like pores (the weak | 0.6423 | 0.6411 | new best |
| e017 | e016 | optimisation | With only ~660 training steps (≈2.5 epochs), the model is undertrained and the cosine schedule decays the LR to near-zero before meaningful convergence on weak classes (pore 0.537, stain 0.612). Doubl | 0.6364 | - | rejected |
| e018 | e016 | architecture | Reducing IMG_SIZE from 224 to 192 cuts per-step cost by ~26% (fewer pixels in both data pipeline and model), yielding ~35% more training steps (≈880 vs 660). Since the model is clearly undertrained (l | 0.6387 | - | rejected |
| e019 | e016 | scoring | Replacing the score `conf × prob[c]` with `conf × (prob[c] − max_other_prob)` (the class margin) will improve mAP because it directly measures how much the model prefers the predicted class over alter | 0.5805 | - | rejected |
| e020 | e016 | optimisation | The cosine LR schedule decays to near-zero in the last ~20% of training (~125 of 660 steps), wasting that time at negligible LR while the loss is still dropping fast (0.52→0.29). Raising the minimum L | 0.6363 | - | rejected |
| e021 | e016 | architecture | tf_efficientnetv2_b0 has a more modern architecture (squeeze-and-excitation blocks, improved scaling strategy) than efficientnet_b0, which should produce better discriminative features for the fine-gr | 0.6447 | 0.6352 | not confirmed |
| e022 | e016 | architecture | MobileNetV3-Large has ~44% fewer FLOPs than EfficientNet-B0 at 224×224, enabling ~45% more training steps (≈950 vs 660) within the same wall-clock budget. The extra training epochs (≈3.6 vs 2.5) shoul | 0.6380 | - | rejected |
| e023 | e016 | loss | Label smoothing (0.1) will improve mAP by preventing the model from becoming overconfident on training examples, yielding better-calibrated class probabilities that improve the ranking component of mA | 0.6340 | - | rejected |
| e024 | e016 | sampling | The classifier only ever sees detector boxes at inference, so upweighting the matched detector boxes (IoU≥0.5) 2× relative to GT crops during training will reduce the train/inference distribution gap  | 0.6397 | - | rejected |
| e025 | e016 | optimisation | Replacing AdamW (LR=1e-3, wd=0.05) with SGD+momentum (LR=0.1, momentum=0.9, wd=1e-4) will improve mAP because SGD with momentum is the standard and often superior optimizer for fine-tuning pretrained  | 0.4765 | - | suspicious (likely bug) |
| e026 | e016 | scoring | The raw YOLOv7-tiny confidence is likely miscalibrated (overconfident on false alarms), so replacing it with a histogram-calibrated "P(defect / conf)" estimated from the ~12k training detector boxes s | 0.6278 | - | rejected |
| e027 | e016 | inference | The weak classes (pore 0.537, stain 0.612) are small/textural defects, so input detail is likely the bottleneck: the 160→224 jump (e016) was the biggest recent gain. Classifying eval crops at 256×256  | 0.6326 | - | rejected |
| e028 | e016 | augmentation | The detector's boxes are misaligned with ground truth by up to ~20-30% of the box size (IoU 0.5-0.7 range), but the current training jitter (±10% center, 0.85-1.2× side) underestimates this misplaceme | 0.6369 | - | rejected |
| e029 | e016 | scoring | The undertrained model (2 epochs, loss still dropping 0.52→0.29) produces overly uniform softmax probabilities, compressing the score gap between true and false detections. Dividing logits by T=0.7 be | 0.6277 | - | rejected |
| e030 | e016 | inference | Removing the H-flip TTA (which gave no measurable gain, e009: 0.6386→0.6387) and reducing the prediction reserve from 55s to 40s frees ~15s for additional training steps (~14% more, 660→~750), improvi | 0.6374 | - | rejected |
| e031 | e016 | sampling | Curriculum training—first learning on clean GT crops (60% of budget) to establish class boundaries, then fine-tuning on detector boxes (40%) to adapt to the inference distribution—will improve mAP bec | 0.6309 | - | rejected |
| e032 | e016 | architecture | RegNetY-016 (0.9M params) is ~3x faster per forward/backward step than EfficientNet-B0 (4M params), enabling ~2000 training steps (vs 660) within the same wall-clock budget. The 3x more optimizer upda | - | - | timeout |
| e033 | e016 | architecture | Training at 160×160 (≈2× fewer pixels per step) while keeping inference at 224×224 will roughly double training steps (≈1300 vs 660), improving convergence for the weak pore/stain classes, while the h | 0.6052 | - | rejected |
| e034 | e016 | loss | mAP@0.5 decomposes into independent per-class rankings, but the softmax couples all four class scores (they must sum to 1), distorting each class's ranking. Replacing CE with 4 independent BCE sigmoid | 0.6447 | 0.6401 | not confirmed |
| e035 | e016 | optimisation | The cosine schedule decays LR to near-zero over ~630 of 660 steps, but the model is clearly undertrained (loss still dropping 0.52→0.29). Replacing cosine decay with constant LR after the 5% warmup ke | 0.6256 | - | rejected |
| e036 | e016 | augmentation | Random erasing (Cutout-style) during training will improve mAP by forcing the model to rely on multiple spatial regions rather than a single texture patch, improving robustness to the detector's box m | 0.6426 | 0.6400 | not confirmed |
| e037 | e016 | augmentation | Reducing the crop context from 2.5× to 2.0× gives the model a tighter view of the defect, reducing the diluting effect of surrounding weld bead and improving discrimination between pore (small circula | 0.6350 | - | rejected |
| e038 | e016 | data | The weld images almost certainly have a very different intensity distribution from ImageNet (likely narrow dynamic range, near-grayscale), so the hardcoded ImageNet mean/std suboptimally scales inputs | 0.6360 | - | rejected |
| e039 | e016 | loss | Adding an auxiliary 2-group CE loss ({pore, stain} vs {deposit, discontinuity}) provides an additional learning signal that separates "small/dark" defects from "structural" defects, helping the weak p | 0.6288 | - | rejected |

Each experiment's folder (/data/ar2/runs/ar2-2/eNNN/) holds train.py, diff.patch, the prompt, the LLM's reply and thinking, and the run logs.
