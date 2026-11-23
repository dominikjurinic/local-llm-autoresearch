# AutoResearch report: b5

- Task: det (mAP@0.5); LLM: **qwen3.8-27b-D_gpu5_mtp3** (thinking: xhigh), training budget 180.0 s wall time per run, 5 seeds per confirmed result
- Started 2026-11-03 05:49:57, report written 2026-11-03 10:45:07
- Harness: brief ar2_program_v21.md; acceptance gain > 1.0 SE; second seed for close calls; lessons table in prompt
- Ideas tried: 17 | edits unusable: 0 | crashed/timeout/bad output: 1 | likely bugs (score far below best): 1 | ran but worse: 8 | not confirmed: 6 | **confirmed improvements: 1**
- LLM time: 2.2 h in 17 calls (456 s per call); training time: 2.8 h

| category | ideas | ran | confirmed |
|---|---|---|---|
| architecture | 4 | 4 | 1 |
| data | 3 | 3 | 0 |
| inference | 3 | 2 | 0 |
| scoring | 2 | 2 | 0 |
| optimisation | 2 | 2 | 0 |
| augmentation | 1 | 1 | 0 |
| sampling | 1 | 1 | 0 |
| loss | 1 | 1 | 0 |

## Final evaluation (hidden test set, read only here)

| version | seeds | val mAP@0.5 | test mAP@0.5 | test gain vs start |
|---|---|---|---|---|
| start file | 8 | 0.6274 ± 0.0018 | 0.6869 ± 0.0015 | - |
| final best (e012) | 8 | 0.6329 ± 0.0018 | 0.6841 ± 0.0009 | -0.0028 ± 0.0018 (-1.6 SE) |

Values are mean ± standard error over seeds. A gain above about 2 SE is unlikely to be seed noise.

Per-class val score, start file: pore 0.502, deposit 0.624, discontinuity 0.780, stain 0.609
Per-class val score, final best e012: pore 0.536, deposit 0.616, discontinuity 0.778, stain 0.606

Chain of accepted changes: e000 -> e012

## What changed from the start file to the final best

```diff
--- start/train.py
+++ e012/train.py
@@ -47,4 +47,5 @@
 CONTEXT = 2.5           # wide crop: side = CONTEXT * long side of the box
 TIGHT_CONTEXT = 1.5     # tight crop
+WIDE2_CONTEXT = 4.0     # very wide crop: whole-bead context to tell stain from deposit
 MIN_CROP = 16
 TRAIN_ON = ("high",)    # resolutions of crops.csv used for training
@@ -101,5 +102,5 @@
             fh, fv, bright = random.random() < 0.5, random.random() < 0.5, random.uniform(0.8, 1.2)
         views = []
-        for ctx in (TIGHT_CONTEXT, CONTEXT):
+        for ctx in (TIGHT_CONTEXT, CONTEXT, WIDE2_CONTEXT):
             x = make_crop(img, r, ctx, self.train)
             if fh:
@@ -111,5 +112,5 @@
             x = (x - MEAN) / STD
             views.append(torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))))
-        return views[0], views[1], r.get("cls", -1)
+        return views[0], views[1], views[2], r.get("cls", -1)
 
 
@@ -159,5 +160,5 @@
     while not done:
         tot, nb = 0.0, 0
-        for xt, xw, y in dl_tr:
+        for xt, xw, x2, y in dl_tr:
             frac = min(1.0, (time.time() - t_start) / budget)
             f = frac / 0.05 if frac < 0.05 else 0.5 * (1 + math.cos(math.pi * (frac - 0.05) / 0.95))
@@ -166,7 +167,8 @@
             xt = xt.to(device, non_blocking=True).to(memory_format=torch.channels_last)
             xw = xw.to(device, non_blocking=True).to(memory_format=torch.channels_last)
+            x2 = x2.to(device, non_blocking=True).to(memory_format=torch.channels_last)
             y = y.to(device, non_blocking=True)
             with torch.autocast(**amp):
-                loss = crit((model(xt) + model(xw)) / 2, y)
+                loss = crit((model(xt) + model(xw) + model(x2)) / 3, y)
             opt.zero_grad(set_to_none=True)
             loss.backward()
@@ -183,8 +185,9 @@
     probs = []
     with torch.no_grad(), torch.autocast(**amp):
-        for xt, xw, _ in dl_ev:
+        for xt, xw, x2, _ in dl_ev:
             xt = xt.to(device).to(memory_format=torch.channels_last)
             xw = xw.to(device).to(memory_format=torch.channels_last)
-            probs.append(torch.softmax(((model(xt) + model(xw)) / 2).float(), 1).cpu())
+            x2 = x2.to(device).to(memory_format=torch.channels_last)
+            probs.append(torch.softmax(((model(xt) + model(xw) + model(x2)) / 3).float(), 1).cpu())
     probs = torch.cat(probs).numpy()
     os.makedirs(args.out, exist_ok=True)
```

## All experiments

| id | base | category | hypothesis | seed-0 | 5-seed mean | outcome |
|---|---|---|---|---|---|---|
| e000 | - | - | starting file | 0.6314 | 0.6288 | baseline |
| e001 | e000 | data | Train the crop classifier on stage-1 detector boxes from the train split (labelled by IoU>0.5 overlap with crops.csv) instead of the ground-truth boxes: at test time we classify exactly these detector | 0.6264 | - | rejected |
| e002 | e000 | data | Include the "low" resolution crops in training (TRAIN_ON = ("high","low")) instead of high-res only. The benchmark is LoHi-WELD, so the eval images are almost certainly a mix of low- and high-res, and | 0.6296 | 0.6261 | not confirmed |
| e003 | e000 | data | Add a 5th "background" class to the classifier, built from stage-1 detector boxes on the train split that do NOT overlap any true box (IoU<=0.5) — i.e. genuine detector false alarms — subsampled to ~1 | 0.5935 | - | rejected |
| e004 | e000 | scoring | Divide the averaged logits by a temperature TAU=0.5 before the softmax at inference, making the class-probability distribution sharper. This boosts the score of the top-predicted class for each box an | 0.6119 | - | rejected |
| e005 | e000 | architecture | Raise the crop input resolution from 160 to 224 (keeping EfficientNet-B0 and the two-crop recipe). Pore (0.502), stain (0.609) and deposit (0.624) AP are texture-driven confusions, and at 160px the fi | 0.6303 | 0.6251 | not confirmed |
| e006 | e000 | augmentation | Add random rotation (any angle, shared by both crop views, gray fill matching the padding) to the training-time augmentation, which currently only has crop jitter, flips and brightness. Weld beads and | 0.6246 | - | rejected |
| e007 | e000 | architecture | The dominant error is class assignment — the gap to the perfect-class ceiling is 0.758 − 0.6288 ≈ 0.13, and the e004 temperature result (sharpening the softmax *hurt* by 0.017) shows the model's argma | 0.6231 | - | rejected |
| e008 | e000 | inference | Add test-time augmentation at inference: for each eval box run the two existing views (tight 1.5x, wide 2.5x) plus their horizontal flips, and average the four softmax outputs before scoring. The 0.13 | 0.6367 | 0.6306 | not confirmed |
| e009 | e000 | architecture | Swap the backbone from efficientnet_b0 to efficientnet_b2 (same family, ~2x capacity) while keeping the two-crop recipe, resolution, and schedule. The gap to the perfect-class ceiling (0.758−0.6288≈0. | 0.6344 | 0.6231 | not confirmed |
| e010 | e000 | sampling | The 0.129 gap to the perfect-class ceiling is class-assignment error concentrated in the weak classes (pore 0.502, stain 0.609, deposit 0.624), which are the least frequent defect types and therefore  | 0.6288 | 0.6276 | not confirmed |
| e011 | e000 | inference | Add vertical-flip TTA to the existing horizontal-flip TTA (e008's recipe: 4 views = tight/wide x H-flip, now 8 views), averaging all 8 softmax outputs before scoring. e008 (flip TTA) was the only expe | - | - | timeout |
| e012 | e000 | architecture | Add a third view to the classifier: a very wide crop (4x the box's long side) alongside the existing tight (1.5x) and wide (2.5x) crops, with the three logits averaged at both train and inference. Sta | 0.6321 | 0.6342 | new best |
| e013 | e012 | inference | Replace the third inference view (the 4x context crop) with horizontal-flip TTA: at inference we run tight, wide, tight-flipped, wide-flipped (4 views, averaged) while training stays exactly as in e01 | 0.6273 | - | rejected |
| e014 | e012 | scoring | Output only the argmax class per eval box (one row, score = conf × p_argmax) instead of all four class rows. Currently every box emits up to 4 rows, and the ~3 non-argmax rows are almost always false  | 0.5786 | - | rejected |
| e015 | e012 | loss | Replace cross-entropy with focal loss (gamma=2.0) so that well-classified examples are down-weighted and training focuses on the hard, visually-similar confusion cases (pore↔stain, stain↔deposit) that | 0.6193 | - | rejected |
| e016 | e012 | optimisation | The run currently stops training at a fixed 55s reserve, but prediction only takes ~58s of the (larger) budget, so the model is cut off after ~3 epochs while the loss is still falling fast (0.275->0.2 | 0.4045 | - | suspicious (likely bug) |
| e017 | e016 | optimisation | The e016 idea (train until the *measured* prediction cost is reserved instead of a fixed 55 s) was sound, but its probe measured 10 eval batches at the very start of the run, when DataLoader workers a | 0.6325 | 0.6355 | not confirmed |

Each experiment's folder (/data/ar2/runs/b5/eNNN/) holds train.py, diff.patch, the prompt, the LLM's reply and thinking, and the run logs.
