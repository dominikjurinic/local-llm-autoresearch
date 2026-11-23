# AutoResearch report: b2

- Task: det (mAP@0.5); LLM: **qwen3.8-27b-D_gpu5_mtp3** (thinking: xhigh), training budget 180.0 s wall time per run, 5 seeds per confirmed result
- Started 2026-11-02 15:32:11, report written 2026-11-02 20:23:22
- Harness: brief ar2_program_v21.md; acceptance gain > 1.0 SE; second seed for close calls; lessons table in prompt
- Ideas tried: 19 | edits unusable: 0 | crashed/timeout/bad output: 2 | likely bugs (score far below best): 2 | ran but worse: 12 | not confirmed: 2 | **confirmed improvements: 1**
- LLM time: 2.8 h in 22 calls (450 s per call); training time: 2.1 h

| category | ideas | ran | confirmed |
|---|---|---|---|
| data | 8 | 7 | 1 |
| scoring | 4 | 4 | 0 |
| architecture | 3 | 2 | 0 |
| inference | 1 | 1 | 0 |
| augmentation | 1 | 1 | 0 |
| optimisation | 1 | 1 | 0 |
| sampling | 1 | 1 | 0 |

## Final evaluation (hidden test set, read only here)

| version | seeds | val mAP@0.5 | test mAP@0.5 | test gain vs start |
|---|---|---|---|---|
| start file | 8 | 0.6290 ± 0.0021 | 0.6883 ± 0.0022 | - |
| final best (e019) | 8 | 0.6346 ± 0.0024 | 0.6808 ± 0.0014 | -0.0075 ± 0.0026 (-2.9 SE) |

Values are mean ± standard error over seeds. A gain above about 2 SE is unlikely to be seed noise.

Per-class val score, start file: pore 0.496, deposit 0.629, discontinuity 0.780, stain 0.609
Per-class val score, final best e019: pore 0.539, deposit 0.618, discontinuity 0.779, stain 0.602

Chain of accepted changes: e000 -> e019

## What changed from the start file to the final best

```diff
--- start/train.py
+++ e019/train.py
@@ -47,4 +47,5 @@
 CONTEXT = 2.5           # wide crop: side = CONTEXT * long side of the box
 TIGHT_CONTEXT = 1.5     # tight crop
+WIDE2_CONTEXT = 4.0     # very wide crop: bead-level context
 MIN_CROP = 16
 TRAIN_ON = ("high",)    # resolutions of crops.csv used for training
@@ -111,5 +112,14 @@
             x = (x - MEAN) / STD
             views.append(torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))))
-        return views[0], views[1], r.get("cls", -1)
+        x = make_crop(img, r, WIDE2_CONTEXT, self.train)
+        if fh:
+            x = x[:, ::-1]
+        if fv:
+            x = x[::-1, :]
+        if self.train:
+            x = np.clip(x * bright, 0, 1)
+        x = (x - MEAN) / STD
+        views.append(torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))))
+        return views[0], views[1], views[2], r.get("cls", -1)
 
 
@@ -139,5 +149,5 @@
 
     g = torch.Generator().manual_seed(args.seed)
-    dl_tr = torch.utils.data.DataLoader(BoxDataset(crops, images, True), batch_size=BATCH_SIZE, shuffle=True,
+    dl_tr = torch.utils.data.DataLoader(BoxDataset(crops, images, True), batch_size=48, shuffle=True,
                                         num_workers=workers, pin_memory=True, drop_last=True,
                                         persistent_workers=True, generator=g, worker_init_fn=worker_init)
@@ -159,5 +169,5 @@
     while not done:
         tot, nb = 0.0, 0
-        for xt, xw, y in dl_tr:
+        for xt, xw, xw2, y in dl_tr:
             frac = min(1.0, (time.time() - t_start) / budget)
             f = frac / 0.05 if frac < 0.05 else 0.5 * (1 + math.cos(math.pi * (frac - 0.05) / 0.95))
@@ -166,7 +176,8 @@
             xt = xt.to(device, non_blocking=True).to(memory_format=torch.channels_last)
             xw = xw.to(device, non_blocking=True).to(memory_format=torch.channels_last)
+            xw2 = xw2.to(device, non_blocking=True).to(memory_format=torch.channels_last)
             y = y.to(device, non_blocking=True)
             with torch.autocast(**amp):
-                loss = crit((model(xt) + model(xw)) / 2, y)
+                loss = crit((model(xt) + model(xw) + model(xw2)) / 3, y)
             opt.zero_grad(set_to_none=True)
             loss.backward()
@@ -183,8 +194,9 @@
     probs = []
     with torch.no_grad(), torch.autocast(**amp):
-        for xt, xw, _ in dl_ev:
+        for xt, xw, xw2, _ in dl_ev:
             xt = xt.to(device).to(memory_format=torch.channels_last)
             xw = xw.to(device).to(memory_format=torch.channels_last)
-            probs.append(torch.softmax(((model(xt) + model(xw)) / 2).float(), 1).cpu())
+            xw2 = xw2.to(device).to(memory_format=torch.channels_last)
+            probs.append(torch.softmax(((model(xt) + model(xw) + model(xw2)) / 3).float(), 1).cpu())
     probs = torch.cat(probs).numpy()
     os.makedirs(args.out, exist_ok=True)
```

## All experiments

| id | base | category | hypothesis | seed-0 | 5-seed mean | outcome |
|---|---|---|---|---|---|---|
| e000 | - | - | starting file | 0.6292 | 0.6288 | baseline |
| e001 | e000 | data | Adding low-resolution training boxes roughly doubles the training set (from ~6052 to ~12000), exposing the classifier to more diverse defect examples and reducing overfitting. This should improve gene | 0.6246 | - | rejected |
| e002 | e000 | architecture | The gap to the 0.758 ceiling is class mis-assignment (classifier macro-F1 ~0.87), concentrated in the weak pore (0.496) and stain/deposit-confused classes, so more backbone capacity should improve fin | 0.6234 | - | rejected |
| e003 | e000 | data | Pore (AP 0.496) is the weakest class and is a fine-grained texture, so I suspect the fixed 160px crop resize is destroying the small-hole detail that separates pore from stain/background. Raising IMG_ | 0.6197 | - | rejected |
| e004 | e000 | data | The classifier is trained on ground-truth crop boxes (crops.csv) but at test time it is fed detector boxes (dets.csv), which are slightly misplaced and include false alarms — a train/test input-distri | 0.6277 | - | rejected |
| e005 | e000 | scoring | The score is `conf * p_c`, but the closed-set classifier is confident even on false alarms and on misassigned true boxes, so `p_c` is a noisy ranking signal that can demote true boxes (low p_c) below  | 0.6049 | - | rejected |
| e006 | e000 | scoring | The class-agnostic ceiling (0.755) is nearly equal to the perfect-class ceiling (0.758), so the ~0.13 gap to our 0.6288 is class-confusion damage, not box detection. Every box is emitted in all 4 clas | 0.5994 | - | rejected |
| e007 | e000 | inference | The dominant error is class mis-assignment (stage-1 boxes are 0.755 class-agnostic but we only reach 0.6288, a ~0.126 mAP cost), so I will improve the classifier's probability estimates with test-time | 0.6237 | - | rejected |
| e008 | e000 | augmentation | The ~0.129 mAP gap to the perfect-class ceiling is class mis-assignment (macro-F1 ~0.87, concentrated in the weak pore/deposit/stain classes), so I add mixup (image+label blending with a per-batch Bet | 0.6136 | - | rejected |
| e009 | e000 | data | The ~0.129 mAP gap to the 0.758 perfect-class ceiling is class mis-assignment, and every attempt to improve the classifier on the same inputs (more data, bigger backbone, higher resolution, TTA, mixup | 0.6264 | - | rejected |
| e010 | e000 | optimisation | The training loss is still falling steeply when training stops (0.244 -> 0.220 in the last epoch), so the classifier is undertrained within the fixed budget, and undertraining is a prime suspect for t | 0.4363 | - | suspicious (likely bug) |
| e011 | e000 | scoring | The 0.129 gap to the 0.758 ceiling is class mis-assignment, and the key mechanism is that a misclassified true box (e.g. a deposit the classifier calls pore) enters the wrong class's ranking at score  | 0.6315 | 0.6267 | not confirmed |
| e012 | e000 | architecture | The ~0.13 mAP gap to the 0.758 ceiling is class mis-assignment, and every single-model attempt to push the classifier past ~0.87 macro-F1 (more data, bigger backbone, higher res, TTA, mixup, re-scorin | - | - | timeout |
| e013 | e012 | architecture | e012's 3-model ensemble timed out because resnet50 is too expensive (steady-state step jumped to ~0.40s and prediction ~3x), not because ensembling is wrong. I keep the ensemble idea but swap resnet50 | 0.6146 | - | rejected |
| e014 | e000 | sampling | Pore (AP 0.496) and stain (0.609) are the weakest classes and the classifier is trained on a fixed 6052-box set for only ~5 epochs, so I suspect the rare/weak classes are under-represented in the batc | 0.6134 | - | rejected |
| e015 | e000 | data | The 0.129 gap to the perfect-class ceiling has two parts: mis-classed true boxes AND false alarms that, unlike in the 0.758 ceiling (where FPs are kept only as class 0), pollute all four class ranking | - | - | crashed |
| e016 | e015 | data | e015's idea (a 5th "background" class from detector boxes overlapping no GT, with every class score multiplied by 1-p_bg) was never actually tested because the file crashed: `bg` was referenced in the | 0.0000 | - | suspicious (likely bug) |
| e017 | e000 | data | The 0.126 gap to the 0.755 class-agnostic ceiling comes from false alarms and misassigned boxes polluting all four class rankings, and no previous experiment ever gave the model an explicit "not-a-def | 0.6288 | 0.6288 | not confirmed |
| e018 | e000 | scoring | The ~0.129 mAP gap to the 0.758 perfect-class ceiling is dominated by false alarms and misassigned boxes polluting all four class rankings, and the closed-set classifier is overconfident on them (it m | 0.5546 | - | rejected |
| e019 | e000 | data | The stain/deposit confusion (stain 0.609, deposit 0.629, and the known stain↔deposit confusion) suggests the model needs to see where the defect sits relative to the bead, but the wide 2.5x crop is re | 0.6466 | 0.6342 | new best |

Each experiment's folder (/data/ar2/runs/b2/eNNN/) holds train.py, diff.patch, the prompt, the LLM's reply and thinking, and the run logs.
