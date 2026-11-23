# AutoResearch report: ar2-3

- Task: det (mAP@0.5); LLM: **qwen3.8-27b-D_gpu5_mtp3** (thinking: xhigh), training budget 180.0 s wall time per run, 5 seeds per confirmed result
- Started 2026-11-01 10:49:10, report written 2026-11-01 20:32:33
- Ideas tried: 35 | edits unusable: 1 | crashed/timeout/bad output: 2 | likely bugs (score far below best): 2 | ran but worse: 24 | not confirmed: 4 | **confirmed improvements: 2**
- LLM time: 6.6 h in 51 calls (465 s per call); training time: 3.1 h

| category | ideas | ran | confirmed |
|---|---|---|---|
| data | 8 | 7 | 0 |
| architecture | 6 | 5 | 0 |
| scoring | 5 | 5 | 0 |
| augmentation | 5 | 5 | 0 |
| optimisation | 5 | 5 | 0 |
| loss | 2 | 2 | 0 |
| sampling | 2 | 2 | 1 |
| inference | 1 | 1 | 1 |
| - | 1 | 0 | 0 |

## Final evaluation (hidden test set, read only here)

| version | seeds | val mAP@0.5 | test mAP@0.5 | test gain vs start |
|---|---|---|---|---|
| start file | 8 | 0.6299 ± 0.0010 | 0.6862 ± 0.0017 | - |
| final best (e016) | 8 | 0.6341 ± 0.0013 | 0.6856 ± 0.0014 | -0.0005 ± 0.0022 (-0.2 SE) |

Values are mean ± standard error over seeds. A gain above about 2 SE is unlikely to be seed noise.

Per-class val score, start file: pore 0.506, deposit 0.630, discontinuity 0.780, stain 0.608
Per-class val score, final best e016: pore 0.524, deposit 0.626, discontinuity 0.779, stain 0.606

Chain of accepted changes: e000 -> e013 -> e016

## What changed from the start file to the final best

```diff
--- start/train.py
+++ e016/train.py
@@ -52,5 +52,5 @@
 LR = 1e-3
 WEIGHT_DECAY = 0.05
-RESERVE_S = 55          # seconds kept free for prediction and writing
+RESERVE_S = 65          # seconds kept free for prediction and writing
 MIN_SCORE = 0.001       # rows below this score are not written
 MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
@@ -126,9 +126,10 @@
 
     crops = [r for r in read_csv(os.path.join(args.data, "crops.csv")) if r["res"] in TRAIN_ON]
+    train_rows = list(crops) + [c for c in crops if c["cls"] in (0, 3)]  # 2x oversample weak pore/stain
     dets = read_csv(os.path.join(args.data, "dets.csv"))
     ev = [d for d in dets if d["split"] == "eval"]
     needed = {r["image"] for r in crops} | {d["image"] for d in ev}
     images = {p: np.asarray(Image.open(os.path.join(args.data, "images", p)).convert("RGB")) for p in needed}
-    print(f"train {len(crops)} boxes, eval {len(ev)} detections, {len(images)} images ({time.time() - T0:.0f}s)", flush=True)
+    print(f"train {len(train_rows)} boxes ({len(crops)} GT, weak 2x), eval {len(ev)} detections, {len(images)} images ({time.time() - T0:.0f}s)", flush=True)
 
     workers = min(8, os.cpu_count() or 2)
@@ -139,5 +140,5 @@
 
     g = torch.Generator().manual_seed(args.seed)
-    dl_tr = torch.utils.data.DataLoader(BoxDataset(crops, images, True), batch_size=BATCH_SIZE, shuffle=True,
+    dl_tr = torch.utils.data.DataLoader(BoxDataset(train_rows, images, True), batch_size=BATCH_SIZE, shuffle=True,
                                         num_workers=workers, pin_memory=True, drop_last=True,
                                         persistent_workers=True, generator=g, worker_init_fn=worker_init)
@@ -186,5 +187,6 @@
             xt = xt.to(device).to(memory_format=torch.channels_last)
             xw = xw.to(device).to(memory_format=torch.channels_last)
-            probs.append(torch.softmax(((model(xt) + model(xw)) / 2).float(), 1).cpu())
+            logits = (model(xt) + model(xt.flip(3)) + model(xw) + model(xw.flip(3))) / 4
+            probs.append(torch.softmax(logits.float(), 1).cpu())
     probs = torch.cat(probs).numpy()
     os.makedirs(args.out, exist_ok=True)
```

## All experiments

| id | base | category | hypothesis | seed-0 | 5-seed mean | outcome |
|---|---|---|---|---|---|---|
| e000 | - | - | starting file | 0.6349 | 0.6311 | baseline |
| e001 | e000 | data | Training the classifier on detector boxes (dets.csv train) labelled by IoU>0.5 overlap with GT crops, with an explicit background class for false alarms, will improve mAP by teaching the model to supp | - | - | crashed |
| e002 | e001 | data | The e001 crash was a trivial type bug (`img_w`/`img_h` left as strings by `read_csv`); fixing it lets us test the core idea—training on detector boxes with an explicit background class—whose hypothesi | 0.5172 | - | rejected |
| e003 | e000 | data | Including low-resolution training crops alongside high-resolution ones roughly increases the training set size, giving the classifier more examples of each defect type (especially the weak pore and st | 0.6251 | - | rejected |
| e004 | e000 | scoring | Squaring the class probability in the score (conf × p² instead of conf × p) will improve mAP by more aggressively suppressing predictions where the classifier is uncertain (likely background boxes), t | 0.6161 | - | rejected |
| e005 | e000 | data | Training the classifier on detector boxes (from dets.csv train split) that overlap true crops (IoU>0.5) instead of the true box coordinates will close the train/test distribution gap: the classifier w | 0.6248 | - | rejected |
| e006 | e000 | architecture | Replacing EfficientNet-B0 with ConvNeXt-Nano, a modern backbone using depthwise-separable convolutions and layer normalisation, will generalise better to the weak pore and stain classes because its di | 0.2901 | - | suspicious (likely bug) |
| e007 | e000 | augmentation | Adding MixUp (α=0.2) to the training loop will regularize the classifier and improve generalization, especially for the weak pore and stain classes, by creating virtual training examples that smooth t | 0.6326 | 0.6292 | not confirmed |
| e008 | e000 | scoring | Multiplying the score by the classifier's max probability (conf × p[c] × max(p)) will improve mAP by suppressing background boxes more than true detections, since background crops tend to produce more | 0.6265 | - | rejected |
| e009 | e000 | data | Increasing the wide-crop context from 2.5× to 3.5× will improve the weak stain and pore classes by giving the classifier more surrounding weld-pattern context, helping it distinguish stain from deposi | 0.6342 | 0.6309 | not confirmed |
| e010 | e000 | loss | Label smoothing (0.1) will improve mAP by preventing the classifier from becoming overconfident, which improves the ranking of true detections above false alarms (critical for mAP@0.5) and acts as a r | 0.6328 | 0.6254 | not confirmed |
| e011 | e000 | augmentation | The detector boxes are "slightly misplaced" relative to GT, but the current training jitter (±10% center, 0.85–1.2× size) may be too conservative to cover the actual misplacement distribution. Increas | 0.6356 | 0.6294 | not confirmed |
| e012 | e000 | loss | Focal loss (γ=2) will improve mAP by down-weighting easy-to-classify examples and concentrating gradient updates on hard examples (the weak pore/stain classes and confusing stain-vs-deposit cases), im | 0.6209 | - | rejected |
| e013 | e000 | inference | Test-time augmentation with horizontal flip (averaging logits over original and h-flipped crops) will reduce prediction variance and improve classification accuracy, especially on the weak pore and st | 0.6392 | 0.6314 | new best |
| e014 | e013 | scoring | Outputting only the top-1 class per detector box (instead of all 4 class rows) will improve mAP by eliminating wrong-class false positives: each non-top-1 row is an unconditional FP for its class (a b | 0.5916 | - | rejected |
| e015 | e013 | data | Adding detector boxes (dets.csv train) that overlap GT crops (IoU>0.5) to the GT training set will improve mAP by exposing the classifier to the detector's actual box placement distribution (slightly  | 0.6285 | - | rejected |
| e016 | e013 | sampling | Pore (0.507) and stain (0.609) are the weak classes, and class-weighted loss hurt every time; oversampling these two classes in the training set (2x) instead increases the number of gradient updates o | 0.6333 | 0.6337 | new best |
| e017 | e016 | architecture | EfficientNet-B0 is pretrained on 224×224 images; using 160×160 discards spatial detail that is critical for distinguishing small pore holes and fine stain textures. Matching the pretrained resolution  | - | - | timeout |
| e018 | e016 | sampling | Pore (0.524) and stain (0.606) remain the weakest classes; increasing their oversampling from 2× to 3× will give the classifier more exposure to these hard-to-distinguish defect types per training ste | 0.6200 | - | rejected |
| e019 | e016 | optimisation | The classifier is undertrained (loss still falling 0.62→0.31→0.24 over the 3 fit epochs) and prediction only consumes ~42s of the 65s reserve, so cutting the reserve to 50s buys ~20% more training ste | 0.6291 | - | rejected |
| e020 | e016 | optimisation | AdamW with high weight decay (0.05) and low LR (1e-3) may be suboptimal for fine-tuning a pretrained backbone; switching to SGD with momentum (the standard fine-tuning recipe: LR=0.01, momentum=0.9, W | 0.6242 | - | rejected |
| e021 | e016 | architecture | Replacing EfficientNet-B0 with tf_efficientnetv2_b0, a newer and more accurate variant of the same family, should improve per-crop classification accuracy (especially for the weak pore and stain class | 0.6259 | - | rejected |
| e022 | e016 | scoring | Squaring the detector confidence (conf² × p[c] instead of conf × p[c]) will improve mAP by more aggressively suppressing low-confidence detector boxes (likely background false alarms), since the detec | 0.6113 | - | rejected |
| e023 | e016 | scoring | Using score = conf × p[c]² / max(p) keeps the top-class score identical to the current conf × p[c] (since p[c]²/max(p) = p[c] when p[c]=max(p)) but suppresses all non-top-class scores (p[c]²/max(p) <  | 0.6080 | - | rejected |
| e024 | e016 | data | Adding low-resolution training crops exclusively for the two weak classes (pore, stain) will improve their classification by providing additional appearance diversity (different resolution artifacts,  | 0.6187 | - | rejected |
| e025 | e016 | data | Roughly 85% of stage-1 boxes are false alarms, and the current 4-class classifier has no "not a defect" notion, so it assigns non-trivial defect probabilities to them and pollutes the ranking. Adding  | 0.6233 | - | rejected |
| e026 | e016 | augmentation | Adding random erasing (cutout) to training will improve generalization on the weak pore and stain classes by preventing the model from relying on a single small local feature, forcing it to learn more | 0.6064 | - | rejected |
| e027 | e016 | augmentation | Pore and stain defects have no fixed image orientation, yet training only uses flips/brightness, so the model may overfit to the bead's typical horizontal orientation; adding small random rotation (±1 | 0.6323 | - | rejected |
| e028 | e016 | architecture | ResNet-50 has similar FLOPs to EfficientNet-B0 at 160 px (so similar speed) but a deeper residual architecture with 25 M parameters; its richer pretrained features should improve per-crop classificati | 0.6256 | - | rejected |
| e029 | e016 | optimisation | Reducing the input resolution from 160 to 128 cuts per-step compute by ~36%, allowing roughly 60% more training epochs within the same wall-clock budget; since the loss is still falling after 3 epochs | 0.6177 | - | rejected |
| e030 | e016 | optimisation | The cosine LR schedule decays to ~0 in the last ~20% of training (77 of 387 steps), making those steps near no-ops. Clamping the LR floor to 10% of the initial LR (1e-4 instead of 1e-6) lets the model | 0.6224 | - | rejected |
| e031 | e016 | optimisation | The classifier is clearly undertrained (loss still dropping 0.62→0.31→0.24 over 3 fit epochs), and halving the batch size from 64 to 32 roughly doubles the number of gradient updates per epoch, allowi | 0.6241 | - | rejected |
| e032 | e016 | architecture | Replacing EfficientNet-B0 (4.5M params) with ConvNeXt-Pico (1.5M params) makes each training step ~3× faster, allowing ~8–9 epochs instead of 3 within the same wall-clock budget; since the loss is sti | 0.4910 | - | suspicious (likely bug) |
| e033 | e016 | architecture | The classifier is the bottleneck (4-class 0.634 vs the 0.755 single-class ceiling) and training duration is not the limiting factor, so a larger, more accurate backbone in the same family (EfficientNe | 0.6288 | - | rejected |
| e034 | e016 | augmentation | Adding per-channel colour jitter (independent ±8% brightness on R/G/B) on top of the existing global brightness will make the classifier more robust to small illumination and colour-temperature variat | 0.6203 | - | rejected |
| e035 | e016 | - |  | - | - | edit failed |

Each experiment's folder (/data/ar2/runs/ar2-3/eNNN/) holds train.py, diff.patch, the prompt, the LLM's reply and thinking, and the run logs.
