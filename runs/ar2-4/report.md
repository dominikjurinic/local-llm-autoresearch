# AutoResearch report: ar2-4

- Task: det (mAP@0.5); LLM: **qwen3.8-27b-D_gpu5_mtp3** (thinking: xhigh), training budget 180.0 s wall time per run, 5 seeds per confirmed result
- Started 2026-11-01 20:58:01, report written 2026-11-02 06:23:34
- Ideas tried: 23 | edits unusable: 1 | crashed/timeout/bad output: 3 | likely bugs (score far below best): 0 | ran but worse: 15 | not confirmed: 3 | **confirmed improvements: 1**
- LLM time: 7.3 h in 44 calls (594 s per call); training time: 2.2 h

| category | ideas | ran | confirmed |
|---|---|---|---|
| architecture | 8 | 6 | 0 |
| data | 6 | 5 | 1 |
| scoring | 2 | 2 | 0 |
| augmentation | 2 | 2 | 0 |
| sampling | 1 | 1 | 0 |
| - | 1 | 0 | 0 |
| loss | 1 | 1 | 0 |
| optimisation | 1 | 1 | 0 |
| inference | 1 | 1 | 0 |

## Final evaluation (hidden test set, read only here)

| version | seeds | val mAP@0.5 | test mAP@0.5 | test gain vs start |
|---|---|---|---|---|
| start file | 8 | 0.6255 ± 0.0012 | 0.6854 ± 0.0012 | - |
| final best (e001) | 8 | 0.6305 ± 0.0022 | 0.6800 ± 0.0016 | -0.0054 ± 0.0020 (-2.7 SE) |

Values are mean ± standard error over seeds. A gain above about 2 SE is unlikely to be seed noise.

Per-class val score, start file: pore 0.494, deposit 0.627, discontinuity 0.780, stain 0.607
Per-class val score, final best e001: pore 0.517, deposit 0.622, discontinuity 0.776, stain 0.605

Chain of accepted changes: e000 -> e001

## What changed from the start file to the final best

```diff
--- start/train.py
+++ e001/train.py
@@ -48,5 +48,5 @@
 TIGHT_CONTEXT = 1.5     # tight crop
 MIN_CROP = 16
-TRAIN_ON = ("high",)    # resolutions of crops.csv used for training
+TRAIN_ON = ("high", "low")    # resolutions of crops.csv used for training
 BATCH_SIZE = 64
 LR = 1e-3
```

## All experiments

| id | base | category | hypothesis | seed-0 | 5-seed mean | outcome |
|---|---|---|---|---|---|---|
| e000 | - | - | starting file | 0.6263 | 0.6270 | baseline |
| e001 | e000 | data | Currently only high-res crops are used for training (`TRAIN_ON = ("high",)`), discarding all low-res labelled boxes; including them roughly doubles the training set and matches the mixed-resolution di | 0.6354 | 0.6302 | new best |
| e002 | e001 | architecture | Train on a single tight crop (1.5x) instead of averaging two views (tight 1.5x + wide 2.5x), which halves the per-step cost and should roughly double optimizer steps in the same budget (~430 -> ~800-1 | - | - | crashed |
| e003 | e002 | architecture | Fix the crash in e002 by updating the training and inference loops to handle the single-crop output (2-tuple instead of 3-tuple). The original idea—training on one tight crop instead of averaging two  | 0.6247 | - | rejected |
| e004 | e001 | architecture | The training loss is still falling fast at the end of the budget (0.31 after only ~2 epochs / 428 steps), so the classifier is undertrained rather than feature-limited, and the main error is class mis | 0.6296 | - | rejected |
| e005 | e001 | architecture | The model is clearly undertrained—loss is still 0.31 after only 428 steps (~2 epochs) and falling fast. Reducing IMG_SIZE from 160 to 128 cuts per-step compute by ~36% ((128/160)²), yielding ~56% more | 0.6217 | - | rejected |
| e006 | e001 | data | The classifier is trained on ground-truth box coordinates but tested on detector boxes, which are slightly offset and differently sized. By replacing each GT box with the best-overlapping detector box | 0.6338 | 0.6278 | not confirmed |
| e007 | e001 | sampling | Pore (0.517) and stain (0.605) are the two weakest classes, and mAP is macro-averaged, so improving them moves the mean most. I will duplicate pore and stain training boxes 2x so batches contain more  | 0.6280 | - | rejected |
| e008 | e001 | data | The classifier is only ever trained on crops of true defects, so detector false alarms get arbitrary class probabilities and pollute every class's score ranking (the 0.758 ceiling stays high largely b | 0.5635 | - | rejected |
| e009 | e001 | scoring | The current code writes one row per class per box (score = conf × p_c), so every detector box injects ~3 spurious false-positive rows into the other classes' lists, deflating per-class precision — con | 0.5710 | - | rejected |
| e010 | e001 | augmentation | Adding random rotation (±15°) and contrast jitter (0.8–1.2) to training augmentation should improve the model's invariance to defect orientation and lighting variation. This targets the weak pore (0.5 | 0.6331 | 0.6255 | not confirmed |
| e011 | e001 | architecture | Replacing EfficientNet-B0 with tf_efficientnetv2_b0 should improve feature extraction for local weld-defect textures (especially pore and stain, the two weakest classes) because V2's FusedMBConv block | 0.6301 | - | rejected |
| e012 | e001 | augmentation | Add mixup to the training loop: each batch interpolates both crop views and the one-hot labels with λ~Beta(0.2,0.2), and the loss becomes the interpolated softmax cross-entropy. The 0.128 gap to the 0 | 0.6141 | - | rejected |
| e013 | e001 | architecture | Replace the single shared classifier head with two separate linear heads—one for the tight (1.5×) crop and one for the wide (2.5×) crop—while keeping the backbone shared. The tight and wide views oper | - | - | crashed |
| e014 | e001 | data | Replace fixed ImageNet normalization with per-crop standardization (each crop zero-mean/unit-variance, clipped to [-3,3]). The gap to the 0.758 ceiling is class misassignment, not false alarms (class- | 0.6212 | - | rejected |
| e015 | e001 | data | Set TIGHT_CONTEXT from 1.5 to 1.0 so the tight view is the box itself instead of a 1.5x enlargement. The weak classes (pore 0.517, stain 0.605) are decided by fine texture, but at 1.5x the defect occu | 0.6221 | - | rejected |
| e016 | e001 | scoring | Fit the exponents of the score function score = conf^a × p_c^b on the train split of detector boxes (true class known by overlap with crops.csv), then apply the best (a,b) to eval. The fixed product a | 0.6241 | - | rejected |
| e017 | e001 | - |  | - | - | edit failed |
| e018 | e001 | architecture | Feed the tight (1.5x) and wide (2.5x) crops as one 6-channel input (channel-wise concatenation) instead of two separate 3-channel forward passes with logits averaged, so the two views can interact fro | 0.6319 | 0.6275 | not confirmed |
| e019 | e001 | loss | Adding label_smoothing=0.1 to CrossEntropyLoss softens the one-hot targets (0.9×one-hot + 0.1/4 per class), acting as a uniform regularizer that prevents overconfidence in an undertrained model. Unlik | 0.6147 | - | rejected |
| e020 | e001 | data | Add a third, extra-wide crop (4x the box side) and average its logits with the tight (1.5x) and wide (2.5x) views, at training and inference. Stain is often confused with deposit, and pore boxes are s | - | - | crashed |
| e021 | e020 | architecture | Fix the crash in e020 by updating the inference loop to handle three views (tight 1.5x, wide 2.5x, extra-wide 4x) and average three logits. The extra-wide crop gives more surface context, which should | 0.6250 | - | rejected |
| e022 | e001 | optimisation | The pretrained backbone is fine-tuned at the same LR (1e-3) as the randomly-initialized head, which likely corrupts the ImageNet features that encode fine-grained texture. Using LR×0.1 for the backbon | 0.6207 | - | rejected |
| e023 | e001 | inference | Average predictions over horizontal-flipped versions of both crops at test time (4 forward passes instead of 2), trading ~15% of training time for test-time augmentation. The remaining gap to the 0.75 | 0.6199 | - | rejected |

Each experiment's folder (/data/ar2/runs/ar2-4/eNNN/) holds train.py, diff.patch, the prompt, the LLM's reply and thinking, and the run logs.
