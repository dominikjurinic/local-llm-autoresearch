# AutoResearch report: ar2-1

- Task: det (mAP@0.5); LLM: **qwen3.8:27b** (thinking: True), training budget 180.0 s wall time per run, 5 seeds per confirmed result
- Started 2026-10-30 21:10:50, report written 2026-10-31 06:41:20
- Ideas tried: 20 | edits unusable: 0 | crashed/timeout/bad output: 3 | likely bugs (score far below best): 1 | ran but worse: 8 | not confirmed: 4 | **confirmed improvements: 3**
- LLM time: 6.8 h in 20 calls (1216 s per call); training time: 2.7 h

| category | ideas | ran | confirmed |
|---|---|---|---|
| architecture | 5 | 3 | 1 |
| augmentation | 4 | 3 | 0 |
| data | 3 | 3 | 0 |
| inference | 2 | 2 | 1 |
| loss | 2 | 2 | 0 |
| optimisation | 2 | 2 | 1 |
| scoring | 2 | 2 | 0 |

## Final evaluation (hidden test set, read only here)

| version | seeds | val mAP@0.5 | test mAP@0.5 | test gain vs start |
|---|---|---|---|---|
| start file | 8 | 0.6275 ± 0.0024 | 0.6870 ± 0.0013 | - |
| final best (e012) | 8 | 0.6371 ± 0.0009 | 0.6924 ± 0.0007 | +0.0054 ± 0.0015 (+3.6 SE) |

Values are mean ± standard error over seeds. A gain above about 2 SE is unlikely to be seed noise.

Per-class val score, start file: pore 0.494, deposit 0.629, discontinuity 0.780, stain 0.610
Per-class val score, final best e012: pore 0.517, deposit 0.636, discontinuity 0.781, stain 0.616

Chain of accepted changes: e000 -> e005 -> e008 -> e012

## What changed from the start file to the final best

```diff
--- start/train.py
+++ e012/train.py
@@ -42,5 +42,5 @@
 
 # ---------------------------------------------------------------- settings
-MODEL = "efficientnet_b0"
+MODEL = "tf_efficientnetv2_b0"
 PRETRAINED = True
 IMG_SIZE = 160
@@ -52,5 +52,5 @@
 LR = 1e-3
 WEIGHT_DECAY = 0.05
-RESERVE_S = 55          # seconds kept free for prediction and writing
+RESERVE_S = 50          # seconds kept free for prediction and writing (extra time for TTA)
 MIN_SCORE = 0.001       # rows below this score are not written
 MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
@@ -186,5 +186,7 @@
             xt = xt.to(device).to(memory_format=torch.channels_last)
             xw = xw.to(device).to(memory_format=torch.channels_last)
-            probs.append(torch.softmax(((model(xt) + model(xw)) / 2).float(), 1).cpu())
+            # TTA: average original and horizontal flip of both views
+            logits = (model(xt) + model(xw) + model(xt.flip(3)) + model(xw.flip(3))) / 4
+            probs.append(torch.softmax(logits.float(), 1).cpu())
     probs = torch.cat(probs).numpy()
     os.makedirs(args.out, exist_ok=True)
```

## All experiments

| id | base | category | hypothesis | seed-0 | 5-seed mean | outcome |
|---|---|---|---|---|---|---|
| e000 | - | - | starting file | 0.6351 | 0.6280 | baseline |
| e001 | e000 | data | Training the classifier on detector boxes (dets.csv train) with a "background" class for false alarms, and using the background probability to suppress scores, will improve ranking by teaching the mod | 0.5512 | - | rejected |
| e002 | e000 | data | Adding low-resolution crops to the training set increases the number of training examples and scale diversity, which should improve generalization for the weak pore and stain classes without introduci | 0.6209 | - | rejected |
| e003 | e000 | architecture | ConvNeXt-Nano's modern inductive biases (depthwise convolutions, LayerNorm, better channel mixing) should improve discrimination of visually similar defect types (pore vs stain vs deposit) compared to | 0.3721 | - | suspicious (likely bug) |
| e004 | e000 | data | Training on detector boxes (dets.csv train) matched to true crops by IoU will improve robustness to the detector's actual box misalignment and size errors, giving the model more examples at the positi | 0.6234 | - | rejected |
| e005 | e000 | inference | Test-time augmentation with horizontal flip at inference will improve classification accuracy (especially for weak pore and stain classes) by averaging out position-dependent model biases, providing a | 0.6372 | 0.6286 | new best |
| e006 | e005 | loss | Replacing cross-entropy with focal loss (gamma=2) will focus training on hard, confused examples (pore vs deposit vs stain) rather than easy separable ones, improving discrimination between visually s | 0.6215 | - | rejected |
| e007 | e005 | loss | Label smoothing (epsilon=0.1) will reduce overconfidence in individual class predictions, improving cross-box ranking (uncertain/false-alarm boxes get lower scores) while preserving within-box class o | 0.6230 | - | rejected |
| e008 | e005 | optimisation | The model is still learning at the end of training (loss 0.283 and decreasing), so reducing the prediction reserve from 90s to 50s gives ~40% more training steps, which should improve classification a | 0.6329 | 0.6314 | new best |
| e009 | e008 | scoring | Multiplying the per-class score by the classifier's max probability (across all 4 classes) will suppress scores for uncertain/background boxes where the classifier is diffuse (~0.25 per class), improv | 0.6355 | 0.6298 | not confirmed |
| e010 | e008 | architecture | Increasing the input resolution from 160 to 224 gives the network ~2x more pixels to resolve small features (especially pores, the weakest class at 0.505 AP), improving discrimination between visually | - | - | timeout |
| e011 | e010 | inference | The 224-resolution model timed out because inference (4 TTA passes at 224px) needs ~84s but only had 50s reserved; increasing RESERVE_S to 90s will let it finish, and the extra resolution should help  | 0.6379 | 0.6272 | not confirmed |
| e012 | e008 | architecture | tf_efficientnetv2_b0's improved normalization (BatchNorm with momentum fix) and SE attention blocks should improve fine-grained discrimination between visually similar defect types (especially pore vs | 0.6417 | 0.6373 | new best |
| e013 | e012 | augmentation | Mixup regularization (interpolating pairs of training crops with soft labels) will smooth decision boundaries between visually similar defect types (pore/stain/deposit), improving generalization and d | 0.6442 | 0.6371 | not confirmed |
| e014 | e012 | augmentation | Adding random rotation (±15°) to training augmentation will improve the model's invariance to weld bead orientation, helping the weak pore (0.517) and stain (0.616) classes that may be sensitive to th | 0.6372 | - | rejected |
| e015 | e012 | architecture | ResNet-50's deeper residual architecture (~25M params vs ~12M for tf_efficientnetv2_b0) provides more representational capacity for fine-grained discrimination between visually similar defect types (p | - | - | timeout |
| e016 | e012 | augmentation | Random erasing (stochastic masking of small regions during training) will act as a regularizer that prevents the model from over-relying on specific local visual patterns, improving generalization esp | 0.6273 | - | rejected |
| e017 | e012 | scoring | False-alarm boxes tend to have lower detector confidence than true defect boxes; squaring the confidence (conf^2 instead of conf) amplifies this gap and improves the precision-recall ranking, which di | 0.6292 | - | rejected |
| e018 | e012 | optimisation | Replacing AdamW (lr=1e-3, wd=0.05) with SGD+momentum (lr=0.05, momentum=0.9, nesterov, wd=1e-4) will improve generalization of the fine-tuned features; SGD's larger effective steps and lower weight de | 0.6410 | 0.6337 | not confirmed |
| e019 | e012 | augmentation | Adding random contrast variation and Gaussian noise to training augmentation will make the classifier more robust to lighting/sensor variations in weld images, reducing over-reliance on specific color | - | - | crashed |
| e020 | e012 | architecture | The wide crop (2.5× context) provides more discriminative weld-bead pattern information that helps distinguish visually similar defect types (pore vs stain); weighting it more heavily (0.7 wide vs 0.3 | 0.6397 | - | unconfirmed (out of time) |

Each experiment's folder (/data/ar2/runs/ar2-1/eNNN/) holds train.py, diff.patch, the prompt, the LLM's reply and thinking, and the run logs.
