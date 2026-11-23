# AutoResearch report: b4

- Task: det (mAP@0.5); LLM: **qwen3.8-27b-D_gpu5_mtp3** (thinking: xhigh), training budget 180.0 s wall time per run, 5 seeds per confirmed result
- Started 2026-11-03 01:08:08, report written 2026-11-03 05:49:57
- Harness: brief ar2_program_v21.md; acceptance gain > 1.0 SE; second seed for close calls; lessons table in prompt
- Ideas tried: 18 | edits unusable: 0 | crashed/timeout/bad output: 3 | likely bugs (score far below best): 0 | ran but worse: 11 | not confirmed: 4 | **confirmed improvements: 0**
- LLM time: 2.7 h in 22 calls (447 s per call); training time: 2.0 h

| category | ideas | ran | confirmed |
|---|---|---|---|
| data | 8 | 5 | 0 |
| scoring | 4 | 4 | 0 |
| inference | 3 | 3 | 0 |
| architecture | 1 | 1 | 0 |
| augmentation | 1 | 1 | 0 |
| sampling | 1 | 1 | 0 |

## Final evaluation (hidden test set, read only here)

| version | seeds | val mAP@0.5 | test mAP@0.5 | test gain vs start |
|---|---|---|---|---|
| start file | 8 | 0.6280 ± 0.0024 | 0.6871 ± 0.0014 | - |

Values are mean ± standard error over seeds. A gain above about 2 SE is unlikely to be seed noise.

## All experiments

| id | base | category | hypothesis | seed-0 | 5-seed mean | outcome |
|---|---|---|---|---|---|---|
| e000 | - | - | starting file | 0.6302 | 0.6297 | baseline |
| e001 | e000 | data | Since every eval box is emitted with one row per class, the hit sets per class are already fixed by stage 1 (same as the 0.758 ceiling); the 0.63 vs 0.758 gap is therefore a *ranking* problem, and the | - | - | crashed |
| e002 | e001 | data | The e001 run never actually tested its idea: it crashed with a NameError on `n` because the prediction-writing block was accidentally deleted, so no preds.csv was ever scored. Restoring that writer (e | 0.5541 | - | rejected |
| e003 | e000 | inference | The 0.630 vs 0.758-ceiling gap is a within-class ranking problem, and the classifier's per-class probability is the only signal that separates true-c boxes from non-c boxes; averaging 4 views (tight+w | 0.6355 | 0.6301 | not confirmed |
| e004 | e000 | scoring | The 0.630 vs 0.758-ceiling gap is dominated by class-decision error, and the current code emits 4 rows (one per class) per box, so every box injects 3 false-positive rows into the other classes' ranke | 0.5689 | - | rejected |
| e005 | e000 | architecture | The 0.630 vs 0.758-ceiling gap is a class-decision/ranking problem concentrated in pore (0.504) and stain (0.609), and the crop classifier's macro-F1 (~0.87) is the bottleneck, so swapping the backbon | 0.6240 | - | rejected |
| e006 | e000 | data | The 0.630 vs 0.758-ceiling gap is a ranking problem driven by noisy per-class probabilities p_c, and the classifier is the bottleneck (pore 0.504, stain 0.609). The current recipe trains only on high- | 0.6225 | - | rejected |
| e007 | e000 | scoring | The 0.630 vs 0.755 class-agnostic gap is cross-class contamination: every box injects a row into all four class lists, so a true deposit box is a false positive in the pore list, and the only thing th | 0.6192 | - | rejected |
| e008 | e000 | augmentation | Apply mixup (α=0.4) during training, blending pairs of crop views with their corresponding soft class targets. This regularizes the classifier and sharpens its decision boundaries between the hard cla | 0.6356 | 0.6327 | not confirmed |
| e009 | e000 | sampling | The classifier is trained on ground-truth crops but scored on stage-1 detector boxes (slightly misplaced, plus ~25% false alarms), a train/test distribution shift that corrupts the per-class probabili | 0.6268 | - | rejected |
| e010 | e000 | scoring | The 0.630 vs 0.758 gap is dominated by the fact that every box appears in all four class lists, so each class list contains ~41k entries of which only a few hundred are true positives. The current sco | 0.6289 | - | rejected |
| e011 | e000 | inference | The 0.630 vs 0.758-ceiling gap is a within-class ranking problem, and the classifier's per-class probability is the only class signal, so I want the most accurate p_c per box. Instead of averaging the | 0.6253 | - | rejected |
| e012 | e000 | data | The 0.630 vs 0.758-ceiling gap lives in the class decision, and the crop normalizes away the one physically informative cue the model currently cannot see: the detector box's absolute size (pores are  | - | - | crashed |
| e013 | e012 | data | e012's size/aspect-channel idea was sound but crashed on a wrong attribute name (timm's EfficientNet exposes the first conv as `model.conv_stem`, not `model.stem.conv`). Pore is the weakest class (0.5 | - | - | crashed |
| e014 | e000 | data | The class decision is the clear bottleneck (single-class 0.755 vs our 0.630, ceiling 0.758) and pore is the weakest class (0.504) largely because the crop normalizes away the box's absolute size, a st | 0.6314 | 0.6273 | not confirmed |
| e015 | e000 | data | For small boxes (pores, the weakest class at 0.504), the tight (1.5x) and wide (2.5x) crops are both clamped to MIN_CROP=16, so the two views are *identical* and the two-crop design gives the model no | 0.6358 | 0.6275 | not confirmed |
| e016 | e000 | scoring | The 0.630→0.758 gap is a within-class ranking problem driven by the classifier's per-class probability p_c, and cross-entropy classifiers are typically under-confident (probabilities pulled toward the | 0.6183 | - | rejected |
| e017 | e000 | data | The 0.630→0.758 gap is a within-class ranking problem, and a key corruptor is that the 4-class classifier was never shown "no defect" crops, so false-alarm detector boxes (no real defect) get non-triv | 0.5177 | - | rejected |
| e018 | e000 | inference | The reserve is set to 55 s but the seed-0 run only spent ~37 s on prediction (125 s→162 s), leaving ~18 s of slack that is currently wasted. Reducing RESERVE_S to 40 s buys ~15 s of extra training (ro | 0.6263 | - | rejected |

Each experiment's folder (/data/ar2/runs/b4/eNNN/) holds train.py, diff.patch, the prompt, the LLM's reply and thinking, and the run logs.
