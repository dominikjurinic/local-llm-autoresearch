# AutoResearch report: g1

- Task: det (mAP@0.5); LLM: **qwen3.8-27b-D_gpu5_mtp3** (thinking: xhigh), training budget 180.0 s wall time per run, 5 seeds per confirmed result
- Started 2026-11-03 23:25:15, report written 2026-11-04 06:39:56
- Harness: brief ar2_program_v3.md; acceptance gain > 1.0 SE; second seed for close calls; lessons table in prompt; diagnostics in prompt; background digest ar2_digest.md; start-file runs reused from b5; uncertainty includes validation images (bootstrap); memory of earlier runs ar_memory.md; research guide ar_guide.md
- Ideas tried: 27 | edits unusable: 0 | crashed/timeout/bad output: 9 | likely bugs (score far below best): 1 | ran but worse: 7 | not confirmed: 9 | **confirmed improvements: 1**
- LLM time: 3.9 h in 33 calls (429 s per call); training time: 3.6 h

| category | ideas | ran | confirmed |
|---|---|---|---|
| data | 12 | 7 | 0 |
| optimisation | 7 | 4 | 0 |
| architecture | 6 | 6 | 1 |
| inference | 1 | 0 | 0 |
| loss | 1 | 1 | 0 |

## Final evaluation (hidden test set, read only here)

| version | seeds | val mAP@0.5 | test mAP@0.5 | test gain vs start |
|---|---|---|---|---|
| start file | 8 | 0.6274 ± 0.0018 | 0.6869 ± 0.0015 | - |
| final best (e005) | 8 | 0.6347 ± 0.0019 | 0.6920 ± 0.0014 | +0.0051 ± 0.0021 (+2.5 SE) |

Values are mean ± standard error over seeds. A gain above about 2 SE is unlikely to be seed noise.

Uncertainty of the final gain over images (paired bootstrap, same seeds): validation ±0.0024, test ±0.0025. Seeds and images together, the test gain's SE is 0.0032.

Per-class val score, start file: pore 0.502, deposit 0.624, discontinuity 0.780, stain 0.609
Per-class val score, final best e005: pore 0.505, deposit 0.639, discontinuity 0.781, stain 0.613

Chain of accepted changes: e000 -> e005

## What changed from the start file to the final best

```diff
--- start/train.py
+++ e005/train.py
@@ -114,4 +114,16 @@
 
 
+class TwoViewModel(nn.Module):
+    def __init__(self, backbone, head):
+        super().__init__()
+        self.backbone = backbone
+        self.head = head
+
+    def forward(self, xt, xw):
+        ft = self.backbone(xt)
+        fw = self.backbone(xw)
+        return self.head(torch.cat([ft, fw], dim=1))
+
+
 def main():
     ap = argparse.ArgumentParser()
@@ -145,6 +157,13 @@
                                         num_workers=workers, pin_memory=True)
 
-    model = timm.create_model(MODEL, pretrained=PRETRAINED, num_classes=4)
-    model = model.to(device).to(memory_format=torch.channels_last)
+    backbone = timm.create_model(MODEL, pretrained=PRETRAINED, num_classes=0)
+    backbone = backbone.to(device).to(memory_format=torch.channels_last)
+    head = nn.Sequential(
+        nn.Linear(2 * backbone.num_features, 128),
+        nn.ReLU(),
+        nn.Dropout(0.2),
+        nn.Linear(128, 4),
+    ).to(device)
+    model = TwoViewModel(backbone, head)
     crit = nn.CrossEntropyLoss()
     opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
@@ -168,5 +187,5 @@
             y = y.to(device, non_blocking=True)
             with torch.autocast(**amp):
-                loss = crit((model(xt) + model(xw)) / 2, y)
+                loss = crit(model(xt, xw), y)
             opt.zero_grad(set_to_none=True)
             loss.backward()
@@ -186,5 +205,5 @@
             xt = xt.to(device).to(memory_format=torch.channels_last)
             xw = xw.to(device).to(memory_format=torch.channels_last)
-            probs.append(torch.softmax(((model(xt) + model(xw)) / 2).float(), 1).cpu())
+            probs.append(torch.softmax(model(xt, xw).float(), 1).cpu())
     probs = torch.cat(probs).numpy()
     os.makedirs(args.out, exist_ok=True)
```

## All experiments

| id | base | category | hypothesis | seed-0 | 5-seed mean | outcome |
|---|---|---|---|---|---|---|
| e000 | - | - | starting file | 0.6314 | 0.6288 | baseline |
| e001 | e000 | data | I add the train-split detector boxes that overlap a true box (IoU>0.5, labelled with that true box's class) as extra training rows alongside the true boxes. The dominant error is the class decision on | - | - | crashed |
| e002 | e001 | data | Fix the broadcasting bug in the IoU computation that crashed e001. The area terms were computed as 1-D arrays and broadcast incorrectly against the 2-D intersection matrix. Computing each area as a co | 0.6330 | 0.6307 | not confirmed |
| e003 | e000 | data | I replace the mild crop jitter (center ±0.1 of crop side, scale 0.85–1.2, i.e. effective box IoU only ~0.75–1.0) with a stage-1 error simulation: per training box, one shared center shift (magnitude u | 0.6342 | 0.6294 | not confirmed |
| e004 | e000 | optimisation | Pre-compute all training crops (both views) as uint8 tensors at load time so the DataLoader no longer performs PIL extract+resize per step. This should cut the first-epoch overhead (46 s vs 18 s for l | 0.0531 | - | suspicious (likely bug) |
| e005 | e000 | architecture | Replace the logit-averaging head with a small MLP on concatenated backbone features from both views. Currently the two crops are classified independently and their 4-dim logits are averaged, which pre | 0.6426 | 0.6345 | new best |
| e006 | e005 | architecture | I widen the shared-head MLP from 2560→128→4 to 2560→512→512→4 (two hidden layers, 512 wide). e005 showed that letting the head interact the two views' features (vs. averaging 4-dim logits) is worth +0 | 0.6338 | - | rejected |
| e007 | e005 | architecture | I add two scalar inputs to the head, log(box long side) and log(aspect ratio), normalized by statistics measured on the training boxes. Because every crop is resized to 160x160, the model currently ca | 0.6337 | 0.6317 | not confirmed |
| e008 | e005 | data | Replace the fixed 1.5x/2.5x views with a per-sample random tight zoom (uniform 1.0–1.8) and random wide zoom (uniform 2.0–4.0), sharing the same center and jitter per view pair, so the classifier is t | - | - | crashed |
| e009 | e005 | data | I add the train-split detector boxes that overlap a true box (IoU>0.5, labelled with that true box's class) as extra training rows alongside the true boxes. The dominant error is the class decision on | 0.6325 | - | rejected |
| e010 | e005 | data | I replace the current mild crop jitter (center ±0.1, scale 0.85–1.2, effective box IoU only ~0.75–1.0) with a stage-1 error simulation: per training box, one shared center shift (magnitude uniform 0–0 | 0.6380 | 0.6335 | not confirmed |
| e011 | e005 | inference | I add two horizontal flips of the two existing views at inference (4 views total, logits averaged per class, no retraining). The dominant error is the class decision on detector boxes (0.7464 vs 0.634 | - | - | timeout |
| e012 | e005 | optimisation | I pre-compute both training views (uint8) once at load time, so the DataLoader only does cheap normalization/flip/brightness per step instead of PIL extract+resize, and I shrink RESERVE_S 55->45 to ma | - | - | crashed |
| e013 | e005 | optimisation | I switch the training DataLoader to `num_workers=0` so the full `images` dict (1022 full-res images) is never pickled to 8 workers. The seed-0 log shows epoch 0 costs 44 s vs ~18 s for later epochs; t | - | - | timeout |
| e014 | e005 | optimisation | I stop loading all 1022 full-res images into a big in-memory dict (13 s up front, then shipped to 8 persistent workers) and instead decode each image on demand inside the workers via a bounded per-wor | 0.6329 | 0.6322 | not confirmed |
| e015 | e005 | optimisation | The steady-state step time (~0.20 s/step, 436 steps, loss still falling at 0.169 when training is cut) is inflated by the DataLoader shipping 39 MB of float32 normalised views per batch from 8 workers | 0.6209 | - | rejected |
| e016 | e005 | data | I add the train-split detector boxes that overlap a true box (IoU>0.5, labelled with that true box's class) as extra training rows alongside the true boxes. The dominant error is the class decision on | - | - | crashed |
| e017 | e016 | data | Fix the crash in e016: `np.asarray(cands)` on a list of dicts yields a 1-D object array, so `a[:, 0]` raised IndexError. Building a proper 2-D float array of (x1,y1,x2,y2,cls) before the vectorised Io | - | - | crashed |
| e018 | e005 | loss | I add a confused-class margin loss to the standard cross-entropy: for each sample, I penalise the logit gap between the true class and the classes it is most confused with (pore↔stain, stain↔deposit,  | 0.6356 | 0.6335 | not confirmed |
| e019 | e005 | optimisation | I replace the cosine-to-zero schedule with a two-phase schedule: cosine decay to 0.1*LR over the first ~70% of the budget, then a small constant LR (0.1*LR) during which I maintain a running average o | - | - | crashed |
| e020 | e019 | optimisation | Fix the e019 SWA crash by averaging only learnable float parameters (via `named_parameters()`) instead of the full `state_dict()` which includes Long-typed buffers (e.g. BatchNorm `num_batches_tracked | 0.6304 | - | rejected |
| e021 | e005 | data | I replace the current mild crop jitter (center ±0.1, scale 0.85-1.2, IoU ~0.75-1.0) with a stage-1 error simulation: per training box, one shared center shift (magnitude uniform 0-0.3 of the box long  | 0.6311 | - | rejected |
| e022 | e005 | architecture | Replace the naive concatenation of the two views' backbone features with a richer fusion that also feeds the element-wise product and element-wise difference of the two feature vectors into the head.  | 0.6389 | 0.6356 | not confirmed |
| e023 | e005 | architecture | I replace the two independent backbone passes with one shared backbone that processes the tight and wide crops as two tiles of a single 160x320 input (no per-view pooling, no concat), so the head sees | 0.6317 | - | rejected |
| e024 | e005 | data | I add a third training/inference view: the 2.5x wide crop with its central 1.5x region (the box interior) replaced by gray, so the head receives [interior, surroundings, interior+surroundings] and can | 0.6323 | - | rejected |
| e025 | e005 | architecture | I replace the single global-average-pooled backbone vector per view with a two-scale feature: the final pooled vector plus a pooled vector from an intermediate block (blocks[5], ~4x the spatial resolu | 0.6389 | 0.6364 | not confirmed |
| e026 | e005 | data | I replace the fixed 1.5x tight view with a per-sample random tight zoom (uniform 1.0–1.8) that shares the same center jitter as the wide view, so the classifier is trained across the full range of tig | 0.6399 | 0.6345 | not confirmed |
| e027 | e005 | data | I train the classifier exclusively on train-split detector boxes (IoU > 0.5 with a true box, labelled with that true box's class) instead of the true boxes: every training input then has the same slig | - | - | crashed |

Each experiment's folder (/data/ar2/runs/g1/eNNN/) holds train.py, diff.patch, the prompt, the LLM's reply and thinking, and the run logs.

## Provenance

| item | value |
|---|---|
| python | 3.12.3 |
| start_file | cb33b09ac6afffea0b9ba15234a362367d84fd3cc4ed8bedd4d9832b7e39c334 |
| brief | 023373b012478d21c104fdfb0eba879b16f844d85bd2dd27e1cb51af24694a29 |
| digest | fcd9348fd7f0eeb239881bfb36b5e2be8db268350f16dc525a50e441f1828906 |
| ar_loop.py | 6c81cd7ed3004cc5dac912f3348e783689e352080ed5f635c2301ed342a50bd8 |
| det_metric.py | b055a87c2167735066dbb01f68e4027a09f768f4145f1bd58db2c33ac2685ce1 |
| data/dets.csv | a33042aa47ad358ab6e5700611ccc55a139cf20a5a5589ef8e2524e0abb6d43e |
| data/crops.csv | 7fa6269ca75cb4410aea5057138dc0ca73b5744ff04060893155025b338fb953 |
| hidden labels | 1fd0b668e18f41df8fdbee4d7e0502a806cd255e266637bf8455cbcd533d5f99 |
| model_file | /data/models/gguf/byteshape/Qwen3.8-27B-IQ4_XS-3.84bpw.gguf |
| model_file_sha | fingerprint:5940e8d36a690b185139d0420739651ae44db7676dbb1c12aff458f30a1ff56a |
| llama_cpp_commit | 0ec4027a2a0183de87c007069adddc06e665f7de |
| packages_sha | 5a08c360913eeb3cd7195654571716f10f71401aa4df2ba9515f762380b0a662 |
| key_packages | numpy 2.5.4, pillow 13.0.0, timm 1.0.30, torch 2.14.1, torchvision 0.29.1 |
| train.py of start / final best | - / 6eb814f5dc4e4228a92a878a485b927350dfce2589f16a5b68262ac24a130072 |
| start-file runs reused from | b5 |
