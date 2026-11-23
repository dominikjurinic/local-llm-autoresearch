# AutoResearch report: g2

- Task: det (mAP@0.5); LLM: **qwen3.8-27b-D_gpu5_mtp3** (thinking: xhigh), training budget 180.0 s wall time per run, 5 seeds per confirmed result
- Started 2026-11-13 12:57:59, report written 2026-11-13 20:09:16
- Harness: brief ar2_program_v3.md; acceptance gain > 1.0 SE; second seed for close calls; lessons table in prompt; diagnostics in prompt; background digest ar2_digest.md; start-file runs reused from b5; uncertainty includes validation images (bootstrap); memory of earlier runs ar_memory.md; research guide ar_guide.md
- Ideas tried: 32 | edits unusable: 0 | crashed/timeout/bad output: 17 | likely bugs (score far below best): 1 | ran but worse: 10 | not confirmed: 3 | **confirmed improvements: 1**
- LLM time: 4.8 h in 38 calls (452 s per call); training time: 2.7 h

| category | ideas | ran | confirmed |
|---|---|---|---|
| inference | 13 | 5 | 0 |
| architecture | 10 | 5 | 1 |
| data | 4 | 3 | 0 |
| optimisation | 2 | 1 | 0 |
| augmentation | 1 | 1 | 0 |
| other | 1 | 0 | 0 |
| sampling | 1 | 0 | 0 |

## Final evaluation (hidden test set, read only here)

| version | seeds | val mAP@0.5 | test mAP@0.5 | test gain vs start |
|---|---|---|---|---|
| start file | 8 | 0.6274 ± 0.0018 | 0.6869 ± 0.0015 | - |
| final best (e012) | 8 | 0.6334 ± 0.0019 | 0.6839 ± 0.0014 | -0.0030 ± 0.0021 (-1.4 SE) |

Values are mean ± standard error over seeds. A gain above about 2 SE is unlikely to be seed noise.

Uncertainty of the final gain over images (paired bootstrap, same seeds): validation ±0.0037, test ±0.0025. Seeds and images together, the test gain's SE is 0.0033.

Per-class val score, start file: pore 0.502, deposit 0.624, discontinuity 0.780, stain 0.609
Per-class val score, final best e012: pore 0.546, deposit 0.615, discontinuity 0.778, stain 0.606

Chain of accepted changes: e000 -> e012

## What changed from the start file to the final best

```diff
--- start/train.py
+++ e012/train.py
@@ -47,4 +47,5 @@
 CONTEXT = 2.5           # wide crop: side = CONTEXT * long side of the box
 TIGHT_CONTEXT = 1.5     # tight crop
+WIDE2_CONTEXT = 4.0     # very wide crop (extra scale for fine-grained class cues)
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
+        for xt, xw, xw2, y in dl_tr:
             frac = min(1.0, (time.time() - t_start) / budget)
             f = frac / 0.05 if frac < 0.05 else 0.5 * (1 + math.cos(math.pi * (frac - 0.05) / 0.95))
@@ -166,7 +167,8 @@
             xt = xt.to(device, non_blocking=True).to(memory_format=torch.channels_last)
             xw = xw.to(device, non_blocking=True).to(memory_format=torch.channels_last)
+            xw2 = xw2.to(device, non_blocking=True).to(memory_format=torch.channels_last)
             y = y.to(device, non_blocking=True)
             with torch.autocast(**amp):
-                loss = crit((model(xt) + model(xw)) / 2, y)
+                loss = crit((model(xt) + model(xw) + model(xw2)) / 3, y)
             opt.zero_grad(set_to_none=True)
             loss.backward()
@@ -183,8 +185,9 @@
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
| e000 | - | - | starting file | 0.6314 | 0.6288 | baseline |
| e001 | e000 | data | Add the stage-1 detector boxes from the train split to the training set (labelled by the true box they overlap with IoU > 0.5, computed vectorised per image), alongside the true boxes. Stage 2 is scor | - | - | crashed |
| e002 | e001 | data | Fix the crash in e001 (numpy's `np.clip` requires three arguments: `a_min` and `a_max`) and re-run the same idea: add the stage-1 detector boxes from the train split to training, labelled by the true  | 0.6282 | - | rejected |
| e003 | e000 | augmentation | Replace the small crop jitter with a targeted simulation of stage-1 box error: during training, with probability 0.5 each box's center is shifted by up to ~0.45x its long side in a random direction an | 0.6259 | - | rejected |
| e004 | e000 | architecture | Replace the logit-averaging of two independent EfficientNet-B0 forward passes with a shared backbone whose 1280-dim pooled features from the tight and wide crops are concatenated (2560-dim) before a s | 0.6345 | 0.6329 | not confirmed |
| e005 | e000 | data | Precompute the detector's real box-error distribution from the train split (center offset and side scale of each det box vs the true box it overlaps, IoU>0.5) and, during training, jitter true boxes t | 0.6275 | - | rejected |
| e006 | e000 | inference | The log shows prediction only takes ~37 s (125 s→162 s) but RESERVE_S=55 over-reserves by ~18 s, while the loss is still falling at the cut (0.219 at step 433). Lowering the reserve to 45 (37 s measur | 0.6253 | - | rejected |
| e007 | e000 | architecture | Replace the plain logit-average of the two views with a shared backbone whose tight and wide 1280-dim pooled features are each passed through a small 1280->128 MLP, then combined by element-wise addit | - | - | crashed |
| e008 | e000 | architecture | Feed each box's absolute size (log of long side) and aspect ratio as two extra inputs added to the classifier logits via a small zero-initialised head. The crops resize every box to 160 px, so the mod | 0.6251 | 0.6279 | not confirmed |
| e009 | e000 | optimisation | Split the optimizer into two param groups: the pretrained EfficientNet-B0 backbone at a low LR (1e-4) and the random 4-class head at the full LR (1e-3), instead of the current uniform 1e-3. The diagno | 0.5854 | - | rejected |
| e010 | e000 | inference | Precompute every training crop (tight + wide, at the exact box position) once after loading, store the 160px arrays in memory, and serve them from the dataset so the per-step work is only the cheap fl | - | - | timeout |
| e011 | e010 | inference | The e010 timeout was caused by precomputing crops for the 41,879 eval boxes as well as the 6,052 training boxes (~129 s total, ~1.35 ms/crop). Precomputing only the training crops costs ~16 s but remo | 0.6254 | - | rejected |
| e012 | e000 | architecture | Add a third, very wide crop (4x the box long side) alongside the existing 1.5x and 2.5x views, with all three logits averaged. The dominant remaining error is the class decision on detector boxes (0.7 | 0.6336 | 0.6363 | new best |
| e013 | e012 | architecture | Replace the equal logit average of the three views with a learned weighted combination: each view's 4-dim logits pass through a small 4→1 linear layer (initialised to 1/3) and the outputs are summed.  | 0.6326 | - | rejected |
| e014 | e012 | other | Add a 4th view: the tight 1.5x box crop enlarged to a fixed 320 px (2x the current 160 px), so the object's native pixels are preserved instead of being squeezed into 160 px while the wide views keep  | - | - | timeout |
| e015 | e012 | architecture | Replace EfficientNet-B0 with ConvNeXt-Nano. ConvNeXt's depthwise convolutions and layer-normalisation are better suited to fine-grained texture discrimination (the dominant stain/deposit/pore confusio | - | - | timeout |
| e016 | e012 | inference | The current run spends 14 s loading + ~111 s training + ~60 s prediction (185 s total, only ~5 s under the kill point), and the loss is still falling at the cut (0.242). The prediction phase re-crops  | - | - | timeout |
| e017 | e012 | inference | Train on all three views but at inference drop the 4x wide view, keeping only the 1.5x and 2.5x crops (logit average of the two). The log shows the 3-view prediction phase costs ~60 s against a 55 s r | - | - | timeout |
| e018 | e012 | inference | e017 intended to drop the 4x view at inference to buy training time, but the dataset still computes all three crops per eval box, so the prediction phase kept its full ~60 s cost and the run was kille | - | - | crashed |
| e019 | e018 | inference | e018 intended to drop the 4x view at inference (saving ~20 s of the ~60 s prediction phase) to buy ~15-20 extra training steps, but it crashed because the dataset returned `None` for the third view at | 0.6176 | - | rejected |
| e020 | e012 | data | Add the stage-1 detector boxes from the train split to training, labelled by the true box they overlap (IoU > 0.5, computed vectorised per image with numpy only — 3-arg `np.clip`, which is what crashe | 0.6229 | - | rejected |
| e021 | e012 | architecture | Replace the equal logit-average of three views with a shared backbone whose pooled features from the tight (1.5x) and wide (2.5x) crops are concatenated (2560-dim) before a small two-layer head, while | - | - | crashed |
| e022 | e021 | architecture | e021 crashed only because `backbone.forward_features` returns 4-D (B,1280,5,5) maps, so the concat fed a 2560x5x5 tensor into a 2560-in Linear. I fix it by applying the backbone's `global_pool` and fl | - | - | crashed |
| e023 | e022 | architecture | e021/e022 (shared backbone; tight+wide pooled features concatenated into a small head, 4x view on the stock classifier) crashed only on plumbing: timm's EfficientNet exposes no `head` attribute (the s | - | - | crashed |
| e024 | e012 | architecture | Stack the three crops (1.5x, 2.5x, 4x) into a single 9-channel input and run ONE EfficientNet-B0 forward pass per box (stem re-initialised by repeating the pretrained 3-channel kernel), instead of thr | 0.6341 | 0.6355 | not confirmed |
| e025 | e012 | inference | The per-step cost is dominated by on-the-fly crop/resize work in the data loader (~0.39 s/step, 285 steps in 111 s), and the loss is still falling at the cut (0.242). Precomputing the training crops o | - | - | crashed |
| e026 | e025 | inference | e025's precompute idea (build K jittered variants per training box once, serve them from a num_workers=0 loader so the per-step cost drops from ~0.39 s of crop/resize to a cheap flip/brightness on cac | 0.0997 | - | suspicious (likely bug) |
| e027 | e012 | optimisation | Replace the plain AdamW step with a Polyak/SWA-style weight average: after the first ~30% of the training budget, keep an exponential moving average of the weights (EMA decay 0.999) and use the EMA we | - | - | crashed |
| e028 | e012 | inference | The log shows the run ends at 185s with the loss still falling (0.242 at the cut) and the 55s reserve already exceeded by the ~60s prediction phase, so training is starved of steps. I free ~15s of the | 0.6178 | - | rejected |
| e029 | e012 | inference | The log shows the 55 s reserve is already exceeded (training stops at 125 s, the run ends at 185 s, so prediction takes ~60 s of ~61 s of headroom), and the loss is still falling at the cut (0.242). T | - | - | crashed |
| e030 | e012 | inference | e029's idea (precompute all 41,879 eval crop-sets once with a process pool, so the prediction phase is only normalise + forward) is sound but crashed on a pure plumbing bug: the crop function was a lo | - | - | crashed |
| e031 | e030 | inference | e030's eval-crop precompute crashed on a one-line scoping bug: `_mk_eval_crop` is module-level but reads `images`, which is local to `main`, so forked pool workers have no `images` (NameError). I make | - | - | timeout |
| e032 | e012 | sampling | Train the shared backbone on only the 1.5x and 2.5x views (dropping the 4x view from training) while keeping all three views at inference. The log shows the loss still falling at the cut (0.242 at 285 | - | - | crashed |

Each experiment's folder (/data/ar2/runs/g2/eNNN/) holds train.py, diff.patch, the prompt, the LLM's reply and thinking, and the run logs.

## Provenance

| item | value |
|---|---|
| python | 3.12.3 |
| start_file | cb33b09ac6afffea0b9ba15234a362367d84fd3cc4ed8bedd4d9832b7e39c334 |
| brief | 023373b012478d21c104fdfb0eba879b16f844d85bd2dd27e1cb51af24694a29 |
| digest | fcd9348fd7f0eeb239881bfb36b5e2be8db268350f16dc525a50e441f1828906 |
| ar_loop.py | 8381b8c2a5bb9bcc8b0e67acb896c12663a3b8923404244872def1c1b8df66ae |
| det_metric.py | b055a87c2167735066dbb01f68e4027a09f768f4145f1bd58db2c33ac2685ce1 |
| data/dets.csv | a33042aa47ad358ab6e5700611ccc55a139cf20a5a5589ef8e2524e0abb6d43e |
| data/crops.csv | 7fa6269ca75cb4410aea5057138dc0ca73b5744ff04060893155025b338fb953 |
| hidden labels | 1fd0b668e18f41df8fdbee4d7e0502a806cd255e266637bf8455cbcd533d5f99 |
| model_file | /data/models/gguf/byteshape/Qwen3.8-27B-IQ4_XS-3.84bpw.gguf |
| model_file_sha | fingerprint:5940e8d36a690b185139d0420739651ae44db7676dbb1c12aff458f30a1ff56a |
| llama_cpp_commit | 0ec4027a2a0183de87c007069adddc06e665f7de |
| packages_sha | 5a08c360913eeb3cd7195654571716f10f71401aa4df2ba9515f762380b0a662 |
| key_packages | numpy 2.5.4, pillow 13.0.0, timm 1.0.30, torch 2.14.1, torchvision 0.29.1 |
| train.py of start / final best | - / 2b5ceb67552de37e18b91f52cf9f3ba99e852d76112dab3c81e5262935f6f275 |
| start-file runs reused from | b5 |
