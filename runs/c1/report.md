# AutoResearch report: c1

- Task: det (mAP@0.5); LLM: **qwen3.8-27b-D_gpu5_mtp3** (thinking: xhigh), training budget 180.0 s wall time per run, 5 seeds per confirmed result
- Started 2026-11-03 17:32:02, report written 2026-11-03 22:42:12
- Harness: brief ar2_program_v3.md; acceptance gain > 1.0 SE; second seed for close calls; lessons table in prompt; diagnostics in prompt; background digest ar2_digest.md; start-file runs reused from b5; uncertainty includes validation images (bootstrap); memory of earlier runs ar_memory.md
- Ideas tried: 19 | edits unusable: 0 | crashed/timeout/bad output: 7 | likely bugs (score far below best): 1 | ran but worse: 3 | not confirmed: 7 | **confirmed improvements: 1**
- LLM time: 2.8 h in 21 calls (474 s per call); training time: 2.7 h

| category | ideas | ran | confirmed |
|---|---|---|---|
| data | 9 | 4 | 0 |
| architecture | 6 | 4 | 1 |
| optimisation | 2 | 2 | 0 |
| loss | 2 | 2 | 0 |

## Final evaluation (hidden test set, read only here)

| version | seeds | val mAP@0.5 | test mAP@0.5 | test gain vs start |
|---|---|---|---|---|
| start file | 8 | 0.6274 ± 0.0018 | 0.6869 ± 0.0015 | - |
| final best (e019) | 8 | 0.6400 ± 0.0020 | 0.6857 ± 0.0022 | -0.0012 ± 0.0027 (-0.4 SE) |

Values are mean ± standard error over seeds. A gain above about 2 SE is unlikely to be seed noise.

Uncertainty of the final gain over images (paired bootstrap, same seeds): validation ±0.0047, test ±0.0034. Seeds and images together, the test gain's SE is 0.0043.

Per-class val score, start file: pore 0.502, deposit 0.624, discontinuity 0.780, stain 0.609
Per-class val score, final best e019: pore 0.524, deposit 0.629, discontinuity 0.784, stain 0.609

Chain of accepted changes: e000 -> e018 -> e019

## What changed from the start file to the final best

```diff
--- start/train.py
+++ e019/train.py
@@ -44,7 +44,7 @@
 MODEL = "efficientnet_b0"
 PRETRAINED = True
-IMG_SIZE = 160
-CONTEXT = 2.5           # wide crop: side = CONTEXT * long side of the box
-TIGHT_CONTEXT = 1.5     # tight crop
+IMG_SIZE = 224
+CONTEXT = 4.0           # wide context crop: side = CONTEXT * long side of the box
+MAX_TIGHT = 192         # native-scale tight region clamped to this many px
 MIN_CROP = 16
 TRAIN_ON = ("high",)    # resolutions of crops.csv used for training
@@ -86,4 +86,44 @@
 
 
+def make_composite(img, r, train):
+    """Single 224x224 input: the detector box region at NATIVE scale (top-left) plus a wide context crop.
+    The tight region keeps its real pixels (no rescale-to-fixed-size), so small defects are not blurred."""
+    h, w = img.shape[:2]
+    x1, y1, x2, y2 = r["x1"], r["y1"], r["x2"], r["y2"]
+    if train:  # light geometric jitter on the box
+        dx, dy = random.uniform(-0.1, 0.1) * (x2 - x1), random.uniform(-0.1, 0.1) * (y2 - y1)
+        x1, y1, x2, y2 = x1 + dx, y1 + dy, x2 + dx, y2 + dy
+    # native-scale tight region, clamped to a sane max so huge boxes do not dominate
+    tw = max(2, int(round(x2 - x1)))
+    th = max(2, int(round(y2 - y1)))
+    scale = 1.0
+    if max(tw, th) > MAX_TIGHT:
+        scale = MAX_TIGHT / max(tw, th)
+        tw, th = int(round(tw * scale)), int(round(th * scale))
+    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
+    tx0, ty0 = int(round(cx - tw / 2)), int(round(cy - th / 2))
+    tight = np.full((th, tw, 3), 114, dtype=np.uint8)
+    sx0, sy0, sx1, sy1 = max(0, tx0), max(0, ty0), min(w, tx0 + tw), min(h, ty0 + th)
+    if sx1 > sx0 and sy1 > sy0:
+        tight[sy0 - ty0:sy1 - ty0, sx0 - tx0:sx1 - tx0] = img[sy0:sy1, sx0:sx1]
+    tight = np.asarray(Image.fromarray(tight).resize((tw, th), Image.BILINEAR), dtype=np.uint8)
+    # wide context crop at fixed context
+    wide = make_crop(img, r, CONTEXT, train)  # already 224x224 float in [0,1]
+    wide_u8 = np.clip(wide * 255.0, 0, 255).astype(np.uint8)
+    comp = np.full((IMG_SIZE, IMG_SIZE, 3), 114, dtype=np.uint8)
+    comp[:th, :tw] = tight
+    comp[th:, tw:] = wide_u8[th:, tw:]  # wide context fills the rest (both are IMG_SIZE square)
+    x = comp.astype(np.float32) / 255.0
+    if train:
+        bright = random.uniform(0.8, 1.2)
+        x = np.clip(x * bright, 0, 1)
+    x = (x - MEAN) / STD
+    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
+
+
+def flip_tensor(x, h):
+    return torch.flip(x, [1]) if h else x
+
+
 class BoxDataset(torch.utils.data.Dataset):
     def __init__(self, rows, images, train):
@@ -96,20 +136,11 @@
         r = self.rows[i]
         img = self.images[r["image"]]
-        fh = fv = False
-        bright = 1.0
-        if self.train:  # same flips and brightness for both views
-            fh, fv, bright = random.random() < 0.5, random.random() < 0.5, random.uniform(0.8, 1.2)
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
+        x = make_composite(img, r, self.train)
+        if self.train:  # same flips for the composite
+            if random.random() < 0.5:
+                x = flip_tensor(x, True)
+            if random.random() < 0.5:
+                x = flip_tensor(x, False)
+        return x, r.get("cls", -1)
 
 
@@ -139,8 +170,8 @@
 
     g = torch.Generator().manual_seed(args.seed)
-    dl_tr = torch.utils.data.DataLoader(BoxDataset(crops, images, True), batch_size=BATCH_SIZE, shuffle=True,
+    dl_tr = torch.utils.data.DataLoader(BoxDataset(crops, images, True), batch_size=48, shuffle=True,
                                         num_workers=workers, pin_memory=True, drop_last=True,
                                         persistent_workers=True, generator=g, worker_init_fn=worker_init)
-    dl_ev = torch.utils.data.DataLoader(BoxDataset(ev, images, False), batch_size=256, shuffle=False,
+    dl_ev = torch.utils.data.DataLoader(BoxDataset(ev, images, False), batch_size=128, shuffle=False,
                                         num_workers=workers, pin_memory=True)
 
@@ -159,5 +190,5 @@
     while not done:
         tot, nb = 0.0, 0
-        for xt, xw, y in dl_tr:
+        for xt, y in dl_tr:
             frac = min(1.0, (time.time() - t_start) / budget)
             f = frac / 0.05 if frac < 0.05 else 0.5 * (1 + math.cos(math.pi * (frac - 0.05) / 0.95))
@@ -165,8 +196,7 @@
                 pg["lr"] = LR * max(f, 1e-3)
             xt = xt.to(device, non_blocking=True).to(memory_format=torch.channels_last)
-            xw = xw.to(device, non_blocking=True).to(memory_format=torch.channels_last)
             y = y.to(device, non_blocking=True)
             with torch.autocast(**amp):
-                loss = crit((model(xt) + model(xw)) / 2, y)
+                loss = crit(model(xt), y)
             opt.zero_grad(set_to_none=True)
             loss.backward()
@@ -183,8 +213,7 @@
     probs = []
     with torch.no_grad(), torch.autocast(**amp):
-        for xt, xw, _ in dl_ev:
+        for xt, _ in dl_ev:
             xt = xt.to(device).to(memory_format=torch.channels_last)
-            xw = xw.to(device).to(memory_format=torch.channels_last)
-            probs.append(torch.softmax(((model(xt) + model(xw)) / 2).float(), 1).cpu())
+            probs.append(torch.softmax(model(xt).float(), 1).cpu())
     probs = torch.cat(probs).numpy()
     os.makedirs(args.out, exist_ok=True)
```

## All experiments

| id | base | category | hypothesis | seed-0 | 5-seed mean | outcome |
|---|---|---|---|---|---|---|
| e000 | - | - | starting file | 0.6314 | 0.6288 | baseline |
| e001 | e000 | architecture | Replace the two separate 160×160 crops (each through the model, logits averaged) with a single 224×224 image formed by placing the tight (1.5×) and wide (2.5×) crops side-by-side at 112×112 each, so t | 0.6323 | 0.6275 | not confirmed |
| e002 | e000 | data | Feed each box's long-side size and aspect ratio to the classifier as two extra constant input channels (in_chans=5) so the network can use a size/shape prior alongside appearance. The diagnostics show | - | - | crashed |
| e003 | e000 | data | Add the detector's own train-split boxes (IoU > 0.5 with a labelled crop, labelled by that crop's class) to stage-2 training, doubling the training set to boxes drawn from the exact distribution stage | 0.6356 | 0.6295 | not confirmed |
| e004 | e000 | optimisation | Add exponential moving average (EMA) of the weights: keep a shadow copy of the model, update it every step (decay 0.999), and use the EMA weights for the final eval predictions while training on the l | 0.3561 | - | suspicious (likely bug) |
| e005 | e000 | data | The dominant error is wrong classes (perfect-class scores would give 0.746 vs 0.629), driven by stain (the ~40% majority class) absorbing pore and deposit, and pores being the weakest class (AP 0.502) | 0.6310 | 0.6301 | not confirmed |
| e006 | e000 | data | The diagnostic shows the dominant error is wrong classes (0.7461 with perfect classes vs 0.6288 = 0.117 lost), concentrated in deposit→stain (55), stain→deposit (46) and pore→stain (20). These classes | - | - | crashed |
| e007 | e000 | data | Give the classifier absolute box geometry it currently cannot see: the crop is always rescaled so the box fills a fixed fraction of the 160px image, so absolute long-side size, aspect ratio and detect | 0.6215 | - | rejected |
| e008 | e000 | loss | Replace the 4-way softmax cross-entropy with multi-label BCE on 4 independent sigmoid heads (score = conf x sigmoid(logit_c)), so each class is learned as "present vs absent" instead of competing in a | 0.6298 | 0.6296 | not confirmed |
| e009 | e000 | data | Replace the fixed ImageNet global normalization with per-crop, per-channel normalization (subtract the crop's own mean, divide by its own std). Industrial weld images almost certainly have varying ill | - | - | timeout |
| e010 | e000 | loss | Replace the plain cross-entropy with a class-margin (triplet-style) loss that, for each crop, pulls the logit of its true class above the logits of the two most-confused classes (deposit↔stain, pore→s | 0.6183 | - | rejected |
| e011 | e000 | architecture | Replace logit averaging of the two views with feature-level concatenation: strip the backbone's classifier head, concatenate the 1280-dim features from the tight (1.5×) and wide (2.5×) views into a 25 | - | - | crashed |
| e012 | e011 | architecture | Fix the crash in e011 (the `model.train()` line referenced a name that no longer exists after splitting the model into backbone + head) and run the feature-concatenation variant: the two views' 1280-d | 0.6365 | 0.6313 | not confirmed |
| e013 | e000 | architecture | Add a third view: a 1.0x "texture" crop of the box (the box fills the whole 160px frame), third logit averaged with the 1.5x and 2.5x views. The dominant error is wrong classes (0.7461 with perfect cl | 0.6267 | - | rejected |
| e014 | e000 | data | Pore is the weakest class (AP 0.502, only 63% of overlapping boxes get the right class, mostly misread as stain) and it is also the rarest, so I add copy-paste augmentation: for pore training rows, wi | - | - | crashed |
| e015 | e014 | data | e014's copy-paste pore augmentation crashed on a size bug (the pasted tile could exceed the target image's height, e.g. a 201px tile on a 177px-tall image). I fix it by clamping the paste size to the  | 0.6355 | 0.6269 | not confirmed |
| e016 | e000 | data | The dominant error is wrong classes (0.7461 with perfect classes vs 0.6288), concentrated in stain being a "magnet" (deposit→stain 55, pore→stain 20) and deposit↔stain confusion (46). These classes di | - | - | timeout |
| e017 | e000 | optimisation | Replace AdamW (lr=1e-3) with SGD+momentum (lr=0.01, momentum=0.9, weight_decay=0.05) for fine-tuning the pretrained EfficientNet-B0. SGD with momentum is the standard fine-tuning optimizer in the tran | 0.6409 | 0.6318 | not confirmed |
| e018 | e000 | architecture | The dominant error is wrong classes (0.7461 with perfect classes vs 0.6288), concentrated in stain acting as a magnet for deposit (55) and pore (20) and deposit↔stain (46). I replace the two fixed-con | - | - | crashed |
| e019 | e018 | architecture | e018's composite input (native-scale tight region + wide context in one 224×224 image) crashed purely on a flip bug (`x[:, ::-1, :].contiguous()` raised "step must be greater than zero"); I fix it wit | 0.6340 | 0.6367 | new best |

Each experiment's folder (/data/ar2/runs/c1/eNNN/) holds train.py, diff.patch, the prompt, the LLM's reply and thinking, and the run logs.

## Provenance

| item | value |
|---|---|
| python | 3.12.3 |
| start_file | cb33b09ac6afffea0b9ba15234a362367d84fd3cc4ed8bedd4d9832b7e39c334 |
| brief | 023373b012478d21c104fdfb0eba879b16f844d85bd2dd27e1cb51af24694a29 |
| digest | fcd9348fd7f0eeb239881bfb36b5e2be8db268350f16dc525a50e441f1828906 |
| ar_loop.py | d5efad8c452028162570fd660b55fd29e2ba769ed7a0e96abeafde3f74c02487 |
| det_metric.py | b055a87c2167735066dbb01f68e4027a09f768f4145f1bd58db2c33ac2685ce1 |
| data/dets.csv | a33042aa47ad358ab6e5700611ccc55a139cf20a5a5589ef8e2524e0abb6d43e |
| data/crops.csv | 7fa6269ca75cb4410aea5057138dc0ca73b5744ff04060893155025b338fb953 |
| hidden labels | 1fd0b668e18f41df8fdbee4d7e0502a806cd255e266637bf8455cbcd533d5f99 |
| model_file | /data/models/gguf/byteshape/Qwen3.8-27B-IQ4_XS-3.84bpw.gguf |
| model_file_sha | fingerprint:5940e8d36a690b185139d0420739651ae44db7676dbb1c12aff458f30a1ff56a |
| llama_cpp_commit | 0ec4027a2a0183de87c007069adddc06e665f7de |
| packages_sha | 5a08c360913eeb3cd7195654571716f10f71401aa4df2ba9515f762380b0a662 |
| key_packages | numpy 2.5.4, pillow 13.0.0, timm 1.0.30, torch 2.14.1, torchvision 0.29.1 |
| train.py of start / final best | - / 06149a684a9c257c2be0d0f8e746e725ccfd30b5cf56f58e63e4c749c2bdaa24 |
| start-file runs reused from | b5 |
