# AutoResearch report: p1

- Task: det (mAP@0.5); LLM: **qwen3.8-27b-D_gpu5_mtp3** (thinking: xhigh), training budget 180.0 s wall time per run, 5 seeds per confirmed result
- Started 2026-11-05 16:54:42, report written 2026-11-05 23:34:37
- Harness: brief ar2_program_v3.md; acceptance gain > 1.0 SE; second seed for close calls; lessons table in prompt; diagnostics in prompt; background digest ar2_digest.md; start-file runs reused from b5; uncertainty includes validation images (bootstrap); memory of earlier runs ar_memory.md; PAIR MODE: sage-a1.5-q4km plans each experiment, qwen3.8-27b-D_gpu5_mtp3 implements it
- Ideas tried: 26 | edits unusable: 0 | crashed/timeout/bad output: 9 | likely bugs (score far below best): 0 | ran but worse: 14 | not confirmed: 2 | **confirmed improvements: 1**
- LLM time: 4.9 h in 56 calls (315 s per call); training time: 2.1 h

| category | ideas | ran | confirmed |
|---|---|---|---|
| architecture | 16 | 8 | 1 |
| data | 4 | 4 | 0 |
| inference | 2 | 2 | 0 |
| scoring | 2 | 2 | 0 |
| optimisation | 2 | 1 | 0 |

## Final evaluation (hidden test set, read only here)

| version | seeds | val mAP@0.5 | test mAP@0.5 | test gain vs start |
|---|---|---|---|---|
| start file | 8 | 0.6274 ± 0.0018 | 0.6869 ± 0.0015 | - |
| final best (e009) | 8 | 0.6351 ± 0.0019 | 0.6870 ± 0.0013 | +0.0001 ± 0.0020 (+0.0 SE) |

Values are mean ± standard error over seeds. A gain above about 2 SE is unlikely to be seed noise.

Uncertainty of the final gain over images (paired bootstrap, same seeds): validation ±0.0041, test ±0.0033. Seeds and images together, the test gain's SE is 0.0039.

Per-class val score, start file: pore 0.502, deposit 0.624, discontinuity 0.780, stain 0.609
Per-class val score, final best e009: pore 0.531, deposit 0.625, discontinuity 0.778, stain 0.610

Chain of accepted changes: e000 -> e009

## What changed from the start file to the final best

```diff
--- start/train.py
+++ e009/train.py
@@ -45,6 +45,5 @@
 PRETRAINED = True
 IMG_SIZE = 160
-CONTEXT = 2.5           # wide crop: side = CONTEXT * long side of the box
-TIGHT_CONTEXT = 1.5     # tight crop
+SCALES = (1.25, 1.75, 2.5, 3.5)  # multi-scale crop pyramid
 MIN_CROP = 16
 TRAIN_ON = ("high",)    # resolutions of crops.csv used for training
@@ -52,5 +51,5 @@
 LR = 1e-3
 WEIGHT_DECAY = 0.05
-RESERVE_S = 55          # seconds kept free for prediction and writing
+RESERVE_S = 85          # seconds kept free for prediction and writing (4 crops ≈ 2x pred time)
 MIN_SCORE = 0.001       # rows below this score are not written
 MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
@@ -86,4 +85,29 @@
 
 
+class MultiScaleClassifier(nn.Module):
+    """Shared backbone over 4 scale crops, per-scale 1x1 projection, concat, MLP head."""
+
+    def __init__(self, backbone, num_scales=4, proj_dim=256, hidden=512, num_classes=4):
+        super().__init__()
+        self.backbone = backbone
+        C = backbone.num_features
+        self.scale_projs = nn.ModuleList([nn.Conv2d(C, proj_dim, 1) for _ in range(num_scales)])
+        self.head = nn.Sequential(
+            nn.Linear(num_scales * proj_dim, hidden),
+            nn.GELU(),
+            nn.Linear(hidden, num_classes),
+        )
+
+    def forward(self, views):
+        feats = []
+        for i, x in enumerate(views):
+            f = self.backbone.forward_features(x)   # [B, C, H, W]
+            f = self.scale_projs[i](f)              # [B, proj_dim, H, W]
+            f = f.mean(dim=[2, 3])                  # [B, proj_dim]
+            feats.append(f)
+        f = torch.cat(feats, dim=1)                 # [B, num_scales * proj_dim]
+        return self.head(f)
+
+
 class BoxDataset(torch.utils.data.Dataset):
     def __init__(self, rows, images, train):
@@ -98,8 +122,8 @@
         fh = fv = False
         bright = 1.0
-        if self.train:  # same flips and brightness for both views
+        if self.train:  # same flips and brightness for all views
             fh, fv, bright = random.random() < 0.5, random.random() < 0.5, random.uniform(0.8, 1.2)
         views = []
-        for ctx in (TIGHT_CONTEXT, CONTEXT):
+        for ctx in SCALES:
             x = make_crop(img, r, ctx, self.train)
             if fh:
@@ -111,5 +135,5 @@
             x = (x - MEAN) / STD
             views.append(torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))))
-        return views[0], views[1], r.get("cls", -1)
+        return views, r.get("cls", -1)
 
 
@@ -145,5 +169,6 @@
                                         num_workers=workers, pin_memory=True)
 
-    model = timm.create_model(MODEL, pretrained=PRETRAINED, num_classes=4)
+    backbone = timm.create_model(MODEL, pretrained=PRETRAINED, num_classes=None)
+    model = MultiScaleClassifier(backbone, num_scales=len(SCALES))
     model = model.to(device).to(memory_format=torch.channels_last)
     crit = nn.CrossEntropyLoss()
@@ -159,14 +184,13 @@
     while not done:
         tot, nb = 0.0, 0
-        for xt, xw, y in dl_tr:
+        for views, y in dl_tr:
             frac = min(1.0, (time.time() - t_start) / budget)
             f = frac / 0.05 if frac < 0.05 else 0.5 * (1 + math.cos(math.pi * (frac - 0.05) / 0.95))
             for pg in opt.param_groups:
                 pg["lr"] = LR * max(f, 1e-3)
-            xt = xt.to(device, non_blocking=True).to(memory_format=torch.channels_last)
-            xw = xw.to(device, non_blocking=True).to(memory_format=torch.channels_last)
+            views = [v.to(device, non_blocking=True).to(memory_format=torch.channels_last) for v in views]
             y = y.to(device, non_blocking=True)
             with torch.autocast(**amp):
-                loss = crit((model(xt) + model(xw)) / 2, y)
+                loss = crit(model(views), y)
             opt.zero_grad(set_to_none=True)
             loss.backward()
@@ -183,8 +207,7 @@
     probs = []
     with torch.no_grad(), torch.autocast(**amp):
-        for xt, xw, _ in dl_ev:
-            xt = xt.to(device).to(memory_format=torch.channels_last)
-            xw = xw.to(device).to(memory_format=torch.channels_last)
-            probs.append(torch.softmax(((model(xt) + model(xw)) / 2).float(), 1).cpu())
+        for views, _ in dl_ev:
+            views = [v.to(device).to(memory_format=torch.channels_last) for v in views]
+            probs.append(torch.softmax(model(views).float(), 1).cpu())
     probs = torch.cat(probs).numpy()
     os.makedirs(args.out, exist_ok=True)
```

## All experiments

| id | base | category | hypothesis | seed-0 | 5-seed mean | outcome |
|---|---|---|---|---|---|---|
| e000 | - | - | starting file | 0.6314 | 0.6288 | baseline |
| e001 | e000 | data | The classifier is trained on perfectly-centered ground-truth boxes but must classify stage-1 detector boxes, which are often offset or sized differently — a training/test distribution mismatch. Traini | 0.6276 | - | rejected |
| e002 | e000 | data | The current classifier averages logits from a 1.5x tight crop and a 2.5x wide crop — diluting high-frequency texture that separates pore from deposit/stain, which are the two weak classes. A single ti | 0.6232 | - | rejected |
| e003 | e000 | inference | The dominant error is classification mistakes on boxes overlapping real defects (+0.117 mAP potential), mostly pore-vs-stain and deposit-vs-stain confusion from limited texture. Instead of changing wh | 0.6157 | - | rejected |
| e004 | e000 | scoring | The model systematically biases toward "stain" for pore and deposit boxes (55+20 errors vs only 96% right on true stain). Applying per-class logit biases fitted from the training data directly re-orde | 0.6143 | - | rejected |
| e005 | e000 | architecture | The dominant error (pore/deposit/stain confusion, ~0.117 mAP potential) comes from simple logit averaging of the 1.5x and 2.5x crops, which dilutes high-frequency texture with low-frequency context. R | - | - | crashed |
| e006 | e000 | architecture | The dominant error is class confusion on true defects (pore 63%, deposit 78% right), costing ~0.117 mAP. I hypothesise this is because the single backbone's two crops are merged by simple logit averag | - | - | crashed |
| e007 | e000 | architecture | The current loss is 0.12 mAP from wrong class labels on true defects (deposit→stain 55, pore→stain 20). Simple logit averaging of two crops under-uses per-crop information; cross-attention lets the ti | 0.6337 | 0.6323 | not confirmed |
| e008 | e007 | architecture | e007 hypothesized cross-attention between crops but implemented only MLP fusion, so it never actually got tested. Cross-attention lets the tight crop's texture features attend to wide-crop context (an | 0.6303 | 0.6320 | not confirmed |
| e009 | e000 | architecture | Replace the two fixed 1.5x/2.5x crops and logit averaging with a 4-scale crop pyramid (scales 1.25x, 1.75x, 2.5x, 3.5x) fused by a learnable feature-pyramid head. This targets the pore/deposit/stain c | 0.6403 | 0.6359 | new best |
| e010 | e009 | architecture | The dominant remaining error is class confusion (pore→stain, deposit→stain) on boxes that overlap real defects, costing ~0.10 mAP. Global average pooling in e009's fusion head discards spatial structu | 0.6282 | - | rejected |
| e011 | e010 | architecture | The dominant error is pore (63%) and deposit (77%) misclassified as stain, i.e. a texture-level discrimination problem that 160px crops + efficientnet_b0 cannot solve. Give the classifier a dedicated  | 0.5990 | - | rejected |
| e012 | e009 | architecture | The dominant remaining loss (~0.10 mAP) is class confusion on true defects, especially pore→stain (35% of pores) and deposit→stain (20% of deposits). I hypothesise this happens because the stage-1 det | 0.6142 | - | rejected |
| e013 | e009 | data | The classifier is trained only on ground-truth boxes but must classify stage-1 detector boxes at test time — a distribution mismatch (offset, size) that hits the weak classes hardest. Training on the  | 0.6168 | - | rejected |
| e014 | e009 | inference | The current single model at 0.6359 leaves ~0.10 mAP on the table purely from class confusion; averaging predictions of several independently trained models is one of the most reliable ways to improve  | 0.5771 | - | rejected |
| e015 | e009 | architecture | Pore→stain and deposit→stain confusion costs ~0.10 mAP; both classes are misread as stain, while true stain is already 90% right. A box's area relative to the image (box-area ratio) is a strong geomet | - | - | crashed |
| e016 | e009 | scoring | The dominant loss (≈0.10 mAP) is class confusion on true defects, especially pore→stain and deposit→stain. A per-class geometric prior — box area relative to image — is a strong, independent cue: pore | 0.5765 | - | rejected |
| e017 | e009 | architecture | The remaining ~0.10 mAP loss is class confusion on true defects (pore 63%, deposit 77% right) that needs fine texture at higher resolution; efficientnet_b0 at 160px cannot resolve pore vs deposit vs s | 0.6194 | - | rejected |
| e018 | e009 | architecture | The dominant loss is class confusion on true defects (pore 63% right, deposit 77% right, both often read as "stain"). I hypothesise this is a data-quality problem, not a capacity problem: the model le | - | - | timeout |
| e019 | e009 | architecture | The dominant error (pore 0.531, deposit 0.625 AP; both often read as "stain") is a resolution/texture problem: EfficientNet-B0 crops at 160 px cannot encode the fine texture that separates pore from d | - | - | timeout |
| e020 | e009 | architecture | The dominant loss (~0.10 mAP) is pore→stain and deposit→stain confusion on true defects. A "large" box covering most of a real stain can be classified from coarse shape, but pore/deposit misread as st | 0.6263 | - | rejected |
| e021 | e009 | optimisation | The current model trains ~157 steps (2 epochs) of a 4-scale pyramid EfficientNet-B0 that is likely undertrained for the hard pore/deposit/stain discrimination; averaging weights along one training tra | - | - | crashed |
| e022 | e021 | data | The remaining ~0.10 mAP loss is class confusion on true defects (pore→stain, deposit→stain), and the systematic model/label disagreement on hard classes suggests label noise (e.g. "stain" boxes that a | 0.5693 | - | rejected |
| e023 | e022 | architecture | The core error is a training/test distribution mismatch: stage 2 learns on perfectly-centered GT boxes but classifies offset/shifted detector boxes, which costs ~0.10 mAP almost entirely through pore→ | - | - | crashed |
| e024 | e023 | optimisation | e023 was rejected only because it crashed on a one-line Python bug (iterating a DataLoader with next() instead of an iterator), so its idea — hybrid training on real detector boxes, label cleaning, an | 0.6110 | - | rejected |
| e025 | e009 | architecture | The current 0.6359 loses ~0.10 mAP almost entirely to pore→stain and deposit→stain confusion — the model sees a crop, not the defect. The hypothesis is that stage-1 detector boxes are frequently offse | - | - | crashed |
| e026 | e009 | architecture | The dominant loss (~0.10 mAP) is pore→stain and deposit→stain confusion on true defects — a texture-resolution limit of EfficientNet-B0 at 160px, not model capacity in general. Give the classifier a s | - | - | crashed |

Each experiment's folder (/data/ar2/runs/p1/eNNN/) holds train.py, diff.patch, the prompt, the LLM's reply and thinking, and the run logs.

## Provenance

| item | value |
|---|---|
| python | 3.12.3 |
| start_file | cb33b09ac6afffea0b9ba15234a362367d84fd3cc4ed8bedd4d9832b7e39c334 |
| brief | 023373b012478d21c104fdfb0eba879b16f844d85bd2dd27e1cb51af24694a29 |
| digest | fcd9348fd7f0eeb239881bfb36b5e2be8db268350f16dc525a50e441f1828906 |
| ar_loop.py | 15477f3667b49b450792c6dc73076e2fec1015292cc59755e4783efaff772938 |
| det_metric.py | b055a87c2167735066dbb01f68e4027a09f768f4145f1bd58db2c33ac2685ce1 |
| data/dets.csv | a33042aa47ad358ab6e5700611ccc55a139cf20a5a5589ef8e2524e0abb6d43e |
| data/crops.csv | 7fa6269ca75cb4410aea5057138dc0ca73b5744ff04060893155025b338fb953 |
| hidden labels | 1fd0b668e18f41df8fdbee4d7e0502a806cd255e266637bf8455cbcd533d5f99 |
| model_file | /data/models/gguf/byteshape/Qwen3.8-27B-IQ4_XS-3.84bpw.gguf |
| model_file_sha | fingerprint:5940e8d36a690b185139d0420739651ae44db7676dbb1c12aff458f30a1ff56a |
| llama_cpp_commit | 0ec4027a2a0183de87c007069adddc06e665f7de |
| packages_sha | 5a08c360913eeb3cd7195654571716f10f71401aa4df2ba9515f762380b0a662 |
| key_packages | numpy 2.5.4, pillow 13.0.0, timm 1.0.30, torch 2.14.1, torchvision 0.29.1 |
| train.py of start / final best | - / 2f5a64c512f698c706189a16e4b3777c1351ca73bc7afd6d3d3512379167e324 |
| start-file runs reused from | b5 |
