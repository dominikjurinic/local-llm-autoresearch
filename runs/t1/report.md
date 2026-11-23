# AutoResearch report: t1

- Task: det (mAP@0.5); LLM: **qwen3.8-27b-D_gpu5_mtp3** (thinking: xhigh), training budget 180.0 s wall time per run, 5 seeds per confirmed result
- Started 2026-11-06 18:18:07, report written 2026-11-13 10:37:42
- Harness: brief ar2_program_v3.md; acceptance gain > 1.0 SE; second seed for close calls; lessons table in prompt; diagnostics in prompt; background digest ar2_digest.md; start-file runs reused from b5; uncertainty includes validation images (bootstrap); memory of earlier runs ar_memory.md; PAIR MODE: sage-a1.5-q4km plans each experiment, qwen3.8-27b-D_gpu5_mtp3 implements it; TRIO: gpt-oss-20b reviews each plan first
- Ideas tried: 12 | edits unusable: 0 | crashed/timeout/bad output: 3 | likely bugs (score far below best): 1 | ran but worse: 3 | not confirmed: 4 | **confirmed improvements: 1**
- LLM time: 3.8 h in 56 calls (245 s per call); training time: 2.0 h

| category | ideas | ran | confirmed |
|---|---|---|---|
| architecture | 10 | 8 | 1 |
| inference | 1 | 0 | 0 |
| data | 1 | 1 | 0 |

## Final evaluation (hidden test set, read only here)

| version | seeds | val mAP@0.5 | test mAP@0.5 | test gain vs start |
|---|---|---|---|---|
| start file | 8 | 0.6274 ± 0.0018 | 0.6869 ± 0.0015 | - |
| final best (e008) | 8 | 0.6300 ± 0.0015 | 0.6882 ± 0.0016 | +0.0012 ± 0.0022 (+0.6 SE) |

Values are mean ± standard error over seeds. A gain above about 2 SE is unlikely to be seed noise.

Uncertainty of the final gain over images (paired bootstrap, same seeds): validation ±0.0019, test ±0.0017. Seeds and images together, the test gain's SE is 0.0028.

Per-class val score, start file: pore 0.502, deposit 0.624, discontinuity 0.780, stain 0.609
Per-class val score, final best e008: pore 0.505, deposit 0.634, discontinuity 0.781, stain 0.611

Chain of accepted changes: e000 -> e008

## What changed from the start file to the final best

```diff
--- start/train.py
+++ e008/train.py
@@ -114,4 +114,16 @@
 
 
+class FusionModel(nn.Module):
+    """EfficientNet-B0 backbone with a learnable per-class tight/wide logit fusion."""
+    def __init__(self, base):
+        super().__init__()
+        self.base = base
+        self.logit_w = nn.Parameter(torch.zeros(4))  # sigmoid -> 0.5 (equal weight)
+
+    def forward(self, xt, xw):
+        w = torch.sigmoid(self.logit_w)
+        return w * self.base(xt) + (1 - w) * self.base(xw)
+
+
 def main():
     ap = argparse.ArgumentParser()
@@ -145,6 +157,7 @@
                                         num_workers=workers, pin_memory=True)
 
-    model = timm.create_model(MODEL, pretrained=PRETRAINED, num_classes=4)
-    model = model.to(device).to(memory_format=torch.channels_last)
+    base = timm.create_model(MODEL, pretrained=PRETRAINED, num_classes=4)
+    base = base.to(device).to(memory_format=torch.channels_last)
+    model = FusionModel(base).to(device)
     crit = nn.CrossEntropyLoss()
     opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
@@ -168,5 +181,5 @@
             y = y.to(device, non_blocking=True)
             with torch.autocast(**amp):
-                loss = crit((model(xt) + model(xw)) / 2, y)
+                loss = crit(model(xt, xw), y)
             opt.zero_grad(set_to_none=True)
             loss.backward()
@@ -186,5 +199,5 @@
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
| e001 | e000 | architecture | The biggest loss is wrong class assignment (fixing it alone raises mAP from 0.629 to ~0.746). Earlier backbone experiments only tried *bigger* models (EfficientNet-B2, ResNet-50), which lose because f | 0.4438 | - | suspicious (likely bug) |
| e002 | e000 | architecture | The current two-crop model is limited by resolution on the tight crop, which hurts fine texture discrimination for pore and deposit/stain. Raising the tight crop to 224px while keeping the wide crop a | 0.6345 | 0.6228 | not confirmed |
| e003 | e002 | inference | The loss is not in the classifier's accuracy on the overlap set but in how detector confidence (from YOLO) and the classifier's probability are combined for ranking. Multiplying by conf pushes false a | - | - | crashed |
| e004 | e000 | architecture | The dominant error is wrong class assignment on detector boxes that overlap true defects (perfect-class mAP is 0.7461 vs our 0.6288). Averaging two independently-seeded EfficientNet-B0 models reduces  | 0.5704 | - | rejected |
| e005 | e000 | architecture | The dominant error is wrong class assignment (0.6288 -> 0.7461 if every overlapping box had its correct class). I attack this by training the classifier to match the test distribution exactly: each tr | 0.6372 | 0.6269 | not confirmed |
| e006 | e000 | architecture | The classifier's accuracy on detector boxes (especially pore 63% and deposit 78%) is the main bottleneck (wrong-class loss = 0.117 mAP). Pore is small and texture-rich, so a stronger backbone at a fix | 0.6263 | - | rejected |
| e007 | e000 | architecture | The dominant error is wrong class assignment, with small objects (80% correct vs 91-93% for medium/large) and pore/deposit (63%/78%) far below discontinuity/stain. A single fixed input resolution of 1 | 0.6380 | 0.6299 | not confirmed |
| e008 | e000 | architecture | The two crops are currently fused with a fixed 50/50 logit average, but different defects need different context: pore is small and texture-only (tight crop), while deposit vs stain needs the wider co | 0.6298 | 0.6327 | new best |
| e009 | e008 | data | The dominant error is wrong class assignment on detector boxes (wrong-class loss ≈ +0.11 mAP). The current model trains only on crops.csv ground-truth boxes, but at test time it classifies dets.csv de | 0.6385 | 0.6336 | not confirmed |
| e010 | e009 | architecture | The dominant error is wrong-class assignment on overlapping boxes (+0.11 mAP potential; pore 62%, deposit 77% vs discontinuity 96%). e008's per-class tight/wide fusion already proved that context weig | - | - | crashed |
| e011 | e008 | architecture | Per-box adaptive fusion of tight/wide logits, driven by box geometry (log w, log h, aspect) via a small MLP, targets the dominant error (wrong-class assignment on overlapping boxes, ~0.11 mAP) where s | 0.6317 | - | rejected |
| e012 | e008 | architecture | The dominant error (wrong class assignment, ~0.11 mAP) is concentrated on small defects (80% right vs 91-92% for medium/large), where the tight crop at 160px leaves too little pixel detail for texture | - | - | timeout |

Each experiment's folder (/data/ar2/runs/t1/eNNN/) holds train.py, diff.patch, the prompt, the LLM's reply and thinking, and the run logs.

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
| train.py of start / final best | - / 48426afbc11f9fe5a9b67d49c00b60a270bfa492bbcc89132589c3e69c5834f1 |
| start-file runs reused from | b5 |
