# AutoResearch report: s1

- Task: det (mAP@0.5); LLM: **sage-a1.5-q4km** (thinking: True), training budget 180.0 s wall time per run, 5 seeds per confirmed result
- Started 2026-11-04 17:55:12, report written 2026-11-04 23:09:15
- Harness: brief ar2_program_v3.md; acceptance gain > 1.0 SE; second seed for close calls; lessons table in prompt; diagnostics in prompt; background digest ar2_digest.md; start-file runs reused from b5; uncertainty includes validation images (bootstrap); memory of earlier runs ar_memory.md
- Ideas tried: 28 | edits unusable: 4 | crashed/timeout/bad output: 12 | likely bugs (score far below best): 2 | ran but worse: 4 | not confirmed: 5 | **confirmed improvements: 0**
- LLM time: 3.3 h in 45 calls (261 s per call); training time: 2.3 h

| category | ideas | ran | confirmed |
|---|---|---|---|
| architecture | 17 | 4 | 0 |
| data | 5 | 5 | 0 |
| scoring | 3 | 2 | 0 |
| optimisation | 2 | 1 | 0 |
| - | 1 | 0 | 0 |

## Final evaluation (hidden test set, read only here)

| version | seeds | val mAP@0.5 | test mAP@0.5 | test gain vs start |
|---|---|---|---|---|
| start file | 8 | 0.6274 ± 0.0018 | 0.6869 ± 0.0015 | - |

Values are mean ± standard error over seeds. A gain above about 2 SE is unlikely to be seed noise.

## All experiments

| id | base | category | hypothesis | seed-0 | 5-seed mean | outcome |
|---|---|---|---|---|---|---|
| e000 | - | - | starting file | 0.6314 | 0.6288 | baseline |
| e001 | e000 | scoring | The product rule conf × p(c) re-ranks predictions within each class worse than using the detector's own confidence, losing ~0.012 mAP even with perfect classes; combined with the 0.1173 loss from wron | - | - | crashed |
| e002 | e000 | architecture | The dominant error is wrong classes (0.117 mAP), mostly pore–deposit and deposit–stain confusions that need finer texture than two fixed 160px crops provide. Switching IMG_SIZE from 160 to 224 uses th | - | - | timeout |
| e003 | e000 | architecture | The dominant loss is wrong-class assignment on boxes overlapping true defects (0.117 mAP), driven by pore–deposit confusions that need finer texture than a 160px crop can provide. Raising IMG_SIZE to  | - | - | edit failed |
| e004 | e000 | scoring | The product rule `conf × p(c)` re-ranks boxes within each class by classifier confidence, demoting true positives when the classifier mislabels them (low `p(c)` for their true class). Reference number | 0.6067 | - | rejected |
| e005 | e000 | optimisation | The dominant loss (0.117 mAP) is wrong-class assignment on detector boxes, driven by poorly calibrated classifier probabilities. Stochastic weight averaging — keeping a snapshot of the model every epo | 0.6329 | 0.6290 | not confirmed |
| e006 | e000 | architecture | The dominant error is wrong-class assignment on true defects (0.117 mAP), mostly pore–deposit/stain confusions needing better features than EfficientNet-B0 provides. Switching to `efficientnetv2_b0` — | - | - | crashed |
| e007 | e000 | data | The main error is wrong classes on overlapping boxes (0.117 mAP), driven by pore–stain and deposit–stain confusions that need wider context than a 2.5× crop provides — a third 4× view helped +0.005 mo | 0.6291 | 0.6290 | not confirmed |
| e008 | e000 | architecture | The dominant loss is wrong class assignment (0.117 mAP), especially pore–stain and deposit–stain confusion driven by texture features. Switching from EfficientNet-B0 to convnext_nano/pico changes the  | 0.2143 | - | suspicious (likely bug) |
| e009 | e000 | data | Train the stage-2 classifier on the *detector boxes themselves* (stage-1 output on train images), labelled by IoU > 0.5 overlap with ground truth — not on ground-truth crops.csv boxes. This eliminates | 0.6351 | 0.6305 | not confirmed |
| e010 | e000 | data | The 0.117 mAP classification loss is driven by a classifier trained only on ground-truth boxes yet scoring noisy detector boxes; e009's gain (+0.0017) came from matching positive boxes but left false  | 0.4972 | - | suspicious (likely bug) |
| e011 | e010 | data | e010's idea — train stage-2 on the detector boxes themselves, with false alarms labeled background — is sound but died to a fatal label collision: background was coded as class 0, which is also "pore" | 0.5663 | - | rejected |
| e012 | e000 | architecture | The dominant error is wrong-class assignment on true defects (0.117 mAP), driven by pore–deposit and deposit–stain confusion. This family tried bigger models and resolution, but not matching the backb | 0.6269 | - | rejected |
| e013 | e000 | architecture | The dominant loss is wrong class assignment (0.117 mAP), concentrated in pore–stain and deposit–stain texture confusions. Previous architecture attempts changed backbones or used 224px with a halved b | 0.6202 | - | rejected |
| e014 | e000 | architecture | The dominant loss (0.117 mAP from wrong classes, mostly pore–deposit–stain texture confusions) comes from the classifier's naive logit-averaging of the two crops, which weights fine texture and contex | - | - | crashed |
| e015 | e014 | architecture | Replace naive logit-averaging of the tight and wide crops with a learned per-class MLP over the concatenated [tight_feat, wide_feat] mean-pooled features, so the classifier learns to weight fine textu | - | - | edit failed |
| e016 | e000 | architecture | Different defect classes rely on different mixtures of fine texture (tight crop) and context (wide crop). The current model averages logits per class, forcing identical weighting for all four classes; | - | - | crashed |
| e017 | e000 | data | The ~0.117 mAP loss from wrong classes is concentrated in pore–deposit and deposit–stain confusions that need finer texture than a 160px tight crop provides; pores are small defects so they lose their | 0.6455 | - | failed on confirmation |
| e018 | e000 | architecture | Different defect classes need different mixtures of fine texture (tight crop) and context (wide crop); the current equal logit averaging forces all four classes to use the same trade-off, which hurts  | - | - | crashed |
| e019 | e000 | architecture | Repair the never-successfully-run learned combination idea from e014-e018 (crashed on an invalid timm kwarg, not a bad idea). Build the features of both crops with a 2×2 pooling + concat head, then cl | - | - | crashed |
| e020 | e000 | architecture | Fix the crash in e019's idea — replace naive equal logit-averaging of the two crops with a learned per-class combination of [tight_feat, wide_feat]. Different defects weight fine texture (pore) agains | - | - | crashed |
| e021 | e000 | architecture | Replace the softmax classifier with a nearest-prototype matcher: extract per-box features from both crops, average them, and score each class by cosine similarity to a running prototype (mean feature) | - | - | crashed |
| e022 | e000 | - |  | - | - | edit failed |
| e023 | e000 | architecture | Replace naive equal logit-averaging of tight/wide crops with a learned per-class combination: concat mean-pooled features [tight, wide] and classify via small MLP. Different defects weigh texture vs c | 0.6308 | 0.6293 | not confirmed |
| e024 | e023 | scoring | The classifier's soft probability is too strong in demoting true positives from boxes it mislabels. The remaining loss (0.7461 with perfect classes vs 0.758 ceiling) is a ranking cost, not class assig | 0.6351 | 0.6248 | not confirmed |
| e025 | e000 | architecture | The dominant 0.117 mAP loss is wrong class assignment (pore↔stain, deposit↔stain), and it comes from a model that sees every defect at only one scale. Pores are tiny so a 160px tight crop often shows  | - | - | timeout |
| e026 | e000 | architecture | Replace naive equal logit-averaging of the tight and wide crops with a learned per-class combination of [tight_feat, wide_feat]. The current model forces all four classes to use the same texture-vs-co | - | - | crashed |
| e027 | e000 | optimisation | Average the class probabilities of 4 independently trained models (different seeds) instead of using one model's output. The current single-model score loses ~0.117 mAP from wrong-class assignment, co | - | - | bad output |
| e028 | e000 | architecture | Append a learned embedding of the detector box's physical size (log area, aspect ratio) to the mean-pooled features before classification, so the classifier can use "box is small → likely pore" as evi | - | - | edit failed |

Each experiment's folder (/data/ar2/runs/s1/eNNN/) holds train.py, diff.patch, the prompt, the LLM's reply and thinking, and the run logs.

## Provenance

| item | value |
|---|---|
| python | 3.12.3 |
| start_file | cb33b09ac6afffea0b9ba15234a362367d84fd3cc4ed8bedd4d9832b7e39c334 |
| brief | 023373b012478d21c104fdfb0eba879b16f844d85bd2dd27e1cb51af24694a29 |
| digest | fcd9348fd7f0eeb239881bfb36b5e2be8db268350f16dc525a50e441f1828906 |
| ar_loop.py | 2ef3846d4d983c671d8c9f5fb7d4c0ae525fdb3ab79b37f74569184b1c7c510d |
| det_metric.py | b055a87c2167735066dbb01f68e4027a09f768f4145f1bd58db2c33ac2685ce1 |
| data/dets.csv | a33042aa47ad358ab6e5700611ccc55a139cf20a5a5589ef8e2524e0abb6d43e |
| data/crops.csv | 7fa6269ca75cb4410aea5057138dc0ca73b5744ff04060893155025b338fb953 |
| hidden labels | 1fd0b668e18f41df8fdbee4d7e0502a806cd255e266637bf8455cbcd533d5f99 |
| model_file | /data/models/gguf/sage/Sage-A1.5-Q4_K_M.gguf |
| model_file_sha | fingerprint:b74a28ee4f68cd76d23ed3a50ad11730ae4a2c22c7c0b411e358f61be13b5420 |
| llama_cpp_commit | 0ec4027a2a0183de87c007069adddc06e665f7de |
| packages_sha | 5a08c360913eeb3cd7195654571716f10f71401aa4df2ba9515f762380b0a662 |
| key_packages | numpy 2.5.4, pillow 13.0.0, timm 1.0.30, torch 2.14.1, torchvision 0.29.1 |
| train.py of start / final best | - / - |
| start-file runs reused from | b5 |
