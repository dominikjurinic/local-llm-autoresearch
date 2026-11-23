# AutoResearch report: h1

- Task: det (mAP@0.5); LLM: **qwen3.8-27b-D_gpu5_mtp3** (thinking: xhigh), training budget 180.0 s wall time per run, 5 seeds per confirmed result
- Started 2026-11-14 01:42:35, report written 2026-11-14 06:32:48
- Harness: brief ar2_program_v3.md; acceptance gain > 1.0 SE; second seed for close calls; lessons table in prompt; diagnostics in prompt; background digest ar2_digest.md; start-file runs reused from b5; uncertainty includes validation images (bootstrap); memory of earlier runs ar_memory.md; HELD-OUT CONFIRMATION: 82 of 164 validation images never shown to the agent, a change must also gain > 0.5 SE on them (split seed 1)
- Ideas tried: 19 | edits unusable: 0 | crashed/timeout/bad output: 7 | likely bugs (score far below best): 1 | ran but worse: 2 | not confirmed: 9 | **confirmed improvements: 0**
- LLM time: 2.4 h in 19 calls (459 s per call); training time: 2.7 h

| category | ideas | ran | confirmed |
|---|---|---|---|
| data | 11 | 9 | 0 |
| architecture | 7 | 2 | 0 |
| optimisation | 1 | 1 | 0 |

## Final evaluation (hidden test set, read only here)

| version | seeds | val mAP@0.5 | test mAP@0.5 | test gain vs start |
|---|---|---|---|---|
| start file | 8 | 0.6320 ± 0.0020 | 0.6869 ± 0.0015 | - |

Values are mean ± standard error over seeds. A gain above about 2 SE is unlikely to be seed noise.

## All experiments

| id | base | category | hypothesis | seed-0 | 5-seed mean | outcome |
|---|---|---|---|---|---|---|
| e000 | - | - | starting file | 0.6268 | 0.6318 | baseline |
| e001 | e000 | data | The dominant error is wrong classes (0.7530 if classes were right vs 0.6318 now), concentrated in pore (60% right) and the stain/deposit confusion, and the model is trained on perfectly-placed GT boxe | 0.6326 | 0.6354 | not confirmed |
| e002 | e001 | architecture | The dominant error is wrong class (0.7530 if classes were right vs 0.6318 now), concentrated in pore (AP 0.478, 60% right) and the stain/deposit pair. The classifier currently sees only pixels, so it  | - | - | crashed |
| e003 | e002 | architecture | e002 crashed only because `box_geom`/`G_MEAN`/`G_STD` were defined inside `main()` but referenced from `BoxDataset.__getitem__`, which runs in DataLoader worker processes where those names are not glo | - | - | crashed |
| e004 | e000 | data | The dominant error is the class decision (0.7530 if classes were right vs 0.6318 now), concentrated in pore (39% wrong, mostly ->stain) and the stain/deposit pair, i.e. fine-texture confusions. I repl | 0.6244 | - | rejected |
| e005 | e000 | architecture | Replace the two separate forward passes with averaged logits by a single 6-channel input (the 1.5x tight and 2.5x wide crops concatenated), so one network must fuse cross-scale evidence in a single re | 0.6251 | - | rejected |
| e006 | e000 | data | The dominant loss is the class decision (0.7530 with correct classes vs 0.6318), concentrated in pore (AP 0.478, only 60% right, mostly misread as stain). Earlier runs found that a third, wider 4x con | - | - | crashed |
| e007 | e006 | data | Fix the e006 crash (4x wide context view) by correcting the dataset's return statement (missing views[2]) and the training loop's for-unpacking (missing x2). The 4x view adds broad context that helps  | 0.6490 | 0.6422 | not confirmed |
| e008 | e000 | architecture | The dominant error is the class decision (0.7530 with perfect classes vs 0.6318), concentrated in pore (60% right, mostly read as stain) and small boxes (77% right vs 94% for large). The fixed 160px c | - | - | crashed |
| e009 | e008 | architecture | e008's box-geometry input (log area + log aspect as two extra zero-init channels) crashed only on a 4-value tuple unpacked into 2 names; the idea itself was never actually measured. Small boxes are th | - | - | crashed |
| e010 | e000 | data | The dominant error is the class decision (0.7530 with correct classes vs 0.6318), concentrated in pore (60% right, mostly misread as stain) and the stain/deposit pair. I combine the two best "close bu | 0.6360 | 0.6359 | not confirmed |
| e011 | e000 | data | The dominant error is the class decision (0.7530 with correct classes vs 0.6318), concentrated in pore (AP 0.478, 60% right, 12 of 16 errors read as stain) and the stain/deposit pair; small boxes are  | - | - | crashed |
| e012 | e000 | data | On top of the current best I add (a) a third 4x-wide context view (the e007 change, the largest single gain found, +0.0104, which targets pore's lack of context) and (b) detector boxes from the train  | 0.6500 | 0.6434 | not confirmed |
| e013 | e000 | architecture | The dominant loss is the class decision (0.7530 with perfect classes vs 0.6318), concentrated in the stain/deposit pair and pore, and small boxes are the weak spot (77% right vs 94% large). Because ev | - | - | crashed |
| e014 | e000 | data | The current model trains only on high-resolution crops (TRAIN_ON = ("high",)), but the LoHi-WELD dataset contains both low- and high-resolution weld images and the eval split likely includes both. Por | 0.6368 | 0.6323 | not confirmed |
| e015 | e000 | optimisation | The current run fine-tunes the whole EfficientNet-B0 with one uniform LR (1e-3, AdamW) for only ~433 steps; the randomly-initialised 4-way head and the pretrained backbone have very different adaptati | 0.6348 | 0.6355 | not confirmed |
| e016 | e000 | data | The class decision costs 0.1212 (0.6318 now vs 0.7530 with perfect classes) and the classifier is trained on perfectly-placed GT boxes but tested on the detector's slightly-misplaced boxes, a train/te | 0.6436 | 0.6376 | not confirmed |
| e017 | e000 | architecture | The dominant error is the class decision (0.7530 with perfect classes vs 0.6318 now), concentrated in the stain/deposit pair (31 stain->deposit + 27 deposit->stain) and pore->stain (12), i.e. fine tex | 0.3912 | - | suspicious (likely bug) |
| e018 | e000 | data | The largest broad error is the stain↔deposit boundary (31 stain→deposit + 27 deposit→stain = 58 of the 121 wrong-class boxes), and stain/deposit are the two large classes, so a fix here is broad enoug | 0.6357 | 0.6374 | not confirmed |
| e019 | e018 | data | The stain↔deposit boundary is the largest broad error (31 stain→deposit + 27 deposit→stain = 58 of 94 wrong-class boxes, between the two biggest classes), and e018's denser tight view (TIGHT_DENS 2.0) | 0.6497 | 0.6529 | not confirmed |

Each experiment's folder (/data/ar2/runs/h1/eNNN/) holds train.py, diff.patch, the prompt, the LLM's reply and thinking, and the run logs.

## Provenance

| item | value |
|---|---|
| python | 3.12.3 |
| start_file | cb33b09ac6afffea0b9ba15234a362367d84fd3cc4ed8bedd4d9832b7e39c334 |
| brief | 023373b012478d21c104fdfb0eba879b16f844d85bd2dd27e1cb51af24694a29 |
| digest | fcd9348fd7f0eeb239881bfb36b5e2be8db268350f16dc525a50e441f1828906 |
| ar_loop.py | 539dd4dbfb69ee28ba7b05bc6300718b0afc93b5d451c2778ab60d9c08bf8a84 |
| det_metric.py | b055a87c2167735066dbb01f68e4027a09f768f4145f1bd58db2c33ac2685ce1 |
| data/dets.csv | a33042aa47ad358ab6e5700611ccc55a139cf20a5a5589ef8e2524e0abb6d43e |
| data/crops.csv | 7fa6269ca75cb4410aea5057138dc0ca73b5744ff04060893155025b338fb953 |
| hidden labels | 1fd0b668e18f41df8fdbee4d7e0502a806cd255e266637bf8455cbcd533d5f99 |
| model_file | /data/models/gguf/byteshape/Qwen3.8-27B-IQ4_XS-3.84bpw.gguf |
| model_file_sha | fingerprint:5940e8d36a690b185139d0420739651ae44db7676dbb1c12aff458f30a1ff56a |
| llama_cpp_commit | 0ec4027a2a0183de87c007069adddc06e665f7de |
| packages_sha | 5a08c360913eeb3cd7195654571716f10f71401aa4df2ba9515f762380b0a662 |
| key_packages | numpy 2.5.4, pillow 13.0.0, timm 1.0.30, torch 2.14.1, torchvision 0.29.1 |
| train.py of start / final best | - / - |
| start-file runs reused from | b5 |
