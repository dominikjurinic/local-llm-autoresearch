# AutoResearch report: n1

- Task: det (mAP@0.5); LLM: **nvidia/nemotron-3-ultra-550b-a55b** (thinking: True), training budget 180.0 s wall time per run, 5 seeds per confirmed result
- Started 2026-11-14 23:47:36, report written 2026-11-15 03:21:04
- Harness: brief ar2_program_v3.md; acceptance gain > 1.0 SE; second seed for close calls; lessons table in prompt; diagnostics in prompt; background digest ar2_digest.md; start-file runs reused from b5; uncertainty includes validation images (bootstrap); memory of earlier runs ar_memory.md; HELD-OUT CONFIRMATION: 82 of 164 validation images never shown to the agent, a change must also gain > 0.5 SE on them (split seed 1)
- Ideas tried: 19 | edits unusable: 0 | crashed/timeout/bad output: 5 | likely bugs (score far below best): 2 | ran but worse: 1 | not confirmed: 11 | **confirmed improvements: 0**
- LLM time: 0.7 h in 19 calls (127 s per call); training time: 3.2 h

| category | ideas | ran | confirmed |
|---|---|---|---|
| architecture | 11 | 8 | 0 |
| data | 3 | 3 | 0 |
| optimisation | 3 | 3 | 0 |
| inference | 2 | 0 | 0 |

## Final evaluation (hidden test set, read only here)

| version | seeds | val mAP@0.5 | test mAP@0.5 | test gain vs start |
|---|---|---|---|---|
| start file | 8 | 0.6320 ± 0.0020 | 0.6869 ± 0.0015 | - |

Values are mean ± standard error over seeds. A gain above about 2 SE is unlikely to be seed noise.

## All experiments

| id | base | category | hypothesis | seed-0 | 5-seed mean | outcome |
|---|---|---|---|---|---|---|
| e000 | - | - | starting file | 0.6268 | 0.6318 | baseline |
| e001 | e000 | architecture | The classifier receives crops resized to 160×160, losing absolute defect size and aspect ratio—cues that help separate pore (small, round) from stain (larger, irregular) and deposit (elongated). I wil | 0.6400 | 0.6363 | not confirmed |
| e002 | e000 | data | The stage-2 classifier trains on perfect crops from crops.csv but at test time sees detector boxes with localization errors and different crop statistics. Earlier runs showed that adding detector boxe | 0.6332 | 0.6389 | not confirmed |
| e003 | e000 | architecture | The pore class (AP 0.478) is the main weakness, often confused with stain and deposit. Earlier runs showed that adding a third, very wide view (4× box context) alongside the 1.5× and 2.5× crops, with  | 0.6447 | 0.6455 | not confirmed |
| e004 | e000 | architecture | The classifier sees crops resized to 160×160, losing absolute defect size and aspect ratio—key cues to separate pore (small, round) from stain (larger, irregular) and deposit (elongated). Earlier runs | 0.6398 | 0.6355 | not confirmed |
| e005 | e000 | architecture | The pore class (AP 0.478) is the main weakness, confused with stain/deposit due to insufficient context at small scales. Earlier runs accepted a fixed third wide view (4×) twice (+0.005 mAP, pore-driv | - | - | crashed |
| e006 | e005 | architecture | The third extra-wide view (3.5×) helps the pore class (AP 0.478) by providing larger context, as earlier runs confirmed twice with a fixed 4× view (+0.005 mAP, pore-driven). The 2-of-3 random sampling | - | - | crashed |
| e007 | e000 | architecture | The classifier currently sees only crops resized to 160×160, losing absolute defect size and aspect ratio—key cues to separate pore (small, round) from stain (larger, irregular) and deposit (elongated | - | - | crashed |
| e008 | e007 | architecture | The e007 experiment crashed due to a simple bug: `read_csv` converts only x1,y1,x2,y2,conf to float, but `compute_box_feats` needs `img_w` and `img_h` as floats too. The box metadata idea (log relativ | 0.6392 | 0.6343 | not confirmed |
| e009 | e000 | data | The stage-2 classifier trains on perfect crops from crops.csv but at test time sees detector boxes with localization errors and different crop statistics. Earlier runs (e002 and two 5-hour runs) showe | 0.6388 | 0.6403 | not confirmed |
| e010 | e000 | architecture | The pore class (AP 0.478) is confused with stain/deposit due to insufficient context at small scales. Earlier runs confirmed a fixed 4× wide view twice (+0.005 mAP, pore-driven) but it increased train | 0.6435 | 0.6493 | not confirmed |
| e011 | e000 | data | The classifier trains only on perfect ground-truth crops (4 defect classes) but at test time sees detector boxes that include false alarms and localization errors. It has no "background" concept, so i | 0.6285 | 0.6319 | not confirmed |
| e012 | e000 | architecture | The pore class (AP 0.478) is the main weakness, confused with stain/deposit due to insufficient context at small scales. Two previously confirmed ideas address complementary problems: (1) training on  | 0.6441 | 0.6454 | not confirmed |
| e013 | e012 | inference | The extra-wide view (3.5×) helps small defects (pores) by providing context but adds noise for larger defects (stains, deposits). Using it only for boxes below the median training box area at both tra | - | - | crashed |
| e014 | e000 | optimisation | Adding an exponential moving average (EMA) of model weights during training (a standard optimisation technique) often improves generalization and calibration without extra compute. The current trainin | 0.4027 | - | suspicious (likely bug) |
| e015 | e000 | optimisation | EMA is a standard optimisation technique that averages weights along the training trajectory, improving generalization and calibration. The e014 attempt failed due to a performance bug (CPU-GPU transf | 0.4410 | - | suspicious (likely bug) |
| e016 | e000 | architecture | The pore class (AP 0.478) is confused with stain and deposit largely because resizing crops to 160×160 destroys absolute size and aspect ratio cues—pores are small/round, stains larger/irregular, depo | 0.6469 | 0.6377 | not confirmed |
| e017 | e000 | architecture | The stage-2 classifier suffers from two complementary problems: (1) it trains on perfect crops but tests on noisy detector boxes (domain shift), and (2) resizing to 160×160 destroys absolute size/shap | 0.6378 | 0.6416 | not confirmed |
| e018 | e000 | optimisation | Stochastic Weight Averaging (SWA) averages model weights along the training trajectory (typically the last 30% of training), improving generalization and calibration with negligible compute overhead.  | 0.6151 | - | rejected |
| e019 | e000 | inference | The pore class (AP 0.478) is the main weakness, confused with stain/deposit because small defects need more context. Earlier runs confirmed a fixed extra-wide view (4×) helps pores (+0.005 mAP) but ad | - | - | crashed |

Each experiment's folder (/data/ar2/runs/n1/eNNN/) holds train.py, diff.patch, the prompt, the LLM's reply and thinking, and the run logs.

## Provenance

| item | value |
|---|---|
| python | 3.12.3 |
| start_file | cb33b09ac6afffea0b9ba15234a362367d84fd3cc4ed8bedd4d9832b7e39c334 |
| brief | 023373b012478d21c104fdfb0eba879b16f844d85bd2dd27e1cb51af24694a29 |
| digest | fcd9348fd7f0eeb239881bfb36b5e2be8db268350f16dc525a50e441f1828906 |
| ar_loop.py | 659fbbc5cfaa377cb737267a8ed27bf80f900936a227d19de1d4339b25e47676 |
| det_metric.py | b055a87c2167735066dbb01f68e4027a09f768f4145f1bd58db2c33ac2685ce1 |
| data/dets.csv | a33042aa47ad358ab6e5700611ccc55a139cf20a5a5589ef8e2524e0abb6d43e |
| data/crops.csv | 7fa6269ca75cb4410aea5057138dc0ca73b5744ff04060893155025b338fb953 |
| hidden labels | 1fd0b668e18f41df8fdbee4d7e0502a806cd255e266637bf8455cbcd533d5f99 |
| packages_sha | 5a08c360913eeb3cd7195654571716f10f71401aa4df2ba9515f762380b0a662 |
| key_packages | numpy 2.5.4, pillow 13.0.0, timm 1.0.30, torch 2.14.1, torchvision 0.29.1 |
| train.py of start / final best | - / - |
| start-file runs reused from | b5 |
