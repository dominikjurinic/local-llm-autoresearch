# Positive control pc1: does the held-out gate accept a real improvement?

Baseline: start file, 180 s budget (5 seeds, from b5). Controls: the same file, unchanged, with a longer budget (5 seeds each). Gate as in h1 and n1: gain on the visible half (82 images) > 1.0 SE, then gain on the hidden half (82 images) > 0.5 SE. SE over seeds and images. Test scores read afterwards, for analysis only.

| control | visible val gain ± SE | passes | hidden val gain ± SE | passes | gate | test gain ± SE | real on test (> 2 SE) |
|---|---|---|---|---|---|---|---|
| 360 s | -0.0173 ± 0.0082 | no | -0.0100 ± 0.0102 | no | rejected | -0.0119 ± 0.0063 | no |
| 720 s | -0.0430 ± 0.0111 | no | -0.0447 ± 0.0151 | no | rejected | -0.0327 ± 0.0078 | no |

Test mAP@0.5: baseline 0.6865; 360 s 0.6746; 720 s 0.6538

Verdict:
- 360 s: no verdict: not clearly better on test, so this control cannot test the gate. Use a longer budget.
- 720 s: no verdict: not clearly better on test, so this control cannot test the gate. Use a longer budget.
