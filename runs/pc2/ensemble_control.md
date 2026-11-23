# Ensemble control pc2: does the held-out gate accept a real gain and reject a non-change?

Baseline: start file, 180 s, seeds from b5. Candidates: the same file trained with fresh seeds 10..19; single = seeds 10..14; ensemble = mean scores of seed pairs (10+11) ... (18+19). Gate as in h1 and n1: visible half (82 images) > 1.0 SE, then hidden half (82 images) > 0.5 SE. SE over seeds and images. Test scores read afterwards, for analysis only.

| candidate | baseline | visible val gain ± SE | passes | hidden val gain ± SE | passes | gate | test gain ± SE |
|---|---|---|---|---|---|---|---|
| single model, fresh seeds (null control) | 8 seeds | +0.0031 ± 0.0041 | no | +0.0042 ± 0.0064 | yes | rejected | +0.0019 ± 0.0035 |
| single model, fresh seeds (null control) | 5 seeds | +0.0033 ± 0.0043 | no | +0.0008 ± 0.0070 | no | rejected | +0.0023 ± 0.0041 |
| 2-seed ensemble (positive control) | 8 seeds | +0.0113 ± 0.0031 | yes | +0.0109 ± 0.0047 | yes | ACCEPTED | +0.0090 ± 0.0027 |
| 2-seed ensemble (positive control) | 5 seeds | +0.0115 ± 0.0039 | yes | +0.0075 ± 0.0056 | yes | ACCEPTED | +0.0094 ± 0.0034 |

Verdict (8-seed baseline):
- single model, fresh seeds (null control): PASS: no real change, and the gate rejected it.
- 2-seed ensemble (positive control): PASS: really better on test, and the gate accepted it.
