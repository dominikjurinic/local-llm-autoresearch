Five earlier 5-hour runs (90 ideas) started from this same train.py with the same data, budget and seeds. The
start file scores 0.629-0.630 val mAP@0.5 (5-seed mean). Below is every idea family they tried, with validation
scores only: "seed 0" is a single run, "5-seed" a confirmed mean. Those runs used a looser acceptance rule (gain
above the seed noise only). Some of their "accepted" changes would not pass today's rule, which also counts the
noise from the particular validation images.

How to use this: these runs are your own earlier work. Do not repeat a family marked "no gain" unless you can say
what is fundamentally different and why that should change the result. The point of tonight's run is to find
something these families did not.

| idea family | attempts | validation result |
|---|---|---|
| Fewer or re-weighted score rows: only the argmax class per box, dropping non-argmax rows, ranking by margin or entropy, softmax temperature, other conf x p rules | 16 | no gain: seed 0 from 0.555 to 0.632, best 5-seed 0.627; 2 bugs (0.17, 0.16). AP of a class depends only on the order of that class's rows; low wrong-class rows cost little, removing rows loses recall |
| 5th "background" class from detector false alarms (scores times 1 - p_bg) | 9 | no gain: seed 0 from 0.518 to 0.629; 2 crashes, 1 bug (0.0) |
| Detector boxes from the train split as extra training data (IoU > 0.5 with a true box, labelled with its class) | 6 | accepted twice (5-seed 0.6337 and 0.6342, about +0.004); rejected 3 times at seed 0 (0.626-0.628); 1 crash in the IoU code |
| Third, very wide view (4x the box) next to the 1.5x and 2.5x crops, logits averaged | 3 | accepted twice (5-seed 0.6342 both, +0.005), gain almost entirely from the pore class; once rejected (0.633). Adds ~20 s per run, close to the time limit |
| Low-resolution crops added to training | 4 | no gain: best 5-seed 0.629 |
| Bigger or other backbone (EfficientNet-B2, ResNet-50, 2-3 model ensembles) | 8 | no gain: best 5-seed 0.636 (not confirmed), others 0.600-0.631; 1 timeout. Fewer steps fit in 180 s |
| Resolution and small-box handling (224 px, larger MIN_CROP, enlarged tight crops for small boxes) | 9 | no gain: best 5-seed 0.634 (not confirmed); 2 crashes |
| Test-time augmentation (flips, extra views at inference) | 8 | no gain: best 5-seed 0.631; 8 views timed out |
| Longer training (smaller reserve, measured prediction time) | 6 | no gain: best 5-seed 0.6355 (not confirmed); 2 bugs where a timing probe ate the training time (0.44, 0.40) |
| Class-balanced sampling or class-weighted loss | 4 | no gain: seed 0 from 0.613 to 0.629 |
| Training augmentation (rotation, mixup, stronger colour or geometry) | 5 | no gain: best 5-seed 0.633 (not confirmed) |
| Focal loss | 2 | no gain: 0.610, 0.619 |
| Box size and aspect ratio as an extra model input | 4 | never properly tested: 2 crashes (wrong attribute name), 1 edit failure, 1 run at 0.629 |

Every family above came up in three or more of the five runs. Ideas outside these families have not been tried.
