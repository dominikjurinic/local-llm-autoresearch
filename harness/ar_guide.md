This guide comes from a senior researcher who has read all earlier runs on this task. It sets the strategy for
tonight; you choose and implement each experiment. Follow it unless your own results clearly contradict it, and
if they do, say so in the hypothesis.

### What the problem really is
- Stage 2 is a classifier that is very good on clean, hand-drawn boxes but is scored on stage-1 detector boxes:
  slightly misplaced, differently sized, many of them false alarms. Most of the remaining error is the class
  decision on detector boxes that do overlap a real defect (see the diagnostics: the gap between your mAP and
  "mAP if every box had its correct class"). Treat this as a train/test mismatch problem first, a model-capacity
  problem second.
- The binding constraint is 180 s. Every second of training you gain is a free improvement; every slower model,
  bigger input or extra view must pay for itself in fewer training steps.
- Small, one-class gains are not accepted tonight. Aim for changes that should move every class.

### Plan for the night (about 30 experiments)
1. **Consolidate (first 2-4 experiments).** The only idea earlier runs accepted twice is training on detector
   boxes from the train split (IoU > 0.5 with a true box, labelled with its class), alongside the true boxes.
   Implement it carefully (vectorised IoU, correct shapes) and confirm it. If it is accepted, build on it; if it
   fails twice, drop it and move on.
2. **Close the box-quality gap (main phase).** Make the classifier robust to imperfect boxes. Directions, roughly
   in order of expected value:
   - simulate detector errors on the true boxes during training: random shifts and scale changes so that the
     IoU between the jittered box and the true box covers the range the detector produces (about 0.5 to 1.0),
     instead of the current small crop jitter;
   - weight or sample training boxes by box quality, so the model sees enough poorly placed boxes;
   - combine the two views better than averaging logits, for example one shared backbone whose pooled
     features from both crops are joined before a small head (one forward pass per view, no extra cost).
3. **Buy training steps (whenever the log shows the GPU waiting or the loss still falling at the cut).** Read
   the log: load time, steps, epochs, prediction time. Ideas: prepare crops once instead of per step, fewer
   or cheaper data-loader operations, a reserve that matches the measured prediction time with a safety margin
   (earlier runs broke this twice by timing during warm-up: measure after warm-up, never before training).
   Each step gained is worth more than a bigger model.
4. **Fine-grained classes (only after 2 and 3).** Use the confusion matrix in the diagnostics. Look for one
   change that helps the whole pore / deposit / stain confusion, not one class.
5. **Last 1.5 hours: no big swings.** Only follow-ups on what worked tonight, with low crash risk.

### Rules of thumb
- Before each experiment, write down which diagnostic number should move, and in which direction, if the
  hypothesis is right (e.g. "true pore predicted as stain drops from ~X to ~Y"). After the result, check
  whether it moved. A score change without the predicted diagnostic change is probably noise.
- Two failed variants of a direction: move to the next direction. A bug: fix it once; a second failure: drop it.
- Re-check the time arithmetic for every change that adds work per box (views, resolution, model size, test-time
  augmentation). Keep the whole run under about 165 s.
- Do not spend experiments on the families listed as "no gain" in the results of earlier runs.
