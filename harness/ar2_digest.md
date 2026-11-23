These are general techniques from the literature, with the conditions under which they tend to help or hurt.
None of it is specific to this dataset; whether any of it helps here is for your experiments to show.

### Second-stage classifiers in two-stage detection
- R-CNN-family detectors train the second stage on the first stage's own proposals, labelled by overlap with
  ground truth (e.g. IoU >= 0.5 positive, < 0.5 background), usually sampling positives and background in a fixed
  ratio (Faster R-CNN: up to 1:3). Training on proposals instead of perfect boxes matches the test distribution.
  (Ren et al., Faster R-CNN, 2015)
- Cascade R-CNN shows that a classifier trained at one IoU threshold is best for boxes of that quality; box
  quality at training and test time should match. (Cai & Vasconcelos, 2018)
- When the first stage already ranks objects well, the final score is often the product of the first-stage
  confidence and the second-stage class probability; how the two are combined changes the ranking only where the
  classifier's probabilities differ between boxes.

### Fine-grained recognition of small objects
- Distinguishing visually similar classes usually depends on fine texture; input resolution and how much of the
  crop the object fills both matter. Pretrained backbones are typically trained at a fixed resolution and often
  work best near it. (Touvron et al., "Fixing the train-test resolution discrepancy", 2019)
- Context around an object can help separate classes that look alike up close; several views at different
  scales can be combined (multi-scale or multi-crop inputs), at a proportional compute cost.

### Training under a fixed time budget
- With a hard wall-clock budget, the number of optimisation steps is limited; a smaller or faster model, lower
  resolution, or fewer views can win over a stronger but slower one, and the reverse can also hold.
  Learning-rate schedules that decay to zero by the end of the budget usually beat schedules designed for longer
  training. (Li, Yumer & Ramanan, "Budgeted Training", 2020)
- Data loading and image decoding can be the bottleneck rather than the GPU; caching or pre-cropping can buy
  steps. Mixed precision and channels-last memory format speed up convolutional networks on recent GPUs.
- Efficient backbone families trade accuracy against speed differently (EfficientNet / EfficientNetV2,
  MobileNetV3, RegNet, ConvNeXt); some need different learning rates or warm-up to fine-tune stably.

### Variance reduction: ensembles, averaging, test-time augmentation
- Averaging the predictions of several independently trained models is one of the most reliable ways to improve
  accuracy and calibration, at a linear cost in compute. (Lakshminarayanan et al., Deep Ensembles, 2017)
- Cheaper variants: snapshot ensembles from one training run (Huang et al., 2017), weight averaging along the
  trajectory (SWA, Izmailov et al., 2018), and test-time augmentation (averaging over flips or crops).
- Under a fixed budget, an ensemble means less training per member; the trade-off depends on how undertrained
  each member is.

### Losses, regularisation and data
- Focal loss helps dense detectors with extreme foreground/background imbalance (Lin et al., 2017); for a crop
  classifier with moderate imbalance its benefit is less clear.
- Label smoothing improves calibration in some settings but can reduce the information in class probabilities
  that ranking metrics use (Muller, Kornblith & Hinton, 2019).
- Mixup and CutMix regularise strongly and typically need longer training to pay off (Zhang et al., 2018;
  Yun et al., 2019).
- Adding data from a different source or resolution can help or hurt depending on how different it is from the
  test data (domain shift).

### Ranking metrics
- Average precision depends only on the order of predictions within each class. Any transformation that keeps
  that order (scaling, temperature, most monotone functions) leaves AP unchanged; changes help only if they
  re-order true detections above false ones.
- Temperature scaling and other calibration methods fix probability levels, not rankings (Guo et al., 2017).

### Experimental practice
- Single runs of a neural network vary with the seed; small differences need several seeds to be trusted.
- A result far below the baseline usually indicates a bug (wrong preprocessing, label order, untrained model)
  rather than a bad idea; checking the training loss and a few predictions often finds it.
- Locating the error first (which classes are confused, which sizes fail, how much is lost to ranking vs.
  classification) tends to point to the change with the largest effect.
