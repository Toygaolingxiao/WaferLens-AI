"""Route B: small CNN on raw wafer maps (PyTorch).

TODO(owner) — this is a core rehearsal piece, intentionally left to build:
  1. Dataset wrapper: resize/pad variable-size maps to a fixed input
     (e.g. 64x64), encode die states as channels (die mask, defect mask).
  2. A compact CNN (3-4 conv blocks) — capacity matched to ~25k labelled maps,
     not an ImageNet-scale model.
  3. Train with the same lot-grouped split as route A; report mean +/- spread
     over at least 2 seeds through waferlens.eval.metrics, never a lone run.
  4. Compare per-class recall against route A before touching architecture:
     where the CNN wins and loses is the finding, not the headline F1.
"""
