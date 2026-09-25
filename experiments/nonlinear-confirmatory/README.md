# Fashion-MNIST common-scaling experiment

This directory contains only the Fashion-MNIST configuration used for the
paper's common-scaling result. It trains a deterministic one-hidden-layer tanh
network and enumerates all 256 coalitions of eight fixed client updates under
fixed aggregation weights.

The primary utility is improvement in negative multiclass Brier score. It is
smooth and bounded, separating nonlinear interaction from finite-test accuracy
discontinuities. Negative cross-entropy and accuracy are secondary utilities.

Inputs are the four canonical Fashion-MNIST IDX gzip files. `input-hashes.json`
records their SHA-256 digests. The run is deterministic given `config.json` and
writes only to a newly created output directory.
`MANIFEST.sha256` seals the executable bundle.

`config-scaling.json` is the frozen theorem-linked scaling experiment. It uses
new partition seeds, holds each 20-epoch update direction fixed, scales every
weighted update by a common factor, records actual update norms, and tests the
predicted cubic absolute and quadratic relative midpoint-error orders.

Run from the repository root:

```bash
uv run --directory experiments/nonlinear-confirmatory --frozen python \
  experiments/nonlinear-confirmatory/run.py \
  --config experiments/nonlinear-confirmatory/config-scaling.json \
  --data .data/fashion-mnist \
  --output reproduced/scaling-confirmatory
```
