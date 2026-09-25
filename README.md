# Privacy-Constrained Cooperative Values: Shapley Identification under Partial Information

Reproducibility code for the empirical results reported in the paper.

This branch is a standalone code package. It does **not** contain the paper,
LaTeX sources, PDFs, bibliography files, reviews, or author information. It
contains only the experiment code, frozen configurations, public-data hashes,
minimal processed inputs, and retained outputs needed to reproduce or audit the
reported results.

## Results covered

The package covers exactly the empirical results retained in the paper:

1. **Fashion-MNIST scaling:** midpoint error under five common update scales,
   including the reported median absolute and relative slopes of 3.125 and
   2.110.
2. **UCI-HAR development experiment:** curvature certificates at
   `epsilon = 0.5`, including interval width, sign, ranking, and top-client
   decisions.
3. **MHEALTH confirmation experiment:** the prespecified certificate result at
   `epsilon = 0.5`, including the 1.24% median relative width and all reported
   decision counts.
4. **Positive third-order scope check:** feasibility on the 273 retained games,
   of which 6 admit a positive third-order extension and 0 of those 6 contain
   the observed exact Shapley vector in every coordinate interval.

Exploratory, superseded, and out-of-scope experiments are excluded. In
particular, the 45-game CIFAR diagnostic is not present.

## Quick verification

Python 3.11 or newer is sufficient to audit the retained outputs:

```bash
python3 scripts/reproduce_paper_results.py
```

The script independently recomputes every number reported in the empirical
section from the retained CSV files. It also checks the data used for the two
empirical figures and exits with a nonzero status if any value differs.

Verify the complete package against its checksum manifest with:

```bash
shasum -a 256 -c MANIFEST.sha256
```

## Repository layout

```text
experiments/
  nonlinear-confirmatory/                  Fashion-MNIST scaling code
  uci-har-logistic-curvature/              UCI-HAR certificate code
  mhealth-logistic-curvature-confirmation/ MHEALTH confirmation code
inputs/
  positive_three_endpoints.json            273-game scope-check input
results/                                   Retained machine-readable outputs
figures/data/                              Audited data tables, without paper files
scripts/
  reproduce_paper_results.py               One-command result audit
  analyze_positive_three_feasibility.py    Scope-check analysis
  plot_midpoint_scaling.py                 Scaling-figure data generation
```

## Re-running the experiments

The source datasets are public but are not redistributed. Download the
canonical Fashion-MNIST, UCI-HAR, and MHEALTH releases and verify them against
the checked-in hashes before running an experiment.

### Fashion-MNIST scaling

The environment is pinned by `experiments/nonlinear-confirmatory/uv.lock`.
From the repository root, run:

```bash
uv run --directory experiments/nonlinear-confirmatory --frozen python \
  experiments/nonlinear-confirmatory/run.py \
  --config experiments/nonlinear-confirmatory/config-scaling.json \
  --data .data/fashion-mnist \
  --output reproduced/scaling-confirmatory
```

### UCI-HAR development certificates

The UCI-HAR pipeline separates profile preparation, certificate construction,
and held-out evaluation. Follow the commands in
`experiments/uci-har-logistic-curvature/README.md` in order. The official input
hashes are in `experiments/uci-har-diagnostic/input-hashes.json`.

### MHEALTH confirmation certificates

The confirmation pipeline uses the same staged design. Follow
`experiments/mhealth-logistic-curvature-confirmation/README.md`. The archive and
extracted-file hashes are in that directory's `input-hashes.json`.

### Positive third-order scope check

This analysis is self-contained because its 273 endpoint observations are
included as a compact processed input:

```bash
python3 scripts/analyze_positive_three_feasibility.py \
  --output reproduced/positive-three-feasibility
```

## Tests

The three experiment bundles include unit tests. A single pinned environment
can run all of them:

```bash
uv run --directory experiments/nonlinear-confirmatory --frozen \
  python test_bundle.py

uv run --directory experiments/nonlinear-confirmatory --frozen \
  python ../../experiments/uci-har-logistic-curvature/test_bundle.py

uv run --directory experiments/nonlinear-confirmatory --frozen \
  python ../../experiments/mhealth-logistic-curvature-confirmation/test_bundle.py
```

Complete coalition enumeration is used only to establish exact experimental
ground truth. The evaluated midpoint scores and certificates receive only the
endpoint observation and the public controls stated by each experiment.

## Data and anonymity

No private dataset is included. Numeric subject identifiers come from the
public UCI datasets and deterministic seeds are experimental variables. The
files contain no author names, acknowledgments, home-directory paths, account
identifiers, or manuscript source.
