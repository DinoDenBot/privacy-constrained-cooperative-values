# Reproducibility package

This branch contains only the code and retained outputs needed to reproduce or
audit the empirical results reported in the paper. It deliberately excludes
the manuscript, review records, exploratory experiments, superseded analyses,
private workspace metadata, and datasets.

## Reported results covered

| Paper result | Code | Retained output |
| --- | --- | --- |
| Fashion-MNIST common-scaling slopes and figure data | `experiments/nonlinear-confirmatory/` | `results/scaling-confirmatory-20260903/` |
| UCI-HAR development certificates at epsilon 0.5 | `experiments/uci-har-logistic-curvature/` | `results/uci-har-logistic-curvature-20260910/` |
| MHEALTH confirmation certificates at epsilon 0.5 | `experiments/mhealth-logistic-curvature-confirmation/` | `results/mhealth-logistic-curvature-confirmation-20260910/` |
| Scope check for the positive third-order model on 273 retained games | audit script | `results/positive-three-feasibility-20260905/` |

The positive-third-order input and output contain exactly the 273 retained games
used by the paper. The excluded 45-game CIFAR diagnostic is not present.

## Quick verification from retained outputs

Python 3.11 or newer is sufficient:

```bash
python3 scripts/reproduce_paper_results.py
```

This command recomputes every number in the paper's empirical section from the
retained CSV files, checks the two plotted data tables, and exits nonzero on a
mismatch. It does not need network access or the source datasets.

To verify file integrity as well:

```bash
shasum -a 256 -c MANIFEST.sha256
```

## Re-running the experiments

The experiments use public datasets that are not redistributed here. Download
the canonical Fashion-MNIST, UCI-HAR, and MHEALTH releases, then verify them
against the checked-in input hashes before running anything. Detailed staged
commands are in each experiment directory.

Fashion-MNIST scaling:

```bash
uv run --directory experiments/nonlinear-confirmatory --frozen python \
  experiments/nonlinear-confirmatory/run.py \
  --config experiments/nonlinear-confirmatory/config-scaling.json \
  --data .data/fashion-mnist \
  --output reproduced/scaling-confirmatory
```

The UCI-HAR and MHEALTH experiments are deliberately split into profile
preparation, certificate construction, and held-out evaluation. Follow their
READMEs in order. Complete coalition enumeration is used only to audit the
endpoint-based scores and certificates.

The positive-third-order scope check can be regenerated independently from its
compact endpoint input:

```bash
python3 scripts/analyze_positive_three_feasibility.py \
  --output reproduced/positive-three-feasibility
```

## Scope and anonymity

No manuscript source, author list, acknowledgments, review material,
machine-local username, or private dataset is included. Subject identifiers
appearing in the public UCI datasets and deterministic seed identifiers are
experimental variables, not personal metadata added by the researchers.

The retained output bundles include environment versions and temporary generic
paths needed for provenance. They contain no home-directory paths. Before a
public anonymous release, also inspect Git hosting metadata and commit authors,
because file-level checks cannot anonymize the hosting account itself.
