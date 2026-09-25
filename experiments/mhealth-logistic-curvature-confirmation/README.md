# MHEALTH bounded-logistic curvature confirmation

This frozen CPU experiment transfers the successful UCI-HAR half-scale setting
to a separate public natural-client dataset. The ten MHEALTH volunteers supply
20 deterministic role assignments: one checkpoint subject, one designated
held-out test subject, and eight client subjects. The model, feature bound,
clipping rule, `epsilon=0.5`, curvature formula, and decision thresholds are
unchanged from the UCI-HAR experiment.

The dataset is UCI MHEALTH, DOI `10.24432/C5TW22`, licensed CC BY 4.0. The
downloaded archive and all eleven extracted files are hash-bound in the frozen
configuration.

Run the stages in order:

```bash
uv run --with numpy==2.3.2 python experiments/mhealth-logistic-curvature-confirmation/prepare_profiles.py \
  --data-dir /staged/MHEALTHDATASET \
  --config experiments/mhealth-logistic-curvature-confirmation/config.json \
  --output /sealed/mhealth-logistic-profiles

uv run --with numpy==2.3.2 python experiments/uci-har-logistic-curvature/construct.py \
  --caps /sealed/mhealth-logistic-profiles/authorized-caps.json \
  --config experiments/mhealth-logistic-curvature-confirmation/config.json \
  --output /sealed/mhealth-logistic-certificates

uv run --with numpy==2.3.2 python experiments/mhealth-logistic-curvature-confirmation/evaluate.py \
  --profiles /sealed/mhealth-logistic-profiles \
  --certificates /sealed/mhealth-logistic-certificates/certificates.json \
  --data-dir /staged/MHEALTHDATASET \
  --config experiments/mhealth-logistic-curvature-confirmation/config.json \
  --output /results/mhealth-logistic-curvature-confirmation
```

The local offer uses at most six logical CPUs, 4 GiB memory, one hour, and
512 MiB output. It needs no GPU and no network after the public archive is
staged. Execution success is not scientific acceptance: every unchanged gate
must pass.
