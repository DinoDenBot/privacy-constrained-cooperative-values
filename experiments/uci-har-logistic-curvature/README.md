# Bounded-logistic UCI-HAR curvature certificate

This frozen experiment asks whether the previous certificate failed because of
the multilayer evaluator rather than because useful curvature intervals are
intrinsically unavailable. It keeps natural UCI-HAR subject-clients but replaces
the tanh MLP with multinomial logistic regression, projects every augmented
feature vector into the unit Euclidean ball, and enforces a public Frobenius
clip on each local update.

The certificate is constructed before official-test access. It uses only the
feature radius, the clipping limit, and public sample-size weights. Exact games
are opened later to evaluate the certificate; they are not construction inputs.

## Frozen stages

After staging the hash-verified public UCI-HAR train and test files, prepare the
profiles without test access:

```bash
uv run --with numpy==2.3.2 python experiments/uci-har-logistic-curvature/prepare_profiles.py \
  --train-dir /staged/UCI-HAR/train \
  --config experiments/uci-har-logistic-curvature/config.json \
  --output /sealed/uci-har-logistic-profiles
```

Construct the certificates from the separately emitted public caps:

```bash
uv run --with numpy==2.3.2 python experiments/uci-har-logistic-curvature/construct.py \
  --caps /sealed/uci-har-logistic-profiles/authorized-caps.json \
  --config experiments/uci-har-logistic-curvature/config.json \
  --output /sealed/uci-har-logistic-certificates
```

Only then evaluate exact games on the official test split:

```bash
uv run --with numpy==2.3.2 python experiments/uci-har-logistic-curvature/evaluate.py \
  --profiles /sealed/uci-har-logistic-profiles \
  --certificates /sealed/uci-har-logistic-certificates/certificates.json \
  --test-dir /staged/UCI-HAR/test \
  --config experiments/uci-har-logistic-curvature/config.json \
  --output /results/uci-har-logistic-curvature
```

Successful execution is distinct from scientific acceptance. The hypothesis is
contradicted unless every predeclared gate passes at the natural scale
`epsilon=1.0`. The smaller scales are descriptive sensitivity results.

## Compute offer

- target: local Mac mini;
- logical CPUs: at most 6;
- resident memory: at most 4 GiB;
- runtime ceiling: 1 hour;
- output ceiling: 512 MiB;
- GPU: none;
- network during execution: disabled;
- declared input: the public UCI-HAR files already covered by
  `../uci-har-diagnostic/input-hashes.json`;
- expected outputs: sealed profiles and caps, certificates, exact games,
  player/game tables, a decision record, environment metadata, and checksums.

The public UCI-HAR input is external to the immutable bundle and must pass the
existing file hashes before execution. A hash-verified copy was staged while
validating this offer. The cheaper alternative is the packaged scikit-learn
Digits dataset, but it would replace natural subjects with synthetic Dirichlet
clients and is therefore less informative for the paper.
