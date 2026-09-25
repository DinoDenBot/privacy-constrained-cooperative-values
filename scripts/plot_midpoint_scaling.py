#!/usr/bin/env python3
"""Render existing midpoint summaries from the original fixed-profile results."""
import csv
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / 'results/scaling-confirmatory-20260903/rounds.csv'
with source.open() as stream:
    raw = [r for r in csv.DictReader(stream) if r['utility'] == 'negative_brier']
rows = []
for epsilon in sorted({float(r['update_scale']) for r in raw}):
    selected = [r for r in raw if float(r['update_scale']) == epsilon]
    assert len(selected) == 30
    row = {'epsilon': epsilon}
    for metric, field in [('abs', 'absolute_midpoint_gap'), ('rel', 'relative_midpoint_gap')]:
        values = [float(r[field]) for r in selected]
        for suffix, probability in [('lo', .25), ('med', .5), ('hi', .75)]:
            row[f'mid_{metric}_{suffix}'] = float(np.quantile(values, probability))
    rows.append(row)
for row in rows:
    for metric, power in [('abs', 3), ('rel', 2)]:
        row[f'mid_{metric}_ref'] = rows[0][f'mid_{metric}_med'] * (row['epsilon'] / rows[0]['epsilon']) ** power
output = ROOT / 'figures/data/midpoint_scaling.csv'
with output.open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
print(output)
