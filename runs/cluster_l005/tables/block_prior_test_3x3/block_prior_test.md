Block pCN, 64x64, r=208, energy >= 0.9, halo 0.0: primal dims [17, 16, 17, 16, 13, 16, 17, 16, 17], dual dims [4, 7, 7, 4, 7, 13, 13, 7, 7, 13, 13, 7, 4, 7, 7, 4], coarse 63; spans R^r: True

| reach (prior variance the moves can change / prior variance) | min | median | max |
|---|---|---|---|
| primal | 0.00 | 0.91 | 1.00 |
| primal+dual | 0.22 | 1.25 | 2.00 |
| coarse | 0.00 | 0.09 | 1.00 |

Prior test: 32 chains x 4000 sweeps, s = 0.3, likelihood off; direct draws: cov error 0.081

| schedule | moves / sweep | cov(θ) error | pointwise var / target | lag-1 per sweep | lag-1 per move |
|---|---|---|---|---|---|
| coarse+primal | 10.0 | 0.187 | 0.94–1.05 | 0.951–0.957 | 0.9950–0.9956 |
| coarse+primal+dual | 26.0 | 0.153 | 0.96–1.05 | 0.907–0.955 | 0.9962–0.9982 |
| random | 2.0 | 0.162 | 0.93–1.08 | 0.952–0.998 | 0.9756–0.9990 |
