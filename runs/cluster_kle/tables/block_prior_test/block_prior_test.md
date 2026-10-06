Block pCN, 32x32, r=57, energy >= 0.9, halo 0.0: primal dims [9, 9, 9, 9], dual dims [3, 4, 3, 4, 6, 4, 3, 4, 3], coarse 21; spans R^r: True

| reach (prior variance the moves can change / prior variance) | min | median | max |
|---|---|---|---|
| primal | 0.00 | 0.84 | 1.00 |
| primal+dual | 0.21 | 1.18 | 1.99 |
| coarse | 0.00 | 0.15 | 1.00 |

Prior test: 32 chains x 4000 sweeps, s = 0.3, likelihood off; direct draws: cov error 0.043

| schedule | moves / sweep | cov(θ) error | pointwise var / target | lag-1 per sweep | lag-1 per move |
|---|---|---|---|---|---|
| coarse+primal | 5.0 | 0.099 | 0.96–1.05 | 0.952–0.956 | 0.9903–0.9911 |
| coarse+primal+dual | 14.0 | 0.082 | 0.96–1.03 | 0.908–0.955 | 0.9931–0.9967 |
| random | 2.0 | 0.080 | 0.95–1.04 | 0.951–0.996 | 0.9752–0.9981 |
