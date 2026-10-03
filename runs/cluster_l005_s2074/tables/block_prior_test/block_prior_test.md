Block pCN, 64x64, r=208, energy >= 0.9, halo 0.0: primal dims [41, 41, 41, 41], dual dims [9, 17, 9, 17, 33, 17, 9, 17, 9], coarse 44; spans R^r: True

| reach (prior variance the moves can change / prior variance) | min | median | max |
|---|---|---|---|
| primal | 0.00 | 0.99 | 1.00 |
| primal+dual | 0.16 | 1.54 | 2.00 |
| coarse | 0.00 | 0.01 | 1.00 |

Prior test: 32 chains x 4000 sweeps, s = 0.3, likelihood off; direct draws: cov error 0.081

| schedule | moves / sweep | cov(θ) error | pointwise var / target | lag-1 per sweep | lag-1 per move |
|---|---|---|---|---|---|
| coarse+primal | 5.0 | 0.186 | 0.93–1.06 | 0.951–0.957 | 0.9899–0.9912 |
| coarse+primal+dual | 14.0 | 0.150 | 0.95–1.05 | 0.906–0.955 | 0.9930–0.9967 |
| random | 2.0 | 0.174 | 0.92–1.06 | 0.951–0.996 | 0.9751–0.9982 |
