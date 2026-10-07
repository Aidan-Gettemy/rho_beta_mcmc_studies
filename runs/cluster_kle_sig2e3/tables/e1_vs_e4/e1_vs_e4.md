E1 naive (log beta ~ N(0,1), log rho ~ N(0,0.75^2)) vs E4 3-move on the curve (hard -> level swap). Same KLE truth, data, sigma_rel 2e-3, 64 chains.

| | E1 naive | E4 3-move | truth |
|---|---|---|---|
| Solves per chain | 2.0M | 2.0M |  |
| θ MPSRF, final | 1.21 | 1.12 |  |
| log β PSRF | 1.21 | — (β on the curve) |  |
| ρ | 1.50 [1.25, 1.83] | 1.47 [1.23, 1.78] | 1.5 |
| Level | 0.862 [0.736, 0.938] | 0.867 [0.754, 0.942] | 0.835 |
| β | 1.92 [1.15, 3.47] | 2.21 [2.15, 2.32] → 2.18 [1.44, 3.26] | 2.3 |
| sd of log β | 0.342 | 0.024 → 0.249 |  |
| Mean Φ (floor 51.7) | 69.5 | 69.6 |  |
