Frozen state: lowest-Phi kept sample, rho* = 1.7066, Phi = 52.725 (noise floor 51.66); sigma_rel = 0.002; truth beta = 2.3

| beta from | median | 90% interval | sd(log beta) | covers truth | new PDE solves |
|---|---|---|---|---|---|
| prior at rho* (curve) | 2.300 | [1.103, 4.753] | 0.4405 | True | 0 |
| frozen (rho*, theta*), pCN on beta | 2.291 | [2.283, 2.298] | 0.0020 | False | 80250 |
| frozen (rho*, theta*), level swap | 2.261 | [1.421, 3.601] | 0.2821 | True | 0 |
| hard curve, all samples | 2.208 | [2.146, 2.317] | 0.0240 | True | 0 |
| level swap, all samples | 2.175 | [1.444, 3.266] | 0.2490 | True | 0 |

Exact 1D posterior at the frozen state (grid, no MCMC): sd(log beta) 0.0020, 90% beta [2.283, 2.298], covers False; 1/sqrt(Phi'' + 1/s*^2) = 0.0020 (sigma_rel 0.002).
pCN: acceptance 0.45, step 9.06e-03, ESS 13160; chains started from the prior.
