Frozen state: lowest-Phi kept sample, rho* = 1.5577, Phi = 51.880 (noise floor 51.66); sigma_rel = 0.002; truth beta = 2.3

| beta from | median | 90% interval | sd(log beta) | covers truth | new PDE solves |
|---|---|---|---|---|---|
| prior at rho* (curve) | 2.247 | [1.164, 4.299] | 0.3941 | True | 0 |
| frozen (rho*, theta*), pCN on beta | 2.233 | [2.226, 2.240] | 0.0020 | False | 80250 |
| frozen (rho*, theta*), level swap | 2.214 | [1.454, 3.374] | 0.2554 | True | 0 |
| hard curve, all samples | 2.210 | [2.147, 2.321] | 0.0243 | True | 0 |
| level swap, all samples | 2.170 | [1.436, 3.264] | 0.2502 | True | 0 |

Exact 1D posterior at the frozen state (grid, no MCMC): sd(log beta) 0.0020, 90% beta [2.225, 2.240], covers False; 1/sqrt(Phi'' + 1/s*^2) = 0.0020 (sigma_rel 0.002).
pCN: acceptance 0.42, step 1.30e-02, ESS 13455; chains started from the prior.
