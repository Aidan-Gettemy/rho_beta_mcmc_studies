# Run log

One entry per real run: date, command, settings, headline numbers, what changed.
Quick (`--quick`) smoke runs are not logged; they go to `runs/quick/`, which git ignores.

## 2026-09-26 - E2 grid study on the July data (`runs/july_raw/`)

- Command: `make e2 ROOT=runs/july_raw ARGS="--truth-mode raw --n-workers 2"`, plus the setup and combined reports
- Settings: 5×5 grid, β ∈ {0.5, 1, 2.3, 5, 10} × ρ ∈ {0.5, 1, 1.5, 2.25, 3}; 4000 iterations, 2000 burn-in per chain; r = 56
- Data check: σp = 3.61e-3, K*_obs = 3.19, Ḡ = 0.286, which matches the July reports exactly
- Result: every pair fits at RMS/σp 1.31–2.77 (July report: 1.4–2.5); best Φ 100–421 (July: 110–268) against the floor of 64; acceptance 0.08–0.21. The top row (ρ = 3) is warmest, as in the July report.
- Open: the upper ends of RMS/σp and best Φ are wider than in the July report (not yet investigated). Runtime 0.6 min on 2 cores.

## Pending decisions before E3 and E4 at full budget

1. Truth: `standardized` (default) or `raw`
2. Curve statistic: `ratio_of_mean` (default) or `mean_of_ratio`
3. Accepted: the ridge move costs one solve with the curve (3 solves per iteration)
4. Accepted: the ρ prior is truncated at the curve range, ρ ≤ 5 (1.6% of prior mass)
