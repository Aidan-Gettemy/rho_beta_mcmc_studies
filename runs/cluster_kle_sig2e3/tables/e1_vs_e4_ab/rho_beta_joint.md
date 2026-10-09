| sampler | ρ median [90%] | β median [90%] | sd(log ρ) | sd(log β) | corr(log ρ, log β) |
|---|---|---|---|---|---|
| naive block-pCN (E1) | 1.504 [1.251, 1.834] | 1.916 [1.150, 3.472] | 0.116 | 0.342 | -0.127 |
| curve sampler (A+B) + level split | 1.473 [1.231, 1.794] | 2.170 [1.436, 3.261] | 0.114 | 0.250 | -0.027 |

Truth: ρ = 1.5, β = 2.3. E1: log beta ~ N(0.0, 1.0^2), log rho ~ N(0.0, 0.75^2); 64 chains, burn-in 20000, every 10th iteration. E4: 64 chains, level-split weight ESS 99.8%.
