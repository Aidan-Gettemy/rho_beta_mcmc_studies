| grid | blocks | p | r_+ (tau = 1e-10) | d_1 | d_min | cond(Sigma_ms) | cond(B) | cut tie groups | result |
|---|---|---|---|---|---|---|---|---|---|
| primal | 16 | 64 | 58 / 64 | 2.396e+00 | -2.929e-16 | inf | 2.213e+16 | 0 | FAIL |
| dual | 25 | 114 | 64 / 64 | 3.029e+00 | 2.838e-01 | 1.067e+01 | 3.267e+00 | 0 | PASS |

Primal block dimensions: [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]

Dual block dimensions (row-major over the 5 x 5 tiling): [3, 4, 4, 4, 3, 4, 6, 6, 6, 4, 4, 6, 6, 6, 4, 4, 6, 6, 6, 4, 3, 4, 4, 4, 3]

Whitening operator check (primal): 6.473e-15
