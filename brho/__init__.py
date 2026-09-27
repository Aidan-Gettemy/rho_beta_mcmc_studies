"""brho: (beta, rho) permeability identification - 2D experiments.

Modules
    config       dataclass configs for every experiment (+ QUICK budgets)
    physics2d    Darcy2D: FV permeameter solver, Keff, chessboard observations
    kle          KLE: global Karhunen-Loeve expansion of C1
    truth        make_synthetic: reference field, pressure data, K*
    problem      Problem: physics + KLE + data + potential Phi(K)
    curve        E3: Monte Carlo sweep and BetaCurve (beta_bar(rho) = K*/f(rho))
    samplers     E2 pcn_fixed_pair; E4 ThreeStepSampler + checkpointed chains
    diagnostics  PSRF, MPSRF, split-R-hat, ESS, summaries, coverage
    io           results/ figures/ tables/ layout, manifests, table writers
    plotting     shared figure helpers
"""
