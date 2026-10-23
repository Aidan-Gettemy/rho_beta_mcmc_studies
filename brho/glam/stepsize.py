"""Two-step-size adaptation for GLAM: beta_G (global stages) and beta_L (local stages).

Same rule as the group's ParameterStepSizeAdapter: Robbins-Monro on the logit scale,

    logit(beta_k) += gain_n * (acceptance_k - target),     gain_n = initial_gain / max(1, n - gain_hold)**gain_exponent,

where n counts the updates of that step size (gain_hold = number of updates at the full initial gain before the
decay starts; gain_exponent 0 = constant gain), with the acceptance counts of a window pooled over all chains by an Allreduce at fixed, synchronized iterations.
Adaptation runs only inside window 1 (iterations [adapt_start, adapt_stop)); `freeze()` ends it for good.  Parameter 0
is beta_G, parameter 1 is beta_L; a parameter that was not proposed in a window (e.g. beta_G during a local stage)
is left unchanged.
"""
import numpy as np

GLOBAL, LOCAL = 0, 1


class TwoStepAdapter:
    def __init__(self, comm, beta_global=0.2, beta_local=0.2, target=0.25, adapt_start=0, adapt_stop=2000,
                 adapt_interval=100, initial_gain=0.5, gain_exponent=0.6, gain_hold=0, beta_min=1e-3, beta_max=0.999):
        self.comm = comm
        self.beta = np.array([beta_global, beta_local], float)
        if np.any(self.beta <= 0) or np.any(self.beta >= 1):
            raise ValueError("initial betas must lie in (0, 1)")
        if not 0 < target < 1:
            raise ValueError("target acceptance must lie in (0, 1)")
        if adapt_stop <= adapt_start or adapt_interval <= 0 or not 0.0 <= gain_exponent <= 1 or initial_gain <= 0 or gain_hold < 0 \
                or not 0 < beta_min < beta_max < 1:
            raise ValueError("invalid adaptation settings")
        self.target, self.start, self.stop, self.interval = float(target), int(adapt_start), int(adapt_stop), int(adapt_interval)
        self.gain0, self.gexp, self.bmin, self.bmax = float(initial_gain), float(gain_exponent), beta_min, beta_max
        self.hold = int(gain_hold)
        self.logit = np.log(self.beta) - np.log1p(-self.beta)
        self.w_att = np.zeros(2, np.int64); self.w_acc = np.zeros(2, np.int64)
        self.t_att = np.zeros(2, np.int64); self.t_acc = np.zeros(2, np.int64)
        self.count = np.zeros(2, np.int64)
        self.last = np.full(2, np.nan)
        self.frozen = False
        self.history = []                    # (iteration, beta_G, beta_L, acc_G, acc_L)

    def record(self, kind, accepted):
        self.w_att[kind] += 1; self.t_att[kind] += 1
        if accepted:
            self.w_acc[kind] += 1; self.t_acc[kind] += 1

    def should_adapt(self, iteration):
        return (not self.frozen and self.start <= iteration < self.stop
                and (iteration - self.start + 1) % self.interval == 0)

    def adapt(self, iteration):
        """Collective: every rank calls it at the same iteration.  Returns True if an update was made."""
        if iteration >= self.stop:
            self.frozen = True
        if not self.should_adapt(iteration):
            return False
        att = self.comm.allreduce(self.w_att.copy())
        acc = self.comm.allreduce(self.w_acc.copy())
        for k in (GLOBAL, LOCAL):
            if att[k] == 0:
                continue
            rate = acc[k] / att[k]
            self.count[k] += 1
            self.logit[k] += self.gain0 / max(1, self.count[k] - self.hold) ** self.gexp * (rate - self.target)
            b = float(np.clip(1.0 / (1.0 + np.exp(-self.logit[k])), self.bmin, self.bmax))
            self.beta[k] = b
            self.logit[k] = np.log(b) - np.log1p(-b)
            self.last[k] = rate
        self.history.append((int(iteration), float(self.beta[0]), float(self.beta[1]), float(self.last[0]), float(self.last[1])))
        self.w_att[:] = 0; self.w_acc[:] = 0
        return True

    def freeze(self):
        self.frozen = True

    def state(self):
        return dict(beta=self.beta.copy(), logit=self.logit.copy(), count=self.count.copy(), frozen=self.frozen,
                    history=list(self.history), t_att=self.t_att.copy(), t_acc=self.t_acc.copy())

    def set_state(self, s):
        self.beta, self.logit, self.count, self.frozen = s["beta"].copy(), s["logit"].copy(), s["count"].copy(), s["frozen"]
        self.history, self.t_att, self.t_acc = list(s["history"]), s["t_att"].copy(), s["t_acc"].copy()
