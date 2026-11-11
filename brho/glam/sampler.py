"""GLAM sampler: one chain per MPI rank, global / local pCN stages, exchanges through the anchor manager.

Per iteration `it` (same order as the group's driver):
  1. if exchanges are on (it <= anchor_stop_iteration) and the current stage has ended: collective exchange
     (archive rebuild, schedule advance, population restart through GlobalLocalAnchorManager.close_phase_and_restart_mpi);
     the chain continues from its restarted state.
  2. one pCN move.  GLOBAL stage: every theta block and w_rho move with beta_G.  LOCAL stage: only block mgr.local_subdomain(rank) (rank % n_sub, or rotating with local_block_schedule=round_robin)
     moves, with beta_L; rho does not move.  The prior (N(0, I) for theta and for w_rho) cancels in pCN, so the
     acceptance probability is min(1, exp(Phi_current - Phi_proposed)); a proposal with rho outside the PCHIP range has
     Phi = +inf (truncated prior) and costs no forward solve.
  3. two-step-size adaptation (window 1 only): Robbins-Monro on the logit scale, pooled over chains by an allreduce;
     both betas are frozen at it = window1.
  4. phase-best bookkeeping for the manager.
Local stages are switched off at window1 + window2 (manager option disable_local_after_iteration).

Output (repetition-encoded): a record (start, count, theta, w_rho, Phi) is written whenever the chain's state changes
by acceptance or by an exchange; the state at iteration t is the state after the move at t, so a record covers
iterations [start, start + count).  Parts are flushed every `save_every` records to out_dir/chains/rankRRR_partPPPP.npz
together with per-iteration traces.  Exchanges write a restartable snapshot (manager state + chain state).

NOTE (stated in every result): the exchange/restart steps depend on the whole population and do not preserve detailed
balance; adaptation is confined to window 1, which is excluded from the post-processing tail.
"""
import json
import os
import re
import pickle
import time

import numpy as np

from .anchors import BuiltField, GlobalLocalAnchorManager, Phase, copy_theta
from .stepsize import GLOBAL, LOCAL, TwoStepAdapter


class ChainWriter:
    def __init__(self, out_dir, rank, n_sub, save_every, part=0):
        self.dir = os.path.join(out_dir, "chains")
        os.makedirs(self.dir, exist_ok=True)
        self.rank, self.n_sub, self.save_every, self.part = rank, n_sub, save_every, part
        self.rec = dict(start=[], count=[], theta=[], w=[], phi=[])
        self.tr = dict(phi=[], acc=[], kind=[], bg=[], bl=[], rc=[], rd=[])
        self.tr_first = None

    def trace(self, it, phi, acc, kind, bg, bl, rc=np.nan, rd=np.nan):
        if self.tr_first is None:
            self.tr_first = it
        for k, v in zip(("phi", "acc", "kind", "bg", "bl", "rc", "rd"), (phi, acc, kind, bg, bl, rc, rd)):
            self.tr[k].append(v)

    def emit(self, start, count, state, phi):
        if count <= 0:
            return
        self.rec["start"].append(start); self.rec["count"].append(count)
        self.rec["theta"].append(np.concatenate([np.asarray(b, float).ravel() for b in state[:self.n_sub]]))
        self.rec["w"].append(float(state[self.n_sub][0])); self.rec["phi"].append(float(phi))
        if len(self.rec["start"]) >= self.save_every:
            self.flush()

    def flush(self):
        if not self.rec["start"] and not self.tr["phi"]:
            return
        n = len(self.rec["start"])
        np.savez(os.path.join(self.dir, f"rank{self.rank:03d}_part{self.part:04d}.npz"),
                 start=np.asarray(self.rec["start"], np.int64), count=np.asarray(self.rec["count"], np.int64),
                 theta=np.asarray(self.rec["theta"], float).reshape(n, -1), w=np.asarray(self.rec["w"], float),
                 phi=np.asarray(self.rec["phi"], float),
                 trace_it0=-1 if self.tr_first is None else self.tr_first,
                 trace_phi=np.asarray(self.tr["phi"], float), trace_acc=np.asarray(self.tr["acc"], np.int8),
                 trace_kind=np.asarray(self.tr["kind"], np.int8), trace_beta_g=np.asarray(self.tr["bg"], float),
                 trace_beta_l=np.asarray(self.tr["bl"], float),
                 trace_re_conc=np.asarray(self.tr["rc"], np.float32), trace_re_dp=np.asarray(self.tr["rd"], np.float32))
        self.part += 1
        self.rec = dict(start=[], count=[], theta=[], w=[], phi=[])
        self.tr = dict(phi=[], acc=[], kind=[], bg=[], bl=[], rc=[], rd=[])
        self.tr_first = None


def pcn_propose(theta, blocks, beta, rng):
    """pCN proposal xi' = sqrt(1 - beta^2) xi + beta zeta on the listed blocks; every other block is copied unchanged."""
    prop = [b.copy() for b in theta]
    root = np.sqrt(1.0 - beta * beta)
    for k in blocks:
        prop[k] = root * theta[k] + beta * rng.standard_normal(theta[k].shape)
    return prop


# GlamConfig fields passed one-to-one to the manager (same names); a negative value of the fields in _NONE_IF_NEGATIVE is None
_PASSTHROUGH = (
    "num_cycles", "archive_selection_mode", "covariance_regularization", "pressure_covariance_regularization",
    "permeability_logdet_weight", "pressure_logdet_weight", "normalize_logdet_by_dimension", "distance_score_weight",
    "quality_score_weight", "minimum_logdet_gain", "diversity_alpha", "logdet_gain_scale", "minimum_diversity_score",
    "archive_selection_temperature", "exploit_fraction", "frontier_fraction", "explore_fraction",
    "exploit_restart_scale", "frontier_restart_scale", "exploration_restart_scale", "frontier_target_phi",
    "frontier_minimum_distance", "successful_region_distance", "successful_region_merge_distance",
    "frontier_max_rounds", "frontier_minimum_improvement", "frontier_cooldown_rounds",
    "frontier_relative_quality_tolerance", "frontier_reseed_interval", "frontier_quality_weight",
    "frontier_distance_weight", "preserve_frontier_exact", "minimum_anchor_occupancy", "anchor_temperature",
    "uniform_allocation_weight", "equilibrium_allocation_enabled", "equilibrium_positive_fraction_threshold",
    "equilibrium_drift_tolerance", "equilibrium_min_observations", "equilibrium_dispersion_strength",
    "equilibrium_distance_bias_max", "preserve_exact_anchor", "max_exact_anchor_copies", "restart_phi_tolerance",
    "max_restart_attempts", "restart_scale_decay", "random_reach_attempts", "random_reach_initial_scale",
    "random_reach_decay", "duplicate_tolerance", "local_block_schedule",
)
_NONE_IF_NEGATIVE = frozenset((
    "minimum_logdet_gain", "frontier_target_phi", "frontier_minimum_distance", "successful_region_distance",
    "successful_region_merge_distance", "frontier_relative_quality_tolerance", "random_reach_initial_scale",
    "max_exact_anchor_copies",
))


def manager_kwargs(cfg):
    kw = {}
    for name in _PASSTHROUGH:
        v = getattr(cfg, name)
        kw[name] = None if (name in _NONE_IF_NEGATIVE and v < 0) else v
    return kw


def load_keep_mask(path, n_sub):
    """npz with boolean arrays m0 .. m<n_sub> (block b; True = the chain keeps its own coordinate at a restart) -> list of arrays."""
    z = np.load(path)
    return [np.asarray(z[f"m{b}"], dtype=bool) for b in range(n_sub + 1)]


def make_manager(cfg, n_sub, rank, start_iteration=0, n_chains=1, keep_mask=None, selection_constrained_only=False):
    max_anchors = None if cfg.max_anchors < 0 else (cfg.max_anchors or 2 * max(int(n_chains), 1))
    return GlobalLocalAnchorManager(
        numsubd=n_sub,
        max_anchors=max_anchors,
        global_stage_length=cfg.global_stage_length, local_stage_length=cfg.local_stage_length,
        quality_tolerance=cfg.quality_tolerance, minimum_anchor_distance=cfg.minimum_anchor_distance,
        restart_scale=cfg.restart_scale,
        initial_burnin_length=cfg.burnin_length,
        minimum_global_stage_length=cfg.min_global_stage_length,
        minimum_local_stage_length=cfg.min_local_stage_length,
        global_stage_decay=cfg.global_stage_decay, local_stage_decay=cfg.local_stage_decay,
        disable_local_after_iteration=cfg.window1 + cfg.window2,
        current_iteration_only=(cfg.anchor_current_only or not cfg.anchor_use_best_state),
        include_previous_anchors=cfg.anchor_include_previous,
        start_iteration=start_iteration,
        extra_state_blocks=1, extra_block_weights=[cfg.rho_distance_weight],
        seed=cfg.seed0 + 7919 * rank,
        keep_own_mask=keep_mask, selection_constrained_only=selection_constrained_only,
        **manager_kwargs(cfg),
    )


def _init_state(model, rng, rank, tries=200):
    for _ in range(tries):
        st = model.sample_state(rng, model.block_dims)
        phi, rc, rd = model.eval_state(st, rank)
        if np.isfinite(phi):
            return st, phi, (rc, rd)
    raise RuntimeError("no finite-potential initial state found")


def run_glam(model, cfg, comm, out_dir, resume=False, log=print, keep_mask=None, selection_constrained_only=False):
    rank, P = comm.Get_rank(), comm.Get_size()
    n_sub = model.n_sub
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng([cfg.seed0, rank])
    mgr = make_manager(cfg, n_sub, rank, n_chains=P, keep_mask=keep_mask, selection_constrained_only=selection_constrained_only)
    adapter = TwoStepAdapter(comm, cfg.beta_g0, cfg.beta_l0, cfg.target, adapt_start=0, adapt_stop=cfg.window1,
                             adapt_interval=cfg.adapt_interval, initial_gain=cfg.adapt_gain,
                             gain_exponent=cfg.adapt_gain_exponent, gain_hold=cfg.adapt_gain_hold)
    snap_dir = os.path.join(out_dir, "anchor_state")

    # --- manager callbacks (no forward solve for a rho outside the PCHIP range)
    def field_builder(th):
        return model.build(th) if model.rho_ok(th) else BuiltField(None, None)

    def simulation_fn(inputdata, field, r, ftype):
        return None if field is None else model.simulate(field, r)

    def phi_fn(th, payload, sim):
        return model.phi(th, payload, sim)

    # --- initial state or resume
    it0, writer_part = 0, 0
    it_s = None
    if resume and os.path.isdir(snap_dir):
        # newest iteration for which EVERY rank has its chain snapshot (a job killed mid-write, e.g. preempted, can
        # leave the newest generation incomplete); the anchor state of that generation was written before it
        mine = {int(m.group(1)) for fn in os.listdir(snap_dir)
                if (m := re.fullmatch(rf"chain_r{rank:03d}_ite(\d+)\.pkl", fn))}
        common = set.intersection(*comm.allgather(mine))
        if common:
            it_s = mgr.load_anchor_state(snap_dir, max_iteration=max(common), rank=rank, comm=comm)
        if it_s is not None and int(it_s) not in common:
            raise RuntimeError(f"rank {rank}: anchor state of iteration {it_s} has no chain snapshot on every rank")
    if it_s is not None:
        with open(os.path.join(snap_dir, f"chain_r{rank:03d}_ite{it_s}.pkl"), "rb") as f:
            cs = pickle.load(f)
        theta, phi_cur = cs["theta"], cs["phi"]
        re_cur = cs.get("re")
        if re_cur is None:                                   # snapshot of an older run: one extra solve
            _, rc, rd = model.eval_state(theta, rank)
            re_cur = (rc, rd)
        rng.bit_generator.state = cs["rng"]
        adapter.set_state(cs["adapter"])
        it0, writer_part, cur_start = cs["next_it"], cs["part"], cs["cur_start"]
        if rank == 0:
            log(f"[glam] resumed at iteration {it0} from snapshot {it_s}")
    else:
        theta, phi_cur, re_cur = _init_state(model, rng, rank)
        cur_start = 0
    writer = ChainWriter(out_dir, rank, n_sub, cfg.save_every, part=writer_part)
    if resume:                                 # drop parts written after the snapshot (they will be rewritten);
        for fn in os.listdir(writer.dir):      # with no snapshot at all, that is every part of an earlier attempt
            if fn.startswith(f"rank{rank:03d}_part") and int(fn[-8:-4]) >= writer_part:
                os.remove(os.path.join(writer.dir, fn))
    if mgr.phase_best_theta is None:
        mgr.initialize_phase_best(theta, phi_cur)

    exchanges = []
    n_acc_win = np.zeros(2); n_att_win = np.zeros(2)
    t0 = time.time()
    for it in range(it0, cfg.n_iter):
        anchors_on = cfg.anchor_stop_iteration < 0 or it <= cfg.anchor_stop_iteration
        # ---------------------------------------------------------------- 1. exchange
        if anchors_on and mgr.phase_has_ended(it):
            res, _ = mgr.close_phase_and_restart_mpi(
                current_theta=copy_theta(theta), current_phi=phi_cur, iteration=it, rank=rank,
                number_of_chains=P, comm=comm, field_builder=field_builder, simulation_fn=simulation_fn,
                phi_fn=phi_fn, inputdata=None, datafiletype=None, current_simuldata=None,
                state_directory=snap_dir, state_keep=3)
            writer.emit(cur_start, it - cur_start, theta, phi_cur)     # state through it-1
            cur_start = it
            if res is not None:
                theta, phi_cur = copy_theta(res.theta), float(res.phi)
                _, rc, rd = model.eval_state(theta, rank)      # relative errors of the restart state (one solve)
                re_cur = (rc, rd)
            exchanges.append(dict(iteration=it, phase=mgr.phase.name, cycle=mgr.cycle,
                                  archive=dict(mgr.last_archive_diagnostics),
                                  restart=dict(mgr.last_restart_diagnostics)))
            writer.flush()
            with open(os.path.join(snap_dir, f"chain_r{rank:03d}_ite{it}.pkl"), "wb") as f:
                pickle.dump(dict(theta=theta, phi=phi_cur, re=re_cur, rng=rng.bit_generator.state, adapter=adapter.state(),
                                 next_it=it, part=writer.part, cur_start=cur_start), f)
            comm.Barrier()
        # ---------------------------------------------------------------- 2. one pCN move
        phase = mgr.phase
        local = phase is Phase.LOCAL and anchors_on
        kind = LOCAL if local else GLOBAL
        blocks = [mgr.local_subdomain(rank)] if local else list(range(n_sub + 1))
        beta = float(adapter.beta[kind])
        prop = pcn_propose(theta, blocks, beta, rng)
        if model.rho_ok(prop):
            fld = model.build(prop)
            sim = model.simulate(fld, rank)
            phi_p, rc_p, rd_p = model.phi_parts(prop, fld, sim)
        else:
            phi_p, rc_p, rd_p = np.inf, np.nan, np.nan
        accepted = bool(np.isfinite(phi_p) and np.log(rng.random()) < phi_cur - phi_p)
        if accepted:
            writer.emit(cur_start, it - cur_start, theta, phi_cur)
            theta, phi_cur, cur_start = prop, float(phi_p), it
            re_cur = (rc_p, rd_p)
        # ---------------------------------------------------------------- 3. adaptation (window 1)
        adapter.record(kind, accepted)
        n_att_win[kind] += 1; n_acc_win[kind] += accepted
        adapter.adapt(it)                                 # collective; no-op outside window 1
        if it + 1 >= cfg.window1:
            if not adapter.frozen and rank == 0:
                g, l = adapter.last
                bad = [f"{n} acc {a:.2f}" for n, a in (("global", g), ("local", l))
                       if np.isfinite(a) and not 0.5 * cfg.target <= a <= 1.5 * cfg.target]
                log(f"[glam] step sizes frozen at it {it + 1}: beta_G {adapter.beta[0]:.4f} beta_L {adapter.beta[1]:.4f}, "
                    f"last window acceptance G {g:.2f} L {l:.2f} (target {cfg.target})"
                    + (f"  WARNING: far from target ({', '.join(bad)}); adaptation was too short or too slow "
                       f"(raise window1 / adapt_gain / adapt_gain_hold, lower beta_g0)" if bad else ""))
            adapter.freeze()
        # ---------------------------------------------------------------- 4. bookkeeping
        mgr.record_phase_best(copy_theta(theta), phi_cur)
        writer.trace(it, phi_cur, int(accepted), kind, adapter.beta[0], adapter.beta[1], re_cur[0], re_cur[1])
        if rank == 0 and cfg.log_every and (it + 1) % cfg.log_every == 0:
            acc = n_acc_win / np.maximum(n_att_win, 1)
            log(f"[glam] it {it + 1}/{cfg.n_iter} phase {phase.name} cycle {mgr.cycle} Phi {phi_cur:.3f} "
                f"beta_G {adapter.beta[0]:.4f} beta_L {adapter.beta[1]:.4f} acc(G,L) {acc[0]:.2f},{acc[1]:.2f} "
                f"rho {model.rho_of(theta[n_sub][0]):.3f} {time.time() - t0:.0f}s")
            n_acc_win[:] = 0; n_att_win[:] = 0
    writer.emit(cur_start, cfg.n_iter - cur_start, theta, phi_cur)
    writer.flush()
    comm.Barrier()
    if rank == 0:
        with open(os.path.join(out_dir, "exchanges.pkl"), "wb") as f:
            pickle.dump(exchanges, f)
        with open(os.path.join(out_dir, "adaptation.json"), "w") as f:
            json.dump(dict(history=adapter.history, beta=adapter.beta.tolist(), count=adapter.count.tolist()), f)
    return dict(theta=theta, phi=phi_cur, beta=adapter.beta.copy(), adapter=adapter, exchanges=exchanges,
                n_exchanges=len(exchanges), manager=mgr)
