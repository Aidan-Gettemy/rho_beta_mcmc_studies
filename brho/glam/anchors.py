"""Standalone quality-diversity Global--Local restart support for GLAM.

Prototype extension: persistent protected successful-region anchors accumulate
frontier successes across archive rebuilds, while overlapping successful regions
are dynamically merged so that the successful set tracks distinct posterior
basins rather than growing monotonically. This rollback-optimized revision preserves the original conservative MPI restart
synchronization while retaining the safe numerical optimizations. simuldata remains
out of persistent anchors but is reconstructed on every restart evaluation.
This competitive-frontier extension retires dominated/stale campaigns and
periodically re-competes active frontiers against unresolved alternatives.

User-supplied callbacks
-----------------------
field_builder(theta) -> field
simulation_fn(inputdata, field, rank, datafiletype) -> simuldata
phi_fn(theta, field_payload, simuldata) -> absolute archive potential

The state ``theta`` is a list of NumPy arrays.  The first ``numsubd`` entries
are interpreted as subdomain coefficients.  Additional entries are copied and
carried with the state, but are not perturbed by the default restart logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Sequence
import copy
import csv
import os
import pickle
import re
import time
from pathlib import Path


import numpy as np
from numpy.typing import NDArray

idtype = np.int64


Array = NDArray[np.floating]
Theta = list[Array]
FieldBuilder = Callable[[Theta], Any]
SimulationFunction = Callable[[Any, Any, int, Any], Any]
PhiFunction = Callable[[Theta, Any, Any], float]


def copy_theta(theta: Sequence[np.ndarray]) -> Theta:
    """Deep-copy a list-of-arrays parameter state."""
    return [np.array(component, copy=True) for component in theta]

def flatten_theta(theta, numsubd, weights=None):
    """Flatten the first numsubd arrays into one vector (block b multiplied by weights[b] if given)."""
    return np.concatenate(
        [
            (1.0 if weights is None else float(weights[subd]))
            * np.asarray(theta[subd], dtype=float).ravel()
            for subd in range(numsubd)
        ]
    )


def copy_simuldata(simuldata: Any) -> Any:
    """Deep-copy retained pressure/simulation data."""
    return copy.deepcopy(simuldata)


def flatten_simuldata(simuldata: Any) -> np.ndarray:
    """Flatten pressure/simulation data into one numeric vector."""
    if simuldata is None:
        raise ValueError(
            "simuldata is required for joint permeability-pressure selection."
        )
    if isinstance(simuldata, dict):
        pieces = [flatten_simuldata(simuldata[key]) for key in sorted(simuldata)]
        return np.concatenate(pieces) if pieces else np.empty(0, dtype=float)
    if isinstance(simuldata, (list, tuple)):
        pieces = [flatten_simuldata(value) for value in simuldata]
        return np.concatenate(pieces) if pieces else np.empty(0, dtype=float)
    return np.asarray(simuldata.data[0], dtype=float).ravel()

def anchor_distance_statistics(
    anchors,
    numsubd,
    weights=None,
    permeability_regularization=1.0e-6,
    pressure_regularization=1.0e-6,
):
    """Compute theta-space distance/covariance diagnostics using a small Gram matrix."""
    if not anchors:
        raise ValueError("anchors must contain at least one anchor.")

    X = np.vstack([flatten_theta(anchor.theta, numsubd, weights) for anchor in anchors])
    n, d = X.shape
    eps = float(permeability_regularization)

    if n == 1:
        logdet = float(d * np.log(eps))
        return {
            "distance_min": 0.0, "distance_mean": 0.0,
            "distance_median": 0.0, "distance_max": 0.0,
            "anchor_cov_trace": 0.0,
            "anchor_cov_largest_eigenvalue": 0.0,
            "anchor_cov_largest_eigenvalue_fraction": 0.0,
            "anchor_effective_rank": 0.0,
            "logdet_Ck": logdet,
            "logdet_Ck_per_dimension": logdet / max(d, 1),
            "nearest_neighbor_distances": np.array([0.0]),
        }

    norms = np.sum(X * X, axis=1)
    distance_sq = norms[:, None] + norms[None, :] - 2.0 * X @ X.T
    distance_sq = np.maximum(distance_sq, 0.0)
    distance_matrix = np.sqrt(distance_sq)
    upper = distance_matrix[np.triu_indices(n, k=1)]
    nearest_matrix = distance_matrix.copy()
    np.fill_diagonal(nearest_matrix, np.inf)
    nearest = np.min(nearest_matrix, axis=1)

    Xc = X - np.mean(X, axis=0, keepdims=True)
    gram_cov = (Xc @ Xc.T) / (n - 1.0)
    gram_cov = 0.5 * (gram_cov + gram_cov.T)
    eigs = np.clip(np.linalg.eigvalsh(gram_cov), 0.0, None)
    trace = float(np.sum(eigs))
    largest = float(np.max(eigs)) if eigs.size else 0.0
    positive = eigs[eigs > 0.0]
    if trace > 0.0 and positive.size:
        p = positive / trace
        effective_rank = float(np.exp(-np.sum(p * np.log(p))))
    else:
        effective_rank = 0.0

    sign, small_logdet = np.linalg.slogdet(np.eye(n) + gram_cov / eps)
    logdet = float(d * np.log(eps) + small_logdet) if sign > 0 else -np.inf
    return {
        "distance_min": float(np.min(upper)),
        "distance_mean": float(np.mean(upper)),
        "distance_median": float(np.median(upper)),
        "distance_max": float(np.max(upper)),
        "anchor_cov_trace": trace,
        "anchor_cov_largest_eigenvalue": largest,
        "anchor_cov_largest_eigenvalue_fraction": (largest / trace if trace > 0.0 else 0.0),
        "anchor_effective_rank": effective_rank,
        "logdet_Ck": float(logdet),
        "logdet_Ck_per_dimension": float(logdet / max(d, 1)),
        "nearest_neighbor_distances": nearest,
    }

def anchor_occupancy_statistics(allocation, number_of_anchors):
    counts = np.bincount(
        np.asarray(allocation, dtype=int),
        minlength=number_of_anchors,
    )

    total = np.sum(counts)

    if total == 0:
        return {
            "counts": counts,
            "effective_occupied_anchors": 0.0,
            "normalized_occupancy_entropy": 0.0,
            "minimum_occupancy": 0,
            "maximum_occupancy": 0,
        }

    probabilities = counts[counts > 0] / total
    entropy = -np.sum(
        probabilities * np.log(probabilities)
    )

    effective = float(np.exp(entropy))

    if number_of_anchors > 1:
        normalized_entropy = float(
            entropy / np.log(number_of_anchors)
        )
    else:
        normalized_entropy = 0.0

    return {
        "counts": counts,
        "effective_occupied_anchors": effective,
        "normalized_occupancy_entropy": normalized_entropy,
        "minimum_occupancy": int(np.min(counts)),
        "maximum_occupancy": int(np.max(counts)),
    }


def append_csv_row(filename, row):
    path = Path(filename)
    write_header = not path.exists()

    with path.open("a", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(row.keys()),
        )

        if write_header:
            writer.writeheader()

        writer.writerow(row)


def theta_distance_squared(
    theta_a: Sequence[np.ndarray],
    theta_b: Sequence[np.ndarray],
    numsubd: int,
    weights=None,
) -> float:
    """Squared Euclidean distance over the first ``numsubd`` components (block b weighted by weights[b])."""
    if len(theta_a) < numsubd or len(theta_b) < numsubd:
        raise ValueError("theta contains fewer components than numsubd.")

    total = 0.0
    for subd in range(numsubd):
        difference = (
            np.asarray(theta_a[subd], dtype=float)
            - np.asarray(theta_b[subd], dtype=float)
        )
        w2 = 1.0 if weights is None else float(weights[subd]) ** 2
        total += w2 * float(np.dot(difference.ravel(), difference.ravel()))
    return total


def covariance_effective_rank_from_vectors(
    vectors: np.ndarray, *, tolerance: float = 1.0e-12,
) -> float:
    """Entropy-based effective rank via the sample-space Gram covariance."""
    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim != 2:
        raise ValueError("vectors must be two-dimensional.")
    n = vectors.shape[0]
    if n <= 1:
        return 0.0
    centered = vectors - np.mean(vectors, axis=0, keepdims=True)
    gram_cov = (centered @ centered.T) / (n - 1.0)
    gram_cov = 0.5 * (gram_cov + gram_cov.T)
    eigenvalues = np.clip(np.linalg.eigvalsh(gram_cov), 0.0, None)
    positive = eigenvalues[eigenvalues > tolerance]
    if positive.size == 0:
        return 0.0
    probabilities = positive / np.sum(positive)
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def covariance_logdet_from_vectors(
    vectors: np.ndarray, *, regularization: float,
) -> float:
    """Regularized covariance logdet via matrix determinant lemma (n x n, not d x d)."""
    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim != 2:
        raise ValueError("vectors must be two-dimensional.")
    n, d = vectors.shape
    eps = float(regularization)
    if eps <= 0.0:
        raise ValueError("regularization must be positive.")
    if n <= 1:
        return float(d * np.log(eps))
    centered = vectors - np.mean(vectors, axis=0, keepdims=True)
    gram_cov = (centered @ centered.T) / (n - 1.0)
    gram_cov = 0.5 * (gram_cov + gram_cov.T)
    sign, small_logdet = np.linalg.slogdet(np.eye(n) + gram_cov / eps)
    if sign <= 0:
        return -np.inf
    return float(d * np.log(eps) + small_logdet)


class Phase(Enum):
    BURN_IN = auto()
    GLOBAL = auto()
    LOCAL = auto()
    FINISHED = auto()


@dataclass
class BuiltField:
    """
    Field payload returned by ``field_builder``.

    ``simulation_field`` is passed to ``simulation_fn``. ``prior_field`` is the
    permeability/log-permeability representation used by the absolute archive
    potential, for example when projecting into global-prior coordinates.
    """
    simulation_field: Any
    prior_field: Any


@dataclass
class Anchor:
    theta: Theta
    phi: float
    source: str = "unknown"
    source_rank: int = -1
    phase_index: int = -1
    anchor_id: int = -1
    parent_anchor_id: int = -1
    source_role: str = "unassigned"
    exploration_visits: int = 0

    def clone(self) -> "Anchor":
        return Anchor(
            theta=copy_theta(self.theta),
            phi=float(self.phi),
            source=str(self.source),
            source_rank=int(self.source_rank),
            phase_index=int(self.phase_index),
            anchor_id=int(self.anchor_id),
            parent_anchor_id=int(self.parent_anchor_id),
            source_role=str(self.source_role),
            exploration_visits=int(self.exploration_visits),
        )


@dataclass
class RestartResult:
    theta: Theta
    field: Any
    simuldata: Any
    phi: float
    anchor_index: int
    anchor_phi: float
    exact_anchor_copy: bool
    accepted_under_threshold: bool
    attempts: int
    final_scale: float
    role: str = "unassigned"
    source_anchor_id: int = -1


class GlobalLocalAnchorManager:
    """Quality-diversity manager designed to splice into an existing GLAM driver."""

    def __init__(
        self,
        *,
        numsubd: int,
        global_stage_length: int,
        local_stage_length: int,
        num_cycles: int,
        quality_tolerance: float,
        minimum_anchor_distance: float,
        restart_scale: float,
        max_anchors: int | None = None,
        archive_selection_mode: str = "diversity",
        covariance_regularization: float = 1.0e-6,
        pressure_covariance_regularization: float = 1.0e-6,
        permeability_logdet_weight: float = 0.50,
        pressure_logdet_weight: float = 0.50,
        normalize_logdet_by_dimension: bool = True,
        distance_score_weight: float = 0.02,
        quality_score_weight: float = 0.50,
        minimum_logdet_gain: float | None = None,
        # New stochastic diversity archive controls.
        diversity_alpha: float = 0.70,
        logdet_gain_scale: float = 1.0,
        minimum_diversity_score: float = 0.20,
        archive_selection_temperature: float = 0.10,
        # Exploit-frontier-explore allocation fractions.
        exploit_fraction: float = 0.25,
        frontier_fraction: float = 0.40,
        explore_fraction: float = 0.35,
        # Role-specific restart radii.
        exploit_restart_scale: float = 0.20,
        frontier_restart_scale: float = 0.35,
        exploration_restart_scale: float = 0.35,
        # Frontier campaign controls.
        frontier_target_phi: float | None = None,
        frontier_minimum_distance: float | None = None,
        successful_region_distance: float | None = None,
        successful_region_merge_distance: float | None = None,
        frontier_max_rounds: int = 3,
        frontier_minimum_improvement: float = 0.0,
        frontier_cooldown_rounds: int = 3,
        # Competitive-frontier controls.
        # A frontier must remain reasonably competitive with the current best
        # archive state and is periodically re-competed against unresolved
        # alternatives.
        frontier_relative_quality_tolerance: float | None = None,
        frontier_reseed_interval: int = 0,
        frontier_quality_weight: float = 0.60,
        frontier_distance_weight: float = 0.40,
        preserve_frontier_exact: bool = True,
        relative_error_fn: Callable[[Theta], float] | None = None,
        disable_local_after_iteration: int | None = None,
        initial_burnin_length: int = 0,
        minimum_global_stage_length: int | None = None,
        minimum_local_stage_length: int | None = None,
        global_stage_decay: float = 1.0,
        local_stage_decay: float = 1.0,
        restart_basis: Sequence[np.ndarray] | None = None,
        minimum_anchor_occupancy: int = 1,
        anchor_temperature: float = 1.0,
        uniform_allocation_weight: float = 0.30,
        # Equilibrium-aware allocation controls.
        equilibrium_allocation_enabled: bool = True,
        equilibrium_positive_fraction_threshold: float = 0.60,
        equilibrium_drift_tolerance: float = 0.0,
        equilibrium_min_observations: int = 8,
        equilibrium_dispersion_strength: float = 0.75,
        equilibrium_distance_bias_max: float = 0.85,
        preserve_exact_anchor: bool = True,
        max_exact_anchor_copies: int | None = None,
        restart_phi_tolerance: float = 5.0,
        max_restart_attempts: int = 3,
        restart_scale_decay: float = 0.8,
        random_reach_attempts: int = 10,
        random_reach_initial_scale: float | None = None,
        random_reach_decay: float = 0.95,
        start_iteration: int = 0,
        duplicate_tolerance: float = 1.0e-12,
        current_iteration_only: bool = False,
        seed: int | None = None,
        # --- brho extension: extra state blocks (rho) that belong to the full state ---
        extra_state_blocks: int = 0,
        extra_block_weights: Sequence[float] | None = None,
    ) -> None:
        """
        Parameters added for the open-archive schedule
        ------------------------------------------------
        max_anchors:
            Optional safety cap.  Use ``None`` for a genuinely open archive.
            In that mode, every quality-qualified candidate that is at least
            ``minimum_anchor_distance`` from the selected archive is retained.

        initial_burnin_length:
            Number of initial GLOBAL iterations performed before the first
            archive rebuild and population restart.

        minimum_*_stage_length and *_stage_decay:
            The phase length at cycle ``c`` is

                max(minimum_length, round(initial_length * decay**c)).

            Values below one make exchanges progressively more frequent.

        current_iteration_only:
            If False (default) the candidate pool of an archive rebuild is the
            previous archive, every rank's current state, and every rank's
            best state of the stage that just ended (``phase_best``). If True,
            the ``phase_best`` candidates are not submitted, so the new
            candidates come only from the chains' states at the exchange
            iteration. The previous archive, the persistent best-ever anchor
            and the successful/frontier bookkeeping are unchanged.

        max_exact_anchor_copies:
            Maximum number of active anchors that receive an exact copy at a
            restart.  ``None`` preserves one exact copy per active anchor,
            matching the earlier behavior.
        """
        if numsubd <= 0:
            raise ValueError("numsubd must be positive.")
        if global_stage_length <= 0 or local_stage_length <= 0:
            raise ValueError("stage lengths must be positive.")
        if num_cycles <= 0:
            raise ValueError("num_cycles must be positive.")
        if max_anchors is not None and max_anchors <= 0:
            raise ValueError("max_anchors must be positive or None.")
        if archive_selection_mode not in {
            "logdet", "farthest", "diversity"
        }:
            raise ValueError(
                "archive_selection_mode must be 'diversity', 'logdet', or 'farthest'."
            )
        if covariance_regularization <= 0.0:
            raise ValueError("covariance_regularization must be positive.")
        if pressure_covariance_regularization <= 0.0:
            raise ValueError("pressure_covariance_regularization must be positive.")
        if permeability_logdet_weight < 0.0 or pressure_logdet_weight < 0.0:
            raise ValueError("log-determinant weights must be nonnegative.")
        if permeability_logdet_weight + pressure_logdet_weight <= 0.0:
            raise ValueError("At least one log-determinant weight must be positive.")
        if distance_score_weight < 0.0:
            raise ValueError("distance_score_weight must be nonnegative.")
        if quality_score_weight < 0.0:
            raise ValueError("quality_score_weight must be nonnegative.")
        if disable_local_after_iteration is not None and disable_local_after_iteration < start_iteration:
            raise ValueError("disable_local_after_iteration cannot precede start_iteration.")
        if minimum_logdet_gain is not None and not np.isfinite(minimum_logdet_gain):
            raise ValueError("minimum_logdet_gain must be finite or None.")
        if not 0.0 <= diversity_alpha <= 1.0:
            raise ValueError("diversity_alpha must lie in [0, 1].")
        if logdet_gain_scale <= 0.0:
            raise ValueError("logdet_gain_scale must be positive.")
        if not 0.0 <= minimum_diversity_score <= 1.0:
            raise ValueError("minimum_diversity_score must lie in [0, 1].")
        if archive_selection_temperature <= 0.0:
            raise ValueError("archive_selection_temperature must be positive.")
        fractions = np.array(
            [exploit_fraction, frontier_fraction, explore_fraction], dtype=float
        )
        if np.any(fractions < 0.0):
            raise ValueError("Allocation fractions must be nonnegative.")
        if not np.isclose(np.sum(fractions), 1.0, atol=1.0e-10):
            raise ValueError(
                "exploit_fraction + frontier_fraction + explore_fraction must equal 1."
            )
        if exploit_restart_scale < 0.0 or frontier_restart_scale < 0.0 or exploration_restart_scale < 0.0:
            raise ValueError("Role-specific restart scales must be nonnegative.")
        if successful_region_distance is not None and successful_region_distance < minimum_anchor_distance:
            raise ValueError(
                "successful_region_distance must be at least minimum_anchor_distance."
            )
        if successful_region_merge_distance is not None:
            if successful_region_merge_distance < minimum_anchor_distance:
                raise ValueError(
                    "successful_region_merge_distance must be at least "
                    "minimum_anchor_distance so merged representatives cannot "
                    "violate the archive hard-distance invariant."
                )
            if (
                successful_region_distance is not None
                and successful_region_merge_distance > successful_region_distance
            ):
                raise ValueError(
                    "successful_region_merge_distance must not exceed "
                    "successful_region_distance. Use a smaller merge distance "
                    "than creation distance to provide hysteresis."
                )
        if frontier_max_rounds <= 0:
            raise ValueError("frontier_max_rounds must be positive.")
        if frontier_minimum_improvement < 0.0:
            raise ValueError("frontier_minimum_improvement must be nonnegative.")
        if frontier_cooldown_rounds < 0:
            raise ValueError("frontier_cooldown_rounds must be nonnegative.")
        if (
            frontier_relative_quality_tolerance is not None
            and frontier_relative_quality_tolerance < 0.0
        ):
            raise ValueError(
                "frontier_relative_quality_tolerance must be nonnegative or None."
            )
        if frontier_reseed_interval < 0:
            raise ValueError("frontier_reseed_interval must be nonnegative.")
        if frontier_quality_weight < 0.0 or frontier_distance_weight < 0.0:
            raise ValueError("frontier competition weights must be nonnegative.")
        if frontier_quality_weight + frontier_distance_weight <= 0.0:
            raise ValueError(
                "At least one frontier competition weight must be positive."
            )
        if initial_burnin_length < 0:
            raise ValueError("initial_burnin_length must be nonnegative.")
        if quality_tolerance < 0.0:
            raise ValueError("quality_tolerance must be nonnegative.")
        if minimum_anchor_distance < 0.0:
            raise ValueError("minimum_anchor_distance must be nonnegative.")
        if restart_scale < 0.0:
            raise ValueError("restart_scale must be nonnegative.")
        if anchor_temperature <= 0.0:
            raise ValueError("anchor_temperature must be positive.")
        if not 0.0 <= uniform_allocation_weight <= 1.0:
            raise ValueError("uniform_allocation_weight must lie in [0, 1].")
        if not 0.0 <= equilibrium_positive_fraction_threshold < 1.0:
            raise ValueError("equilibrium_positive_fraction_threshold must lie in [0, 1).")
        if equilibrium_drift_tolerance < 0.0:
            raise ValueError("equilibrium_drift_tolerance must be nonnegative.")
        if equilibrium_min_observations <= 0:
            raise ValueError("equilibrium_min_observations must be positive.")
        if not 0.0 <= equilibrium_dispersion_strength <= 1.0:
            raise ValueError("equilibrium_dispersion_strength must lie in [0, 1].")
        if not 0.0 <= equilibrium_distance_bias_max <= 1.0:
            raise ValueError("equilibrium_distance_bias_max must lie in [0, 1].")
        if minimum_anchor_occupancy < 0:
            raise ValueError("minimum_anchor_occupancy must be nonnegative.")
        if max_exact_anchor_copies is not None and max_exact_anchor_copies < 0:
            raise ValueError("max_exact_anchor_copies must be nonnegative or None.")
        if max_restart_attempts <= 0:
            raise ValueError("max_restart_attempts must be positive.")
        if random_reach_attempts < 0:
            raise ValueError("random_reach_attempts must be nonnegative.")
        if not 0.0 < restart_scale_decay <= 1.0:
            raise ValueError("restart_scale_decay must lie in (0, 1].")
        if not 0.0 < random_reach_decay <= 1.0:
            raise ValueError("random_reach_decay must lie in (0, 1].")
        if not 0.0 < global_stage_decay <= 1.0:
            raise ValueError("global_stage_decay must lie in (0, 1].")
        if not 0.0 < local_stage_decay <= 1.0:
            raise ValueError("local_stage_decay must lie in (0, 1].")

        self.numsubd = int(numsubd)
        # Full state = numsubd subdomain blocks followed by `extra_state_blocks` extra blocks (the rho
        # coordinate). Distances, log-determinants and restart perturbations use the full state (extra
        # blocks scaled by extra_block_weights); local stages still move one of the numsubd blocks only.
        self.extra_state_blocks = int(extra_state_blocks)
        self.nstate = self.numsubd + self.extra_state_blocks
        w = [1.0] * self.extra_state_blocks if extra_block_weights is None else list(extra_block_weights)
        if len(w) != self.extra_state_blocks:
            raise ValueError("extra_block_weights must have one entry per extra state block.")
        self.block_weights = [1.0] * self.numsubd + [float(x) for x in w]
        self.global_stage_length = int(global_stage_length)
        self.local_stage_length = int(local_stage_length)
        self.num_cycles = int(num_cycles)

        self.minimum_global_stage_length = int(
            global_stage_length
            if minimum_global_stage_length is None
            else minimum_global_stage_length
        )
        self.minimum_local_stage_length = int(
            local_stage_length
            if minimum_local_stage_length is None
            else minimum_local_stage_length
        )
        if self.minimum_global_stage_length <= 0:
            raise ValueError("minimum_global_stage_length must be positive.")
        if self.minimum_local_stage_length <= 0:
            raise ValueError("minimum_local_stage_length must be positive.")

        self.global_stage_decay = float(global_stage_decay)
        self.local_stage_decay = float(local_stage_decay)
        self.initial_burnin_length = int(initial_burnin_length)
        self.disable_local_after_iteration = (
            None if disable_local_after_iteration is None
            else int(disable_local_after_iteration)
        )

        self.max_anchors = None if max_anchors is None else int(max_anchors)
        self.archive_selection_mode = str(archive_selection_mode)
        self.covariance_regularization = float(covariance_regularization)
        self.pressure_covariance_regularization = float(
            pressure_covariance_regularization
        )
        total_weight = float(permeability_logdet_weight + pressure_logdet_weight)
        self.permeability_logdet_weight = float(permeability_logdet_weight) / total_weight
        self.pressure_logdet_weight = float(pressure_logdet_weight) / total_weight
        self.normalize_logdet_by_dimension = bool(normalize_logdet_by_dimension)
        self.distance_score_weight = float(distance_score_weight)
        self.quality_score_weight = float(quality_score_weight)
        self.minimum_logdet_gain = (
            None if minimum_logdet_gain is None else float(minimum_logdet_gain)
        )
        self.diversity_alpha = float(diversity_alpha)
        self.logdet_gain_scale = float(logdet_gain_scale)
        self.minimum_diversity_score = float(minimum_diversity_score)
        self.archive_selection_temperature = float(archive_selection_temperature)

        self.exploit_fraction = float(exploit_fraction)
        self.frontier_fraction = float(frontier_fraction)
        self.explore_fraction = float(explore_fraction)
        self.exploit_restart_scale = float(exploit_restart_scale)
        self.frontier_restart_scale = float(frontier_restart_scale)
        self.exploration_restart_scale = float(exploration_restart_scale)

        self.frontier_target_phi = (
            None if frontier_target_phi is None else float(frontier_target_phi)
        )
        self.frontier_minimum_distance = float(
            minimum_anchor_distance
            if frontier_minimum_distance is None
            else frontier_minimum_distance
        )
        self.successful_region_distance = float(
            max(self.frontier_minimum_distance, minimum_anchor_distance)
            if successful_region_distance is None
            else successful_region_distance
        )

        # Creation and merging use two different radii.  The neutral band
        #
        #     merge_distance < d < creation_distance
        #
        # supplies hysteresis: two regions are not repeatedly merged and then
        # immediately re-created by small representative motion.
        if successful_region_merge_distance is None:
            self.successful_region_merge_distance = float(
                max(
                    minimum_anchor_distance,
                    0.75 * self.successful_region_distance,
                )
            )
        else:
            self.successful_region_merge_distance = float(
                successful_region_merge_distance
            )

        if self.successful_region_merge_distance > self.successful_region_distance:
            raise ValueError(
                "successful_region_merge_distance must not exceed "
                "successful_region_distance."
            )

        self.frontier_max_rounds = int(frontier_max_rounds)
        self.frontier_minimum_improvement = float(frontier_minimum_improvement)
        self.frontier_cooldown_rounds = int(frontier_cooldown_rounds)
        self.frontier_relative_quality_tolerance = (
            None
            if frontier_relative_quality_tolerance is None
            else float(frontier_relative_quality_tolerance)
        )
        self.frontier_reseed_interval = int(frontier_reseed_interval)
        frontier_weight_total = float(
            frontier_quality_weight + frontier_distance_weight
        )
        self.frontier_quality_weight = (
            float(frontier_quality_weight) / frontier_weight_total
        )
        self.frontier_distance_weight = (
            float(frontier_distance_weight) / frontier_weight_total
        )
        self.preserve_frontier_exact = bool(preserve_frontier_exact)
        self.relative_error_fn = relative_error_fn

        self.quality_tolerance = float(quality_tolerance)
        self.current_iteration_only = bool(current_iteration_only)
        self.minimum_anchor_distance = float(minimum_anchor_distance)
        self.duplicate_tolerance = float(duplicate_tolerance)

        self.restart_scale = float(restart_scale)
        self.restart_basis = (
            None
            if restart_basis is None
            else [np.asarray(matrix, dtype=float) for matrix in restart_basis]
        )
        if self.restart_basis is not None and len(self.restart_basis) != self.numsubd:
            raise ValueError("restart_basis must have one matrix per subdomain.")

        self.minimum_anchor_occupancy = int(minimum_anchor_occupancy)
        self.anchor_temperature = float(anchor_temperature)
        self.uniform_allocation_weight = float(uniform_allocation_weight)
        self.equilibrium_allocation_enabled = bool(equilibrium_allocation_enabled)
        self.equilibrium_positive_fraction_threshold = float(equilibrium_positive_fraction_threshold)
        self.equilibrium_drift_tolerance = float(equilibrium_drift_tolerance)
        self.equilibrium_min_observations = int(equilibrium_min_observations)
        self.equilibrium_dispersion_strength = float(equilibrium_dispersion_strength)
        self.equilibrium_distance_bias_max = float(equilibrium_distance_bias_max)
        self.current_restart_phi = np.nan
        self.current_restart_anchor_phi = np.nan
        self.last_equilibrium_diagnostics: dict[str, Any] = {
            "equilibrium_observation_count": 0,
            "equilibrium_positive_drift_fraction": np.nan,
            "equilibrium_median_drift": np.nan,
            "equilibrium_median_end_phi": np.nan,
            "equilibrium_median_restart_phi": np.nan,
            "equilibrium_pressure": 0.0,
        }
        self.preserve_exact_anchor = bool(preserve_exact_anchor)
        self.max_exact_anchor_copies = (
            None
            if max_exact_anchor_copies is None
            else int(max_exact_anchor_copies)
        )

        self.restart_phi_tolerance = float(restart_phi_tolerance)
        self.max_restart_attempts = int(max_restart_attempts)
        self.restart_scale_decay = float(restart_scale_decay)

        self.random_reach_attempts = int(random_reach_attempts)
        self.random_reach_initial_scale = float(
            restart_scale if random_reach_initial_scale is None
            else random_reach_initial_scale
        )
        self.random_reach_decay = float(random_reach_decay)

        self.rng = np.random.default_rng(seed)

        self.phase_index = 0
        self.cycle = 0
        self.phase_start_iteration = int(start_iteration)

        if self.initial_burnin_length > 0:
            self.phase = Phase.BURN_IN
            self.phase_stop_iteration = (
                self.phase_start_iteration + self.initial_burnin_length
            )
        else:
            self.phase = Phase.GLOBAL
            self.phase_stop_iteration = (
                self.phase_start_iteration
                + self.phase_length_for(Phase.GLOBAL, self.cycle)
            )

        self.anchors: list[Anchor] = []
        self.best_anchor: Anchor | None = None
        self.phase_best_theta: Theta | None = None
        self.phase_best_phi = np.inf
        self.phase_best_role = "initial"
        self.phase_best_parent_anchor_id = -1

        # Persistent identities allow frontier campaigns to follow descendants
        # across archive rebuilds. Only rank 0 creates new IDs; broadcasts keep
        # the other ranks synchronized.
        self._next_anchor_id = 0
        self.current_chain_role = "initial"
        self.current_source_anchor_id = -1

        # Frontier campaign state.
        self.frontier_anchor_id: int | None = None
        self.frontier_start_phi: float | None = None
        self.frontier_previous_phi: float | None = None
        self.frontier_best_phi: float | None = None
        self.frontier_rounds = 0
        self.frontier_status = "NONE"
        self.frontier_cooldown_until: dict[int, int] = {}

        # Persistent protected representatives of distinct regions that have
        # reached frontier_target_phi. Their anchor_id is the stable region ID.
        # These representatives bypass the moving hard quality gate on future
        # archive rebuilds, but they still count against max_anchors.
        self.successful_anchors: list[Anchor] = []
        self.successful_region_ids: list[int] = []
        self._successful_source_ids_this_round: set[int] = set()
        self._new_successful_source_ids_this_round: set[int] = set()
        self._successful_updates_this_round = 0
        self._successful_additions_blocked_this_round = 0

        # Successful-region merging state.
        #
        # absorbed_id -> surviving_id
        #
        # The alias map records basin coalescence and is useful for interpreting
        # old lineage metadata.  Absorbed representatives are removed from the
        # protected successful set and from the current archive rebuild.
        self.successful_region_aliases: dict[int, int] = {}
        self.successful_region_merge_history: list[dict[str, Any]] = []
        self._successful_regions_merged_this_round = 0
        self._successful_region_count_before_merge = 0
        self._successful_region_count_after_merge = 0
        self._successful_merged_ids_this_round: set[int] = set()
        self._successful_merge_pairs_this_round: list[tuple[int, int, float]] = []

        self.round_index = 0

        # Monitoring snapshots populated during archive rebuild/allocation/restart.
        self.last_archive_diagnostics: dict[str, Any] = {}
        self.last_allocation_diagnostics: dict[str, Any] = {}
        self.last_restart_diagnostics: dict[str, Any] = {}

    def anchor_archive_statistics(self) -> dict[str, Any]:
        """Return theta-space archive geometry/phi diagnostics using small matrices."""
        if not self.anchors:
            raise RuntimeError("Cannot summarize an empty anchor archive.")
        X = np.vstack([flatten_theta(anchor.theta, self.nstate, self.block_weights) for anchor in self.anchors])
        n_anchor, dimension = X.shape
        phi = np.array([anchor.phi for anchor in self.anchors], dtype=float)

        if n_anchor == 1:
            distance_min = distance_mean = distance_median = distance_max = 0.0
            trace = largest = effective_rank = 0.0
        else:
            norms = np.sum(X * X, axis=1)
            distance_sq = norms[:, None] + norms[None, :] - 2.0 * X @ X.T
            distance_sq = np.maximum(distance_sq, 0.0)
            upper = np.sqrt(distance_sq[np.triu_indices(n_anchor, k=1)])
            distance_min, distance_mean = float(np.min(upper)), float(np.mean(upper))
            distance_median, distance_max = float(np.median(upper)), float(np.max(upper))
            Xc = X - np.mean(X, axis=0, keepdims=True)
            gram_cov = (Xc @ Xc.T) / (n_anchor - 1.0)
            gram_cov = 0.5 * (gram_cov + gram_cov.T)
            eigenvalues = np.clip(np.linalg.eigvalsh(gram_cov), 0.0, None)
            trace = float(np.sum(eigenvalues))
            largest = float(np.max(eigenvalues))
            positive = eigenvalues[eigenvalues > 0.0]
            if trace > 0.0 and positive.size:
                probabilities = positive / trace
                effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
            else:
                effective_rank = 0.0

        logdet = covariance_logdet_from_vectors(X, regularization=self.covariance_regularization)
        out = {
            "anchor_count": int(n_anchor),
            "distance_min": distance_min, "distance_mean": distance_mean,
            "distance_median": distance_median, "distance_max": distance_max,
            "anchor_cov_trace": trace,
            "anchor_cov_largest_eigenvalue": largest,
            "anchor_cov_largest_eigenvalue_fraction": (float(largest / trace) if trace > 0.0 else 0.0),
            "anchor_effective_rank": effective_rank,
            "logdet_Ck": float(logdet),
            "anchor_phi_best": float(np.min(phi)),
            "anchor_phi_q25": float(np.quantile(phi, 0.25)),
            "anchor_phi_median": float(np.median(phi)),
            "anchor_phi_mean": float(np.mean(phi)),
            "anchor_phi_std": float(np.std(phi)),
            "anchor_phi_q75": float(np.quantile(phi, 0.75)),
            "anchor_phi_max": float(np.max(phi)),
            "anchor_phi_gap_median": float(np.median(phi) - np.min(phi)),
            "anchor_phi_gap_max": float(np.max(phi) - np.min(phi)),
        }
        if self.relative_error_fn is not None:
            re = np.array([self.relative_error_fn(anchor.theta) for anchor in self.anchors], dtype=float)
            out.update({
                "anchor_re_best": float(np.min(re)), "anchor_re_mean": float(np.mean(re)),
                "anchor_re_median": float(np.median(re)), "anchor_re_std": float(np.std(re)),
            })
        return out

    def phase_length_for(self, phase: Phase, cycle: int | None = None) -> int:
        """Return the scheduled length of a GLOBAL or LOCAL phase."""
        cycle_index = self.cycle if cycle is None else int(cycle)

        if phase is Phase.GLOBAL:
            raw = self.global_stage_length * self.global_stage_decay**cycle_index
            return max(
                self.minimum_global_stage_length,
                int(round(raw)),
            )

        if phase is Phase.LOCAL:
            raw = self.local_stage_length * self.local_stage_decay**cycle_index
            return max(
                self.minimum_local_stage_length,
                int(round(raw)),
            )

        if phase is Phase.BURN_IN:
            return self.initial_burnin_length

        return 0

    def current_stage_length(self) -> int:
        """Return the length assigned to the current phase."""
        if self.phase is Phase.BURN_IN:
            return self.initial_burnin_length
        return self.phase_length_for(self.phase, self.cycle)

    def active_subdomain_indices(self, rank: int) -> np.ndarray:
        """
        Flat subdomain indices used internally by restart and random-reach code.
        """
        if self.phase in (Phase.BURN_IN, Phase.GLOBAL):
            return np.arange(
                0,
                self.nstate,          # all subdomain blocks and the extra (rho) blocks
                1,
                dtype=idtype,
            )

        if self.phase is Phase.LOCAL:
            return np.array(
                [idtype(int(rank) % self.numsubd)],
                dtype=idtype,
            )

        return np.empty(0, dtype=idtype)
    
    def listsubdomains(self, rank: int):
        """
        Legacy structure required by correct_theta_update().
        """
        if self.phase in (Phase.BURN_IN, Phase.GLOBAL):
            return [
                np.arange(
                    0,
                    self.numsubd,
                    1,
                    dtype=idtype,
                )
            ]

        if self.phase is Phase.LOCAL:
            return [
                idtype(int(rank) % self.numsubd)
            ]

        return []
    
    def subdomain_schedule(self, rank: int) -> list[np.ndarray]:
        """
        Return list_subd in the form expected by subdomain_driver().

        Examples
        --------
        Global phase:
            [array([0, 1, 2, 3])]

        Local phase for target_sub = 2:
            [array([2])]
        """
        if self.phase in (Phase.BURN_IN, Phase.GLOBAL):
            return [
                np.arange(
                    self.numsubd,
                    dtype=idtype,
                )
            ]

        if self.phase is Phase.LOCAL:
            target_sub = int(rank) % self.numsubd

            return [
                np.array(
                    [target_sub],
                    dtype=idtype,
                )
            ]

        return []

    # def active_subdomains(self, rank: int) -> np.ndarray:
    #     """
    #     Return listsubdomains in the form expected by correct_theta_update().

    #     Global phase:
    #         [np.array([0, 1, ..., numsubd - 1])]

    #     Local phase:
    #         [rank % numsubd]
    #     """
    #     if self.phase is Phase.GLOBAL:
    #         return [
    #             np.arange(
    #                 0,
    #                 self.numsubd,
    #                 1,
    #                 dtype=idtype,
    #             )
    #         ]

    #     if self.phase is Phase.LOCAL:
    #         return [
    #             idtype(int(rank) % self.numsubd)
    #         ]

    #     return []

    def local_stages_enabled(self, iteration: int | None = None) -> bool:
        """Return whether LOCAL stages are still enabled."""
        if self.disable_local_after_iteration is None:
            return True
        reference = self.phase_start_iteration if iteration is None else int(iteration)
        return reference < self.disable_local_after_iteration

    def phase_has_ended(self, iteration: int) -> bool:
        return (
            self.phase is not Phase.FINISHED
            and int(iteration) >= self.phase_stop_iteration
        )

    def advance_phase(self, next_iteration: int) -> None:
        """Advance the schedule, optionally switching permanently to GLOBAL stages."""
        next_iteration = int(next_iteration)
        local_enabled = self.local_stages_enabled(next_iteration)

        if self.phase is Phase.BURN_IN:
            self.phase = Phase.LOCAL if local_enabled else Phase.GLOBAL
        elif self.phase is Phase.GLOBAL:
            if local_enabled:
                self.phase = Phase.LOCAL
            else:
                self.cycle += 1
                if self.cycle >= self.num_cycles:
                    self.phase = Phase.FINISHED
                    self.phase_stop_iteration = np.iinfo(np.int64).max
                    return
                self.phase = Phase.GLOBAL
        elif self.phase is Phase.LOCAL:
            self.cycle += 1
            if self.cycle >= self.num_cycles:
                self.phase = Phase.FINISHED
                self.phase_stop_iteration = np.iinfo(np.int64).max
                return
            self.phase = Phase.GLOBAL
        else:
            return

        self.phase_index += 1
        self.phase_start_iteration = next_iteration
        self.phase_stop_iteration = (
            self.phase_start_iteration
            + self.phase_length_for(self.phase, self.cycle)
        )

    def initialize_phase_best(
        self,
        theta: Sequence[np.ndarray],
        phi: float,
        simuldata: Any = None,
    ) -> None:
        self.phase_best_theta = copy_theta(theta)
        self.phase_best_phi = float(phi)
        self.phase_best_role = str(self.current_chain_role)
        self.phase_best_parent_anchor_id = int(self.current_source_anchor_id)

    def record_phase_best(
        self,
        theta: Sequence[np.ndarray],
        phi: float,
        simuldata: Any = None,
    ) -> bool:
        """Record a personal phase best together with restart lineage."""
        phi = float(phi)
        if not np.isfinite(phi):
            return False
        if self.phase_best_theta is None or phi < self.phase_best_phi:
            self.phase_best_theta = copy_theta(theta)
            self.phase_best_phi = phi
            self.phase_best_role = str(self.current_chain_role)
            self.phase_best_parent_anchor_id = int(self.current_source_anchor_id)
            return True
        return False

    def gather_candidates_mpi(
        self,
        *,
        current_theta: Sequence[np.ndarray],
        current_phi: float,
        current_simuldata: Any,
        rank: int,
        comm: Any,
    ) -> list[Anchor] | None:
        """Gather archive memory, current states, and per-rank phase bests."""
        if self.phase_best_theta is None:
            self.initialize_phase_best(current_theta, current_phi, current_simuldata)

        local_candidates = [
            Anchor(
                theta=copy_theta(current_theta),
                phi=float(current_phi),
                source="current",
                source_rank=int(rank),
                phase_index=self.phase_index,
                parent_anchor_id=int(self.current_source_anchor_id),
                source_role=str(self.current_chain_role),
            ),
        ]
        # Default: also submit the best state this chain visited during the
        # stage that just ended. With current_iteration_only it is skipped.
        if not self.current_iteration_only:
            local_candidates.append(
                Anchor(
                    theta=copy_theta(self.phase_best_theta),
                    phi=float(self.phase_best_phi),
                    source="phase_best",
                    source_rank=int(rank),
                    phase_index=self.phase_index,
                    parent_anchor_id=int(self.phase_best_parent_anchor_id),
                    source_role=str(self.phase_best_role),
                )
            )

        gathered = comm.gather(local_candidates, root=0)
        if rank != 0:
            return None

        candidates = [anchor.clone() for anchor in self.anchors]
        for rank_candidates in gathered:
            candidates.extend(anchor.clone() for anchor in rank_candidates)
        return candidates

    def _ensure_anchor_id(self, anchor: Anchor) -> Anchor:
        out = anchor.clone()
        if out.anchor_id < 0:
            out.anchor_id = int(self._next_anchor_id)
            self._next_anchor_id += 1
        return out

    def _update_best_anchor(self, candidates: Sequence[Anchor]) -> Anchor:
        """Retain the best-ever anchor so archive phi_min cannot increase."""
        finite = [candidate for candidate in candidates if np.isfinite(candidate.phi)]

        if finite:
            candidate_best = min(finite, key=lambda candidate: candidate.phi)
            if self.best_anchor is None or candidate_best.phi < self.best_anchor.phi:
                self.best_anchor = self._ensure_anchor_id(candidate_best)

        if self.best_anchor is None:
            raise RuntimeError("No finite candidate was available for best anchor.")

        return self.best_anchor.clone()

    def _update_frontier_descendant(self, candidates: Sequence[Anchor]) -> None:
        """Move the active frontier representative to its best descendant.

        Descendants retain the campaign's anchor_id. This makes the frontier a
        persistent region/campaign even though its representative state moves.
        """
        if self.frontier_anchor_id is None:
            return

        frontier_id = int(self.frontier_anchor_id)
        relevant = [
            candidate
            for candidate in candidates
            if (candidate.anchor_id == frontier_id)
            or (
                candidate.parent_anchor_id == frontier_id
                and candidate.source_role == "frontier"
            )
        ]
        if not relevant:
            return

        best = min(relevant, key=lambda anchor: anchor.phi)
        previous = (
            float(self.frontier_best_phi)
            if self.frontier_best_phi is not None
            else float(best.phi)
        )
        self.frontier_previous_phi = previous
        self.frontier_best_phi = min(previous, float(best.phi))

        if best.parent_anchor_id == frontier_id and best.source_role == "frontier":
            best.anchor_id = frontier_id
            best.source = "frontier_descendant"

    def _deduplicate(self, candidates: Sequence[Anchor]) -> list[Anchor]:
        unique: list[Anchor] = []
        tolerance_sq = self.duplicate_tolerance**2
        retained_ids: set[int] = set()

        # Sorting by phi ensures that if a frontier representative and one of
        # its descendants share an ID, the better representative survives.
        for candidate in sorted(candidates, key=lambda item: item.phi):
            if not np.isfinite(candidate.phi):
                continue
            if candidate.anchor_id >= 0 and candidate.anchor_id in retained_ids:
                continue
            duplicate = any(
                theta_distance_squared(candidate.theta, retained.theta, self.nstate, self.block_weights)
                <= tolerance_sq
                for retained in unique
            )
            if not duplicate:
                unique.append(candidate.clone())
                if candidate.anchor_id >= 0:
                    retained_ids.add(int(candidate.anchor_id))
        return unique

    def _eligible_anchor_candidates(
        self,
        candidates: Sequence[Anchor],
    ) -> tuple[Anchor, list[Anchor]]:
        # build_anchor_set already deduplicates once; avoid a second O(N^2 D) pass.
        unique = list(candidates)
        if not unique:
            raise RuntimeError("No finite candidate states were available.")

        best_anchor = self._update_best_anchor(unique)
        eligible = [
            candidate.clone()
            for candidate in unique
            if candidate.phi <= best_anchor.phi + self.quality_tolerance
        ]

        if not any(
            theta_distance_squared(
                candidate.theta,
                best_anchor.theta,
                self.nstate, self.block_weights,
            ) <= self.duplicate_tolerance**2
            for candidate in eligible
        ):
            eligible.append(best_anchor.clone())

        return best_anchor, eligible

    def _build_farthest_anchor_set(
        self,
        best_anchor: Anchor,
        eligible: Sequence[Anchor],
    ) -> list[Anchor]:
        selected = [best_anchor.clone()]
        remaining = [
            candidate.clone()
            for candidate in eligible
            if theta_distance_squared(
                candidate.theta,
                best_anchor.theta,
                self.nstate, self.block_weights,
            ) > self.duplicate_tolerance**2
        ]
        minimum_distance_sq = self.minimum_anchor_distance**2

        while remaining:
            nearest = np.array(
                [
                    min(
                        theta_distance_squared(
                            candidate.theta,
                            anchor.theta,
                            self.nstate, self.block_weights,
                        )
                        for anchor in selected
                    )
                    for candidate in remaining
                ],
                dtype=float,
            )
            chosen_index = int(np.argmax(nearest))
            if float(nearest[chosen_index]) < minimum_distance_sq:
                break

            selected.append(remaining.pop(chosen_index))
            if self.max_anchors is not None and len(selected) >= self.max_anchors:
                break

        return selected

    def _build_joint_logdet_anchor_set(
        self, best_anchor: Anchor, eligible: Sequence[Anchor],
    ) -> list[Anchor]:
        raise RuntimeError("joint_logdet mode is unavailable: anchors no longer store simuldata.")

    def _build_diversity_anchor_set(
        self,
        best_anchor: Anchor,
        eligible: Sequence[Anchor],
    ) -> list[Anchor]:
        """Build the stochastic diversity archive around protected successes.

        Persistent successful-region representatives are inserted first and are
        exempt from the moving hard quality gate. The best-ever state is also
        guaranteed. Ordinary candidates still pass the existing quality gate and
        minimum-distance gate, then compete using

            S = alpha * G + (1-alpha) * D.

        All protected anchors count against max_anchors.
        """
        minimum_distance_sq = self.minimum_anchor_distance**2
        duplicate_tolerance_sq = self.duplicate_tolerance**2
        distance_scale_sq = max(
            minimum_distance_sq,
            duplicate_tolerance_sq,
            np.finfo(float).eps,
        )

        selected: list[Anchor] = []
        selected_vectors_list: list[np.ndarray] = []

        def add_protected(anchor: Anchor, *, require_distance: bool = True) -> bool:
            nonlocal selected, selected_vectors_list
            candidate = self._ensure_anchor_id(anchor)
            vector = flatten_theta(candidate.theta, self.nstate, self.block_weights)
            if selected_vectors_list:
                X = np.vstack(selected_vectors_list)
                nearest_sq = float(np.min(np.sum((X - vector) ** 2, axis=1)))
                if nearest_sq <= duplicate_tolerance_sq:
                    # If this is the exact same physical state, keep the lower-phi
                    # representative already present.
                    return False
                if require_distance and nearest_sq < minimum_distance_sq:
                    return False
            if self.max_anchors is not None and len(selected) >= self.max_anchors:
                return False
            selected.append(candidate)
            selected_vectors_list.append(vector)
            return True

        # 1) Persistent successful regions. They are already separated using the
        # successful-region distance criterion, which is >= minimum anchor distance.
        for successful in sorted(self.successful_anchors, key=lambda a: a.phi):
            add_protected(successful, require_distance=True)

        # 2) Best-ever state is always retained. If it belongs to an already
        # protected successful region, that region's representative should be the
        # same/better state and the hard-distance test naturally prevents a clone.
        if not selected:
            add_protected(best_anchor, require_distance=False)
        else:
            best_vec = flatten_theta(best_anchor.theta, self.nstate, self.block_weights)
            X = np.vstack(selected_vectors_list)
            nearest_index = int(np.argmin(np.sum((X - best_vec) ** 2, axis=1)))
            nearest_sq = float(np.sum((X[nearest_index] - best_vec) ** 2))
            if nearest_sq < minimum_distance_sq:
                # Preserve the global best by replacing the nearby protected
                # representative if the best has lower phi. Keep the stable
                # successful region ID when applicable.
                if best_anchor.phi < selected[nearest_index].phi:
                    stable_id = int(selected[nearest_index].anchor_id)
                    replacement = best_anchor.clone()
                    replacement.anchor_id = stable_id
                    selected[nearest_index] = replacement
                    selected_vectors_list[nearest_index] = best_vec
            else:
                add_protected(best_anchor, require_distance=True)

        if not selected:
            add_protected(best_anchor, require_distance=False)

        selected_vectors = np.vstack(selected_vectors_list)
        selected_ids = set(int(a.anchor_id) for a in selected)

        remaining_anchors: list[Anchor] = []
        remaining_vectors: list[np.ndarray] = []
        for candidate in eligible:
            # Protected states are already present; do not let them re-enter the
            # stochastic pool under another copy.
            if candidate.anchor_id >= 0 and int(candidate.anchor_id) in selected_ids:
                continue
            vector = flatten_theta(candidate.theta, self.nstate, self.block_weights)
            nearest_sq = float(np.min(np.sum((selected_vectors - vector) ** 2, axis=1)))
            if nearest_sq <= duplicate_tolerance_sq:
                continue
            remaining_anchors.append(candidate.clone())
            remaining_vectors.append(vector)

        self._selection_diagnostics = {
            "distance_eligible_initial": 0,
            "score_rejected_final": 0,
            "archive_hit_cap": False,
            "protected_frontier_retained": False,
            "protected_successful_count": int(len(self.successful_anchors)),
            "selected_scores": [],
            "selected_logdet_novelty": [],
            "selected_distance_novelty": [],
        }

        if not remaining_vectors:
            return selected

        remaining_vectors_array = np.vstack(remaining_vectors)
        initial_distances = np.min(
            np.sum(
                (remaining_vectors_array[:, None, :] - selected_vectors[None, :, :])**2,
                axis=2,
            ),
            axis=1,
        )
        self._selection_diagnostics["distance_eligible_initial"] = int(
            np.sum(initial_distances >= minimum_distance_sq)
        )

        # Protect an active unresolved frontier if it remains inside the ordinary
        # hard quality gate and is sufficiently distinct from all protected states.
        if self.frontier_anchor_id is not None:
            frontier_position = next(
                (
                    i for i, anchor in enumerate(remaining_anchors)
                    if anchor.anchor_id == self.frontier_anchor_id
                ),
                None,
            )
            if frontier_position is not None:
                vector = remaining_vectors_array[frontier_position]
                nearest_sq = float(np.min(np.sum((selected_vectors - vector)**2, axis=1)))
                if nearest_sq >= minimum_distance_sq:
                    frontier = self._ensure_anchor_id(
                        remaining_anchors.pop(frontier_position)
                    )
                    if self.max_anchors is None or len(selected) < self.max_anchors:
                        selected.append(frontier)
                        selected_vectors = np.vstack([selected_vectors, vector])
                        remaining_vectors_array = np.delete(
                            remaining_vectors_array, frontier_position, axis=0
                        )
                        self._selection_diagnostics["protected_frontier_retained"] = True

        epsilon = float(self.covariance_regularization)

        while remaining_anchors:
            if self.max_anchors is not None and len(selected) >= self.max_anchors:
                self._selection_diagnostics["archive_hit_cap"] = True
                break

            k = selected_vectors.shape[0]
            selected_norm_sq = np.sum(selected_vectors**2, axis=1)
            remaining_norm_sq = np.sum(remaining_vectors_array**2, axis=1)
            distance_sq = (
                remaining_norm_sq[:, None]
                + selected_norm_sq[None, :]
                - 2.0 * remaining_vectors_array @ selected_vectors.T
            )
            distance_sq = np.maximum(distance_sq, 0.0)
            nearest_sq = np.min(distance_sq, axis=1)
            admissible = nearest_sq >= minimum_distance_sq
            if not np.any(admissible):
                break

            selected_mean = np.mean(selected_vectors, axis=0)
            centered_selected = selected_vectors - selected_mean
            deltas = remaining_vectors_array - selected_mean
            gram = centered_selected @ centered_selected.T
            gram = 0.5 * (gram + gram.T)

            rank_one_factor = k / (k + 1.0)
            trial_gram_matrix = np.eye(k) + gram / (k * epsilon)
            trial_sign, trial_small_logdet = np.linalg.slogdet(trial_gram_matrix)
            if trial_sign <= 0:
                raise np.linalg.LinAlgError(
                    "Regularized trial Gram matrix is not positive definite."
                )

            if k == 1:
                logdet_constant = 0.0
            else:
                current_gram_matrix = np.eye(k) + gram / ((k - 1.0) * epsilon)
                current_sign, current_small_logdet = np.linalg.slogdet(
                    current_gram_matrix
                )
                if current_sign <= 0:
                    raise np.linalg.LinAlgError(
                        "Regularized current Gram matrix is not positive definite."
                    )
                logdet_constant = trial_small_logdet - current_small_logdet

            small_cholesky = np.linalg.cholesky(trial_gram_matrix)
            projection = centered_selected @ deltas.T
            solved_projection = np.linalg.solve(small_cholesky, projection)
            delta_norm_sq = np.sum(deltas**2, axis=1)
            inverse_quadratic = (
                delta_norm_sq / (k * epsilon)
                - np.sum(solved_projection**2, axis=0) / (k * epsilon)**2
            )
            inverse_quadratic = np.maximum(inverse_quadratic, 0.0)
            leverage = rank_one_factor * inverse_quadratic
            raw_gain = logdet_constant + np.log1p(leverage)

            logdet_novelty = 1.0 - np.exp(
                -np.maximum(raw_gain, 0.0) / self.logdet_gain_scale
            )
            distance_novelty = 1.0 - np.exp(
                -0.5 * nearest_sq / distance_scale_sq
            )
            score = (
                self.diversity_alpha * logdet_novelty
                + (1.0 - self.diversity_alpha) * distance_novelty
            )

            qualified = admissible & (score >= self.minimum_diversity_score)
            if not np.any(qualified):
                self._selection_diagnostics["score_rejected_final"] = int(
                    np.sum(admissible)
                )
                break

            qualified_indices = np.flatnonzero(qualified)
            qualified_scores = score[qualified_indices]
            logits = (qualified_scores - np.max(qualified_scores)) / (
                self.archive_selection_temperature
            )
            probability = np.exp(logits)
            probability /= np.sum(probability)
            chosen_index = int(self.rng.choice(qualified_indices, p=probability))

            chosen = self._ensure_anchor_id(remaining_anchors.pop(chosen_index))
            selected.append(chosen)
            selected_vectors = np.vstack([
                selected_vectors, remaining_vectors_array[chosen_index]
            ])
            self._selection_diagnostics["selected_scores"].append(
                float(score[chosen_index])
            )
            self._selection_diagnostics["selected_logdet_novelty"].append(
                float(logdet_novelty[chosen_index])
            )
            self._selection_diagnostics["selected_distance_novelty"].append(
                float(distance_novelty[chosen_index])
            )
            remaining_vectors_array = np.delete(
                remaining_vectors_array, chosen_index, axis=0
            )

        return selected

    # Backward-compatible alias for experiments that still request "logdet".
    # The new prototype uses the unified stochastic diversity score instead.
    def _build_logdet_anchor_set(
        self,
        best_anchor: Anchor,
        eligible: Sequence[Anchor],
    ) -> list[Anchor]:
        return self._build_diversity_anchor_set(best_anchor, eligible)

    def _successful_anchor_statistics(self) -> dict[str, Any]:
        """Diagnostics for the persistent protected successful-region set."""
        n = len(self.successful_anchors)
        if n == 0:
            return {
                "successful_region_count": 0,
                "successful_phi_best": np.nan,
                "successful_phi_mean": np.nan,
                "successful_phi_median": np.nan,
                "successful_phi_max": np.nan,
                "successful_distance_min": np.nan,
                "successful_distance_median": np.nan,
            }

        phi = np.array([a.phi for a in self.successful_anchors], dtype=float)
        if n == 1:
            dmin = dmedian = 0.0
        else:
            X = np.vstack([
                flatten_theta(a.theta, self.nstate, self.block_weights)
                for a in self.successful_anchors
            ])
            diff = X[:, None, :] - X[None, :, :]
            D = np.linalg.norm(diff, axis=2)
            upper = D[np.triu_indices(n, k=1)]
            dmin = float(np.min(upper))
            dmedian = float(np.median(upper))

        return {
            "successful_region_count": int(n),
            "successful_phi_best": float(np.min(phi)),
            "successful_phi_mean": float(np.mean(phi)),
            "successful_phi_median": float(np.median(phi)),
            "successful_phi_max": float(np.max(phi)),
            "successful_distance_min": dmin,
            "successful_distance_median": dmedian,
        }

    def _update_successful_regions(
        self,
        candidates: Sequence[Anchor],
    ) -> dict[str, Any]:
        """Persistently add/update distinct regions that meet the target phi.

        A successful region is represented by one protected anchor. New target-
        reaching candidates within successful_region_distance of an existing
        representative update that representative only if they have lower phi.
        Candidates farther from every successful representative create a new
        region, subject to max_anchors. Once promoted, a successful region is not
        removed by later tightening of the moving best+quality_tolerance gate.
        """
        self._successful_source_ids_this_round = set()
        self._new_successful_source_ids_this_round = set()
        self._successful_updates_this_round = 0
        self._successful_additions_blocked_this_round = 0

        if self.frontier_target_phi is None:
            self.successful_region_ids = [
                int(a.anchor_id) for a in self.successful_anchors
            ]
            return self._successful_anchor_statistics()

        target = float(self.frontier_target_phi)
        minimum_sq = self.successful_region_distance**2

        qualified = sorted(
            [c.clone() for c in candidates if np.isfinite(c.phi) and c.phi <= target],
            key=lambda a: a.phi,
        )

        for candidate in qualified:
            source_id = int(candidate.anchor_id)
            if source_id >= 0:
                self._successful_source_ids_this_round.add(source_id)

            if not self.successful_anchors:
                if self.max_anchors is not None and self.max_anchors < 1:
                    self._successful_additions_blocked_this_round += 1
                    continue
                promoted = self._ensure_anchor_id(candidate)
                promoted.source = (
                    "successful_frontier" if candidate.source_role == "frontier"
                    else "successful"
                )
                self.successful_anchors.append(promoted)
                self._new_successful_source_ids_this_round.add(
                    source_id if source_id >= 0 else int(promoted.anchor_id)
                )
                continue

            distances_sq = np.array([
                theta_distance_squared(
                    candidate.theta,
                    successful.theta,
                    self.nstate, self.block_weights,
                )
                for successful in self.successful_anchors
            ])
            nearest = int(np.argmin(distances_sq))

            if float(distances_sq[nearest]) < minimum_sq:
                # Same successful region. Keep a stable region/anchor ID but move
                # its representative to a better state when one is discovered.
                if candidate.phi < self.successful_anchors[nearest].phi:
                    stable_id = int(self.successful_anchors[nearest].anchor_id)
                    replacement = candidate.clone()
                    replacement.anchor_id = stable_id
                    replacement.source = (
                        "successful_frontier_update"
                        if candidate.source_role == "frontier"
                        else "successful_update"
                    )
                    self.successful_anchors[nearest] = replacement
                    self._successful_updates_this_round += 1
                continue

            if self.max_anchors is not None and len(self.successful_anchors) >= self.max_anchors:
                self._successful_additions_blocked_this_round += 1
                continue

            promoted = self._ensure_anchor_id(candidate)
            promoted.source = (
                "successful_frontier" if candidate.source_role == "frontier"
                else "successful"
            )
            self.successful_anchors.append(promoted)
            self._new_successful_source_ids_this_round.add(
                source_id if source_id >= 0 else int(promoted.anchor_id)
            )

        # Keep the set deterministic and synchronized: best successful region first.
        self.successful_anchors.sort(key=lambda a: a.phi)
        self.successful_region_ids = [
            int(a.anchor_id) for a in self.successful_anchors
        ]
        return self._successful_anchor_statistics()

    def _canonical_successful_region_id(self, region_id: int) -> int:
        """Resolve a historical/absorbed region ID to its current survivor."""
        region_id = int(region_id)
        seen: set[int] = set()
        while region_id in self.successful_region_aliases:
            if region_id in seen:
                raise RuntimeError("Cycle detected in successful_region_aliases.")
            seen.add(region_id)
            region_id = int(self.successful_region_aliases[region_id])
        return region_id

    def _closest_successful_region_pair(
        self,
    ) -> tuple[int, int, float] | None:
        """Return indices and distance of the closest successful pair."""
        n = len(self.successful_anchors)
        if n < 2:
            return None

        X = np.vstack(
            [
                flatten_theta(anchor.theta, self.nstate, self.block_weights)
                for anchor in self.successful_anchors
            ]
        )
        differences = X[:, None, :] - X[None, :, :]
        distances = np.linalg.norm(differences, axis=2)
        np.fill_diagonal(distances, np.inf)

        flat_index = int(np.argmin(distances))
        i, j = np.unravel_index(flat_index, distances.shape)
        if i == j:
            return None
        if j < i:
            i, j = j, i
        return int(i), int(j), float(distances[i, j])

    def _merge_successful_region_pair(
        self,
        i: int,
        j: int,
        distance: float,
    ) -> None:
        """Merge two overlapping successful regions.

        The lower-phi representative survives.  The survivor keeps its stable
        region/anchor ID, while the absorbed ID is retained in an alias map for
        lineage diagnostics.  Exploration-visit counts are accumulated.

        This operation changes the region model, not the MCMC trajectory: chains
        are still free to move anywhere.  Only the persistent archive
        representatives are coalesced.
        """
        if i == j:
            return
        if j < i:
            i, j = j, i

        left = self.successful_anchors[i]
        right = self.successful_anchors[j]

        if left.phi <= right.phi:
            survivor_index, absorbed_index = i, j
        else:
            survivor_index, absorbed_index = j, i

        survivor = self.successful_anchors[survivor_index].clone()
        absorbed = self.successful_anchors[absorbed_index].clone()

        survivor_id = int(survivor.anchor_id)
        absorbed_id = int(absorbed.anchor_id)

        survivor.source = "successful_merge_survivor"
        survivor.exploration_visits = int(
            survivor.exploration_visits + absorbed.exploration_visits
        )

        # Any earlier aliases that pointed to the absorbed region now resolve
        # transitively through absorbed_id -> survivor_id.
        self.successful_region_aliases[absorbed_id] = survivor_id
        self._successful_merged_ids_this_round.add(absorbed_id)
        self._successful_merge_pairs_this_round.append(
            (survivor_id, absorbed_id, float(distance))
        )
        self._successful_regions_merged_this_round += 1

        self.successful_region_merge_history.append(
            {
                "round_index": int(self.round_index),
                "survivor_region_id": survivor_id,
                "absorbed_region_id": absorbed_id,
                "distance": float(distance),
                "survivor_phi": float(survivor.phi),
                "absorbed_phi": float(absorbed.phi),
            }
        )

        # Delete from high index first, then overwrite the surviving location.
        keep_position = survivor_index
        drop_position = absorbed_index
        if drop_position < keep_position:
            keep_position -= 1

        del self.successful_anchors[drop_position]
        self.successful_anchors[keep_position] = survivor

        # A cooldown entry attached to an absorbed ID should not resurrect the
        # retired representative.  Preserve the stricter cooldown, if any, on
        # the surviving basin.
        absorbed_cooldown = self.frontier_cooldown_until.pop(absorbed_id, -1)
        if absorbed_cooldown >= 0:
            self.frontier_cooldown_until[survivor_id] = max(
                self.frontier_cooldown_until.get(survivor_id, -1),
                absorbed_cooldown,
            )

    def _merge_overlapping_successful_regions(
        self,
    ) -> dict[str, Any]:
        """Iteratively merge successful representatives inside merge radius.

        The closest pair is merged first, then distances are recomputed.  This
        permits cascading coalescence when several apparent regions converge to
        the same basin.

        Creation uses ``successful_region_distance`` whereas merging uses the
        smaller ``successful_region_merge_distance``.  The interval between the
        two radii is a hysteresis band.
        """
        self._successful_regions_merged_this_round = 0
        self._successful_region_count_before_merge = len(self.successful_anchors)
        self._successful_merged_ids_this_round = set()
        self._successful_merge_pairs_this_round = []

        while len(self.successful_anchors) >= 2:
            closest = self._closest_successful_region_pair()
            if closest is None:
                break

            i, j, distance = closest
            if distance > self.successful_region_merge_distance:
                break

            self._merge_successful_region_pair(i, j, distance)

        self.successful_anchors.sort(key=lambda anchor: anchor.phi)
        self.successful_region_ids = [
            int(anchor.anchor_id) for anchor in self.successful_anchors
        ]
        self._successful_region_count_after_merge = len(self.successful_anchors)

        stats = self._successful_anchor_statistics()
        stats.update(
            {
                "successful_region_creation_distance": float(
                    self.successful_region_distance
                ),
                "successful_region_merge_distance": float(
                    self.successful_region_merge_distance
                ),
                "successful_region_count_before_merge": int(
                    self._successful_region_count_before_merge
                ),
                "successful_region_count_after_merge": int(
                    self._successful_region_count_after_merge
                ),
                "successful_regions_merged_this_round": int(
                    self._successful_regions_merged_this_round
                ),
                "successful_merge_pair_count": int(
                    len(self._successful_merge_pairs_this_round)
                ),
            }
        )
        return stats

    def _successful_region_indices(self) -> list[int]:
        """Indices in the current archive corresponding to persistent successes."""
        successful_ids = set(int(a.anchor_id) for a in self.successful_anchors)
        return [
            i for i, anchor in enumerate(self.anchors)
            if int(anchor.anchor_id) in successful_ids
        ]

    def _frontier_candidate_table(
        self,
    ) -> list[dict[str, Any]]:
        """Return unresolved frontier candidates and their competitive scores.

        A candidate must

        1. not already be a successful protected region,
        2. remain outside the frontier distance from protected regions,
        3. not be in cooldown,
        4. remain within ``frontier_relative_quality_tolerance`` of the current
           best archive state when that tolerance is enabled.

        The competitive score balances relative quality and geometric novelty:

            score = w_q * quality_score + w_d * distance_score,

        where both component scores are normalized to [0, 1] over the current
        unresolved candidate pool. Higher is better.
        """
        if not self.anchors:
            return []

        phi_values = np.array([float(anchor.phi) for anchor in self.anchors])
        best_index = int(np.argmin(phi_values))
        best_phi = float(phi_values[best_index])

        successful_ids = set(int(a.anchor_id) for a in self.successful_anchors)
        self.successful_region_ids = sorted(successful_ids)

        reference_anchors = (
            [a.clone() for a in self.successful_anchors]
            if self.successful_anchors
            else [self.anchors[best_index].clone()]
        )
        minimum_sq = self.frontier_minimum_distance**2

        raw: list[dict[str, Any]] = []
        for index, anchor in enumerate(self.anchors):
            anchor_id = int(anchor.anchor_id)

            if index == best_index:
                continue
            if anchor_id in successful_ids:
                continue

            # Already-successful quality states are not unresolved frontiers,
            # even if they were merged into an existing successful region.
            if (
                self.frontier_target_phi is not None
                and float(anchor.phi) <= self.frontier_target_phi
            ):
                continue

            cooldown_until = self.frontier_cooldown_until.get(anchor_id, -1)
            if self.round_index < cooldown_until:
                continue

            # Hard competitive-quality gate. This is the key safeguard against
            # allowing an old, isolated but clearly suboptimal frontier to
            # consume resources indefinitely as the global best improves.
            if self.frontier_relative_quality_tolerance is not None:
                if (
                    float(anchor.phi)
                    > best_phi + self.frontier_relative_quality_tolerance
                ):
                    continue

            nearest_sq = min(
                theta_distance_squared(
                    anchor.theta,
                    reference.theta,
                    self.nstate, self.block_weights,
                )
                for reference in reference_anchors
            )
            if nearest_sq < minimum_sq:
                continue

            raw.append(
                {
                    "index": int(index),
                    "anchor_id": anchor_id,
                    "phi": float(anchor.phi),
                    "distance": float(np.sqrt(nearest_sq)),
                }
            )

        if not raw:
            return []

        phis = np.array([row["phi"] for row in raw], dtype=float)
        distances = np.array([row["distance"] for row in raw], dtype=float)

        # Higher quality_score means lower phi.
        phi_lo = float(np.min(phis))
        phi_hi = float(np.max(phis))
        if phi_hi > phi_lo:
            quality_scores = (phi_hi - phis) / (phi_hi - phi_lo)
        else:
            quality_scores = np.ones_like(phis)

        distance_lo = float(np.min(distances))
        distance_hi = float(np.max(distances))
        if distance_hi > distance_lo:
            distance_scores = (
                (distances - distance_lo) / (distance_hi - distance_lo)
            )
        else:
            distance_scores = np.ones_like(distances)

        for row, q_score, d_score in zip(
            raw, quality_scores, distance_scores
        ):
            row["quality_score"] = float(q_score)
            row["distance_score"] = float(d_score)
            row["score"] = float(
                self.frontier_quality_weight * q_score
                + self.frontier_distance_weight * d_score
            )

        return raw

    def _activate_frontier_index(self, index: int) -> None:
        """Start a frontier campaign from one archive index."""
        anchor = self.anchors[int(index)]
        self.frontier_anchor_id = int(anchor.anchor_id)
        self.frontier_start_phi = float(anchor.phi)
        self.frontier_previous_phi = float(anchor.phi)
        self.frontier_best_phi = float(anchor.phi)
        self.frontier_rounds = 0
        self.frontier_status = "ACTIVE"

    def _select_new_frontier(self) -> None:
        """Choose the strongest unresolved competitive frontier candidate."""
        table = self._frontier_candidate_table()
        if not table:
            self.frontier_anchor_id = None
            self.frontier_start_phi = None
            self.frontier_previous_phi = None
            self.frontier_best_phi = None
            self.frontier_rounds = 0
            self.frontier_status = "NONE"
            return

        # Stable tie-breaking favors lower phi, then larger distance.
        chosen = max(
            table,
            key=lambda row: (
                row["score"],
                -row["phi"],
                row["distance"],
            ),
        )
        self._activate_frontier_index(int(chosen["index"]))

    def _competitive_reseed_frontier(self) -> bool:
        """Periodically re-compete the active frontier against alternatives.

        Returns True if the active campaign was replaced by a different
        unresolved region. The old region is put on cooldown, preventing an
        immediate flip-flop back to the same campaign.
        """
        if self.frontier_anchor_id is None:
            return False
        if self.frontier_reseed_interval <= 0:
            return False
        if self.frontier_rounds <= 0:
            return False
        if self.frontier_rounds % self.frontier_reseed_interval != 0:
            return False

        table = self._frontier_candidate_table()
        if not table:
            return False

        current_id = int(self.frontier_anchor_id)
        current = next(
            (row for row in table if int(row["anchor_id"]) == current_id),
            None,
        )
        if current is None:
            return False

        challenger = max(
            table,
            key=lambda row: (
                row["score"],
                -row["phi"],
                row["distance"],
            ),
        )
        if int(challenger["anchor_id"]) == current_id:
            return False

        # Require the challenger to actually be more competitive. A tiny
        # floating-point tie is not enough to switch campaigns.
        if float(challenger["score"]) <= float(current["score"]) + 1.0e-12:
            return False

        self.frontier_cooldown_until[current_id] = (
            self.round_index + self.frontier_cooldown_rounds
        )
        self._activate_frontier_index(int(challenger["index"]))
        self.frontier_status = "RESEEDED"
        return True


    def _update_frontier_campaign_after_archive(self) -> dict[str, Any]:
        """Promote/merge, continue, or retire the active frontier campaign."""
        event = "NONE"
        completed_frontier_id = -1
        completed_frontier_phi = np.nan
        round_improvement = 0.0
        total_improvement = 0.0
        relative_quality_gap = np.nan
        reseeded = False

        if self.frontier_anchor_id is not None:
            completed_frontier_id = int(self.frontier_anchor_id)
            match = next(
                (
                    anchor
                    for anchor in self.anchors
                    if anchor.anchor_id == self.frontier_anchor_id
                ),
                None,
            )

            reached_target_this_round = (
                completed_frontier_id in self._successful_source_ids_this_round
                or completed_frontier_id in self.successful_region_ids
            )
            created_new_region = (
                completed_frontier_id in self._new_successful_source_ids_this_round
                or completed_frontier_id in self.successful_region_ids
            )

            if reached_target_this_round:
                event = "PROMOTED" if created_new_region else "MERGED_SUCCESS"
                if match is not None:
                    completed_frontier_phi = float(match.phi)
                elif self.frontier_best_phi is not None:
                    completed_frontier_phi = float(self.frontier_best_phi)
                self.frontier_anchor_id = None
                self.frontier_status = event
            elif match is None:
                event = "RETIRED"
                self.frontier_cooldown_until[completed_frontier_id] = (
                    self.round_index + self.frontier_cooldown_rounds
                )
                self.frontier_anchor_id = None
                self.frontier_status = "RETIRED"
            else:
                current_phi = float(match.phi)
                completed_frontier_phi = current_phi
                previous = (
                    current_phi
                    if self.frontier_previous_phi is None
                    else float(self.frontier_previous_phi)
                )
                start = (
                    current_phi
                    if self.frontier_start_phi is None
                    else float(self.frontier_start_phi)
                )
                round_improvement = max(0.0, previous - current_phi)
                total_improvement = max(0.0, start - current_phi)
                self.frontier_best_phi = current_phi
                self.frontier_rounds += 1

                archive_best_phi = float(
                    min(anchor.phi for anchor in self.anchors)
                )
                relative_quality_gap = current_phi - archive_best_phi

                if (
                    self.frontier_relative_quality_tolerance is not None
                    and relative_quality_gap
                    > self.frontier_relative_quality_tolerance
                ):
                    event = "RETIRED_DOMINATED"
                    self.frontier_cooldown_until[completed_frontier_id] = (
                        self.round_index + self.frontier_cooldown_rounds
                    )
                    self.frontier_anchor_id = None
                    self.frontier_status = "RETIRED_DOMINATED"
                elif (
                    self.frontier_rounds >= 2
                    and round_improvement < self.frontier_minimum_improvement
                ):
                    event = "RETIRED_STALE"
                    self.frontier_cooldown_until[completed_frontier_id] = (
                        self.round_index + self.frontier_cooldown_rounds
                    )
                    self.frontier_anchor_id = None
                    self.frontier_status = "RETIRED_STALE"
                elif self.frontier_rounds >= self.frontier_max_rounds:
                    event = "RETIRED_MAX_ROUNDS"
                    self.frontier_cooldown_until[completed_frontier_id] = (
                        self.round_index + self.frontier_cooldown_rounds
                    )
                    self.frontier_anchor_id = None
                    self.frontier_status = "RETIRED_MAX_ROUNDS"
                else:
                    event = "CONTINUE"
                    self.frontier_previous_phi = current_phi
                    self.frontier_status = "ACTIVE"

                    # Every K frontier rounds, re-compete the active campaign
                    # against all unresolved alternatives. This prevents an
                    # early branch from monopolizing frontier resources simply
                    # because it remains geometrically isolated.
                    reseeded = self._competitive_reseed_frontier()
                    if reseeded:
                        event = "RESEEDED"

        self.successful_region_ids = [
            int(a.anchor_id) for a in self.successful_anchors
        ]

        if self.frontier_anchor_id is None:
            self._select_new_frontier()
            if event == "NONE" and self.frontier_anchor_id is not None:
                event = "SELECTED"

        active = next(
            (
                anchor
                for anchor in self.anchors
                if self.frontier_anchor_id is not None
                and anchor.anchor_id == self.frontier_anchor_id
            ),
            None,
        )

        if active is None:
            frontier_distance = np.nan
        else:
            reference = (
                self.successful_anchors
                if self.successful_anchors
                else [min(self.anchors, key=lambda a: a.phi)]
            )
            relevant = [
                a for a in reference if int(a.anchor_id) != int(active.anchor_id)
            ]
            frontier_distance = (
                0.0 if not relevant else float(min(
                    np.sqrt(theta_distance_squared(
                        active.theta, a.theta, self.nstate, self.block_weights
                    ))
                    for a in relevant
                ))
            )

        return {
            "frontier_event": event,
            "frontier_status": str(self.frontier_status),
            "completed_frontier_id": completed_frontier_id,
            "completed_frontier_phi": completed_frontier_phi,
            "frontier_round_improvement": float(round_improvement),
            "frontier_total_improvement": float(total_improvement),
            "frontier_minimum_improvement": float(self.frontier_minimum_improvement),
            "frontier_relative_quality_tolerance": (
                np.nan
                if self.frontier_relative_quality_tolerance is None
                else float(self.frontier_relative_quality_tolerance)
            ),
            "frontier_relative_quality_gap": float(relative_quality_gap),
            "frontier_reseed_interval": int(self.frontier_reseed_interval),
            "frontier_reseeded": int(reseeded),
            "frontier_quality_weight": float(self.frontier_quality_weight),
            "frontier_distance_weight": float(self.frontier_distance_weight),
            "frontier_progress_ok": int(
                total_improvement >= self.frontier_minimum_improvement
            ),
            "frontier_active": int(active is not None),
            "frontier_id": -1 if active is None else int(active.anchor_id),
            "frontier_phi": np.nan if active is None else float(active.phi),
            "frontier_round": 0 if active is None else int(self.frontier_rounds),
            "frontier_start_phi": (
                np.nan if self.frontier_start_phi is None
                else float(self.frontier_start_phi)
            ),
            "frontier_gap_to_best": (
                np.nan if active is None
                else float(active.phi - min(a.phi for a in self.anchors))
            ),
            "frontier_gap_to_target": (
                np.nan
                if active is None or self.frontier_target_phi is None
                else float(active.phi - self.frontier_target_phi)
            ),
            "frontier_distance_to_success": frontier_distance,
            "new_successful_regions_this_round": int(
                len(self._new_successful_source_ids_this_round)
            ),
            "successful_region_updates_this_round": int(
                self._successful_updates_this_round
            ),
            "successful_additions_blocked_this_round": int(
                self._successful_additions_blocked_this_round
            ),
            "successful_region_creation_distance": float(
                self.successful_region_distance
            ),
            "successful_region_merge_distance": float(
                self.successful_region_merge_distance
            ),
            "successful_region_count_before_merge": int(
                self._successful_region_count_before_merge
            ),
            "successful_region_count_after_merge": int(
                self._successful_region_count_after_merge
            ),
            "successful_regions_merged_this_round": int(
                self._successful_regions_merged_this_round
            ),
            "successful_merge_pair_count": int(
                len(self._successful_merge_pairs_this_round)
            ),
            **self._successful_anchor_statistics(),
        }

    def build_anchor_set(self, candidates: Sequence[Anchor]) -> list[Anchor]:
        """Rebuild the stochastic diversity archive with persistent best/frontier."""
        raw_count = len(candidates)
        previous_best_phi = (
            np.nan if self.best_anchor is None else float(self.best_anchor.phi)
        )
        completed_phase = self.phase.name
        completed_phase_index = int(self.phase_index)
        completed_cycle = int(self.cycle)
        completed_stage_length = int(self.current_stage_length())
        self._update_frontier_descendant(candidates)
        unique = self._deduplicate(candidates)
        best_anchor, eligible = self._eligible_anchor_candidates(unique)

        # Promotion is evaluated before archive selection so every already-
        # successful region is inserted as protected memory in this rebuild.
        successful_diag_pre = self._update_successful_regions(
            [*unique, best_anchor]
        )

        # Successful representatives are now allowed to move as better states
        # are discovered.  Once updates/promotions are complete, coalesce any
        # representatives that have entered the merge radius.  This is the
        # pruning step that turns many apparent local regions into one basin.
        successful_merge_diag = self._merge_overlapping_successful_regions()

        # Do not immediately reinsert an absorbed representative from the old
        # archive as an ordinary diversity anchor during this same rebuild.
        if self._successful_merged_ids_this_round:
            eligible = [
                anchor
                for anchor in eligible
                if int(anchor.anchor_id) not in self._successful_merged_ids_this_round
            ]

        if self.archive_selection_mode in {"diversity", "logdet"}:
            selected = self._build_diversity_anchor_set(best_anchor, eligible)
        else:
            selected = self._build_farthest_anchor_set(best_anchor, eligible)

        if not selected:
            selected = [self._ensure_anchor_id(best_anchor)]
        selected = [self._ensure_anchor_id(anchor) for anchor in selected]

        # --- Robustness guard: always preserve the best-ever anchor ----------
        # The successful-region merge exclusion above (dropping
        # ``_successful_merged_ids_this_round`` from ``eligible``) can remove the
        # persistent best anchor from the eligible pool. When that happens at
        # convergence -- e.g. when the shrinking anchor cloud lets many/all
        # anchors cascade into successful regions in a single round -- the
        # selected set no longer contains the best-ever state, which previously
        # tripped the non-increasing phi_min invariant and raised. Re-inject the
        # best anchor here so the invariant it guards ("archive phi_min never
        # increases") holds by construction rather than by crashing. This keeps
        # all other selection behavior intact.
        if min(anchor.phi for anchor in selected) > best_anchor.phi + 1.0e-12:
            best_clone = self._ensure_anchor_id(best_anchor.clone())
            best_id = int(best_clone.anchor_id)
            # Drop any stale copy of the best id, then prepend the protected best.
            selected = [best_clone] + [
                anchor for anchor in selected
                if int(anchor.anchor_id) != best_id
            ]

        # Invariant must now hold; keep it as a hard assertion in case a future
        # change reintroduces a path that loses the best anchor.
        if min(anchor.phi for anchor in selected) > best_anchor.phi + 1.0e-12:
            raise RuntimeError(
                "Archive rebuild violated the non-increasing phi_min invariant "
                "even after best-anchor reinjection."
            )

        self.anchors = [anchor.clone() for anchor in selected]
        self.best_anchor = min(self.anchors, key=lambda anchor: anchor.phi).clone()

        # Refresh persistent successful representatives from their protected
        # archive copies, while retaining their stable IDs across rounds.
        archive_by_id = {int(a.anchor_id): a for a in self.anchors}
        self.successful_anchors = [
            archive_by_id.get(int(a.anchor_id), a).clone()
            for a in self.successful_anchors
        ]
        self.successful_region_ids = [
            int(a.anchor_id) for a in self.successful_anchors
        ]

        frontier_diag = self._update_frontier_campaign_after_archive()
        archive_stats = self.anchor_archive_statistics()
        selection = getattr(self, "_selection_diagnostics", {})
        selected_scores = selection.get("selected_scores", [])
        selected_g = selection.get("selected_logdet_novelty", [])
        selected_d = selection.get("selected_distance_novelty", [])

        unique_phi = np.array([anchor.phi for anchor in unique], dtype=float)
        self.last_archive_diagnostics = {
            "round_index": int(self.round_index),
            "phase": completed_phase,
            "phase_index": completed_phase_index,
            "cycle": completed_cycle,
            "stage_length": completed_stage_length,
            "candidate_count_raw": int(raw_count),
            "candidate_count_unique": int(len(unique)),
            "candidate_count_quality_eligible": int(len(eligible)),
            "candidate_count_distance_eligible": int(
                selection.get("distance_eligible_initial", 0)
            ),
            "candidate_phi_min": (
                np.nan if unique_phi.size == 0 else float(np.min(unique_phi))
            ),
            "candidate_phi_median": (
                np.nan if unique_phi.size == 0 else float(np.median(unique_phi))
            ),
            "candidate_phi_max": (
                np.nan if unique_phi.size == 0 else float(np.max(unique_phi))
            ),
            "rejected_below_score_count": int(
                selection.get("score_rejected_final", 0)
            ),
            "diversity_alpha": float(self.diversity_alpha),
            "minimum_diversity_score": float(self.minimum_diversity_score),
            "selected_score_min": (
                np.nan if not selected_scores else float(np.min(selected_scores))
            ),
            "selected_score_mean": (
                np.nan if not selected_scores else float(np.mean(selected_scores))
            ),
            "selected_score_max": (
                np.nan if not selected_scores else float(np.max(selected_scores))
            ),
            "selected_logdet_novelty_mean": (
                np.nan if not selected_g else float(np.mean(selected_g))
            ),
            "selected_distance_novelty_mean": (
                np.nan if not selected_d else float(np.mean(selected_d))
            ),
            "archive_hit_cap": int(bool(selection.get("archive_hit_cap", False))),
            "protected_frontier_retained": int(
                bool(selection.get("protected_frontier_retained", False))
            ),
            "protected_successful_count": int(
                selection.get("protected_successful_count", len(self.successful_anchors))
            ),
            "best_phi_previous": previous_best_phi,
            "best_phi_current": float(self.best_anchor.phi),
            "best_phi_improvement": (
                np.nan
                if not np.isfinite(previous_best_phi)
                else float(previous_best_phi - self.best_anchor.phi)
            ),
            "best_source_role": str(self.best_anchor.source_role),
            **archive_stats,
            **frontier_diag,
        }
        self.round_index += 1
        return [anchor.clone() for anchor in self.anchors]

    def _gather_equilibrium_diagnostics_mpi(
        self,
        *,
        current_phi: float,
        rank: int,
        comm: Any,
    ) -> dict[str, Any]:
        """Estimate whether restart locations are below chain equilibrium.

        For each chain, delta_phi = phi_end - phi_restart. Positive drift means
        the chain moved uphill from the restart state. Once this behavior is
        common across the population, later allocation shifts effort away from
        concentrated exploitation and toward geometrically dispersed anchors.
        """
        restart_phi = float(self.current_restart_phi)
        end_phi = float(current_phi)
        valid = bool(np.isfinite(restart_phi) and np.isfinite(end_phi))
        local = {
            "valid": int(valid),
            "restart_phi": restart_phi,
            "end_phi": end_phi,
            "drift": (end_phi - restart_phi) if valid else np.nan,
            "role": str(self.current_chain_role),
            "source_anchor_id": int(self.current_source_anchor_id),
        }
        gathered = comm.gather(local, root=0)
        if rank != 0:
            return {}

        rows = [row for row in gathered if row["valid"]]
        if len(rows) < self.equilibrium_min_observations:
            return {
                "equilibrium_observation_count": int(len(rows)),
                "equilibrium_positive_drift_fraction": np.nan,
                "equilibrium_median_drift": np.nan,
                "equilibrium_median_end_phi": np.nan,
                "equilibrium_median_restart_phi": np.nan,
                "equilibrium_pressure": 0.0,
            }

        drifts = np.asarray([row["drift"] for row in rows], dtype=float)
        end_values = np.asarray([row["end_phi"] for row in rows], dtype=float)
        restart_values = np.asarray([row["restart_phi"] for row in rows], dtype=float)
        positive_fraction = float(np.mean(drifts > self.equilibrium_drift_tolerance))
        median_drift = float(np.median(drifts))
        threshold = self.equilibrium_positive_fraction_threshold
        if (
            not self.equilibrium_allocation_enabled
            or median_drift <= self.equilibrium_drift_tolerance
            or positive_fraction <= threshold
        ):
            pressure = 0.0
        else:
            pressure = float(np.clip(
                (positive_fraction - threshold) / max(1.0 - threshold, 1.0e-12),
                0.0, 1.0,
            ))

        return {
            "equilibrium_observation_count": int(len(rows)),
            "equilibrium_positive_drift_fraction": positive_fraction,
            "equilibrium_median_drift": median_drift,
            "equilibrium_median_end_phi": float(np.median(end_values)),
            "equilibrium_median_restart_phi": float(np.median(restart_values)),
            "equilibrium_pressure": pressure,
        }

    def rebuild_anchor_set_mpi(
        self,
        *,
        current_theta: Sequence[np.ndarray],
        current_phi: float,
        current_simuldata: Any,
        rank: int,
        comm: Any,
    ) -> list[Anchor]:
        candidates = self.gather_candidates_mpi(
            current_theta=current_theta,
            current_phi=current_phi,
            current_simuldata=current_simuldata,
            rank=rank,
            comm=comm,
        )
        equilibrium_diag = self._gather_equilibrium_diagnostics_mpi(
            current_phi=current_phi, rank=rank, comm=comm
        )
        if rank == 0:
            self.last_equilibrium_diagnostics = dict(equilibrium_diag)
            anchors = self.build_anchor_set(candidates or [])
            payload = {
                "anchors": anchors,
                "best_anchor": self.best_anchor.clone(),
                "next_anchor_id": self._next_anchor_id,
                "frontier_anchor_id": self.frontier_anchor_id,
                "frontier_start_phi": self.frontier_start_phi,
                "frontier_previous_phi": self.frontier_previous_phi,
                "frontier_best_phi": self.frontier_best_phi,
                "frontier_rounds": self.frontier_rounds,
                "frontier_status": self.frontier_status,
                "frontier_cooldown_until": dict(self.frontier_cooldown_until),
                "successful_region_ids": list(self.successful_region_ids),
                "successful_anchors": [a.clone() for a in self.successful_anchors],
                "round_index": self.round_index,
                "archive_diagnostics": dict(self.last_archive_diagnostics),
                "equilibrium_diagnostics": dict(self.last_equilibrium_diagnostics),
            }
        else:
            payload = None
        payload = comm.bcast(payload, root=0)
        self.anchors = [anchor.clone() for anchor in payload["anchors"]]
        self.best_anchor = payload["best_anchor"].clone()
        self._next_anchor_id = int(payload["next_anchor_id"])
        self.frontier_anchor_id = payload["frontier_anchor_id"]
        self.frontier_start_phi = payload["frontier_start_phi"]
        self.frontier_previous_phi = payload["frontier_previous_phi"]
        self.frontier_best_phi = payload["frontier_best_phi"]
        self.frontier_rounds = int(payload["frontier_rounds"])
        self.frontier_status = str(payload["frontier_status"])
        self.frontier_cooldown_until = dict(payload["frontier_cooldown_until"])
        self.successful_region_ids = list(payload["successful_region_ids"])
        self.successful_anchors = [
            a.clone() for a in payload.get("successful_anchors", [])
        ]
        self.round_index = int(payload["round_index"])
        self.last_archive_diagnostics = dict(payload["archive_diagnostics"])
        self.last_equilibrium_diagnostics = dict(
            payload.get("equilibrium_diagnostics", {})
        )
        return [anchor.clone() for anchor in self.anchors]

    @staticmethod
    def _fractional_counts(number_of_chains: int, fractions: Sequence[float]) -> list[int]:
        raw = np.asarray(fractions, dtype=float) * int(number_of_chains)
        counts = np.floor(raw).astype(int)
        remainder = int(number_of_chains) - int(np.sum(counts))
        if remainder > 0:
            order = np.argsort(-(raw - counts))
            for index in order[:remainder]:
                counts[index] += 1
        return counts.tolist()

    def _equilibrium_anchor_index(self, best_index: int) -> int:
        """Choose an archive representative near empirical equilibrium phi."""
        pressure = float(self.last_equilibrium_diagnostics.get(
            "equilibrium_pressure", 0.0
        ))
        target = self.last_equilibrium_diagnostics.get(
            "equilibrium_median_end_phi", np.nan
        )
        if pressure <= 0.0 or not np.isfinite(target):
            return int(best_index)
        return int(np.argmin([
            abs(float(anchor.phi) - float(target))
            for anchor in self.anchors
        ]))

    def _explore_selection_weights(
        self,
        available: Sequence[int],
        selected_explore: Sequence[int],
        reference_indices: Sequence[int],
        pressure: float,
    ) -> np.ndarray:
        """Blend visit balancing with nearest-reference distance."""
        available = list(available)
        if not available:
            return np.empty(0, dtype=float)

        visit = np.asarray([
            1.0 / (1.0 + self.anchors[i].exploration_visits)
            for i in available
        ], dtype=float)
        visit /= max(float(np.sum(visit)), np.finfo(float).eps)

        distance_bias = float(np.clip(
            pressure * self.equilibrium_distance_bias_max, 0.0, 1.0
        ))
        if distance_bias <= 0.0:
            return visit

        references = list(dict.fromkeys([
            *[int(i) for i in reference_indices],
            *[int(i) for i in selected_explore],
        ]))
        if not references:
            return visit

        reference_vectors = np.vstack([
            flatten_theta(self.anchors[i].theta, self.nstate, self.block_weights)
            for i in references
        ])
        candidate_vectors = np.vstack([
            flatten_theta(self.anchors[i].theta, self.nstate, self.block_weights)
            for i in available
        ])
        cand_norm = np.sum(candidate_vectors**2, axis=1)
        ref_norm = np.sum(reference_vectors**2, axis=1)
        distance_sq = (
            cand_norm[:, None] + ref_norm[None, :]
            - 2.0 * candidate_vectors @ reference_vectors.T
        )
        distance_sq = np.maximum(distance_sq, 0.0)
        nearest = np.sqrt(np.min(distance_sq, axis=1))

        d_min, d_max = float(np.min(nearest)), float(np.max(nearest))
        if d_max > d_min:
            novelty = (nearest - d_min) / (d_max - d_min)
        else:
            novelty = np.ones_like(nearest)
        novelty += 1.0e-12
        novelty /= float(np.sum(novelty))

        weights = (1.0 - distance_bias) * visit + distance_bias * novelty
        weights /= float(np.sum(weights))
        return weights

    def compute_role_allocation(
        self,
        number_of_chains: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Allocate chains with competitive-frontier and equilibrium feedback.

        When phase-end phi values tend to exceed the actual restart phi values,
        exploit effort is progressively transferred to explore. The first exact
        exploit copy still protects the best-ever state; remaining exploit chains
        are allowed to start from an anchor whose phi is closer to the empirical
        phase-end median. Explore anchors are selected with increasing geometric
        separation as the equilibrium pressure rises.
        """
        if not self.anchors:
            raise RuntimeError("Cannot allocate chains: anchor set is empty.")
        number_of_chains = int(number_of_chains)
        if number_of_chains <= 0:
            raise ValueError("number_of_chains must be positive.")

        pressure = float(np.clip(
            self.last_equilibrium_diagnostics.get("equilibrium_pressure", 0.0),
            0.0, 1.0,
        ))
        if not self.equilibrium_allocation_enabled:
            pressure = 0.0

        exploit_fraction_effective = self.exploit_fraction * (
            1.0 - self.equilibrium_dispersion_strength * pressure
        )
        released = self.exploit_fraction - exploit_fraction_effective
        frontier_fraction_effective = self.frontier_fraction
        explore_fraction_effective = self.explore_fraction + released

        n_exploit, n_frontier, n_explore = self._fractional_counts(
            number_of_chains,
            [exploit_fraction_effective, frontier_fraction_effective,
             explore_fraction_effective],
        )
        if n_exploit == 0:
            n_exploit = 1
            if n_explore > 0:
                n_explore -= 1
            elif n_frontier > 0:
                n_frontier -= 1

        best_index = int(np.argmin([anchor.phi for anchor in self.anchors]))
        equilibrium_index = self._equilibrium_anchor_index(best_index)
        frontier_index = next((
            i for i, anchor in enumerate(self.anchors)
            if self.frontier_anchor_id is not None
            and anchor.anchor_id == self.frontier_anchor_id
        ), None)

        excluded = {best_index, equilibrium_index}
        if frontier_index is not None:
            excluded.add(int(frontier_index))
        explore_pool = [
            i for i in range(len(self.anchors)) if i not in excluded
        ]

        if frontier_index is None:
            if explore_pool:
                n_explore += n_frontier
            else:
                n_exploit += n_frontier
            n_frontier = 0

        if not explore_pool and n_explore > 0:
            if frontier_index is not None:
                n_frontier += n_explore
            else:
                n_exploit += n_explore
            n_explore = 0

        allocation: list[int] = []
        roles: list[str] = []
        exact: list[bool] = []
        scales: list[float] = []

        for j in range(n_exploit):
            if self.preserve_exact_anchor and j == 0:
                exploit_index = best_index
                is_exact = True
            else:
                exploit_index = equilibrium_index
                is_exact = False
            allocation.append(int(exploit_index))
            roles.append("exploit")
            exact.append(is_exact)
            scales.append(0.0 if is_exact else self.exploit_restart_scale)

        if frontier_index is not None:
            for j in range(n_frontier):
                allocation.append(int(frontier_index))
                roles.append("frontier")
                exact.append(bool(self.preserve_frontier_exact and j == 0))
                scales.append(0.0 if exact[-1] else self.frontier_restart_scale)

        selected_explore: list[int] = []
        if n_explore > 0 and explore_pool:
            available = list(explore_pool)
            references = [best_index, equilibrium_index]
            if frontier_index is not None:
                references.append(int(frontier_index))

            while len(selected_explore) < n_explore:
                if available:
                    weights = self._explore_selection_weights(
                        available, selected_explore, references, pressure
                    )
                    local = int(self.rng.choice(len(available), p=weights))
                    selected_explore.append(available.pop(local))
                else:
                    weights = self._explore_selection_weights(
                        explore_pool, selected_explore, references, pressure
                    )
                    selected_explore.append(
                        int(self.rng.choice(explore_pool, p=weights))
                    )

            for index in selected_explore:
                self.anchors[index].exploration_visits += 1
                allocation.append(index)
                roles.append("explore")
                exact.append(False)
                scales.append(self.exploration_restart_scale)

        allocation_array = np.asarray(allocation, dtype=np.int64)
        role_array = np.asarray(roles, dtype=object)
        exact_array = np.asarray(exact, dtype=bool)
        scale_array = np.asarray(scales, dtype=float)
        if allocation_array.size != number_of_chains:
            raise RuntimeError(
                f"Role allocation produced {allocation_array.size} slots for "
                f"{number_of_chains} chains."
            )

        permutation = self.rng.permutation(number_of_chains)
        allocation_array = allocation_array[permutation]
        role_array = role_array[permutation]
        exact_array = exact_array[permutation]
        scale_array = scale_array[permutation]

        occupancy = anchor_occupancy_statistics(
            allocation_array, len(self.anchors)
        )
        self.last_allocation_diagnostics = {
            "n_exploit_chains": int(np.sum(role_array == "exploit")),
            "n_frontier_chains": int(np.sum(role_array == "frontier")),
            "n_explore_chains": int(np.sum(role_array == "explore")),
            "n_unique_explore_anchors": int(len(set(selected_explore))),
            "explore_archive_fraction": float(
                len(set(selected_explore)) / max(len(explore_pool), 1)
            ),
            "best_anchor_exact_copy": int(
                np.any((role_array == "exploit") & exact_array)
            ),
            "frontier_exact_copy": int(
                np.any((role_array == "frontier") & exact_array)
            ),
            "effective_occupied_anchors": float(
                occupancy["effective_occupied_anchors"]
            ),
            "normalized_occupancy_entropy": float(
                occupancy["normalized_occupancy_entropy"]
            ),
            "equilibrium_pressure": pressure,
            "equilibrium_anchor_index": int(equilibrium_index),
            "equilibrium_anchor_phi": float(self.anchors[equilibrium_index].phi),
            "effective_exploit_fraction": float(exploit_fraction_effective),
            "effective_frontier_fraction": float(frontier_fraction_effective),
            "effective_explore_fraction": float(explore_fraction_effective),
            "effective_explore_distance_bias": float(
                pressure * self.equilibrium_distance_bias_max
            ),
        }
        return allocation_array, role_array, exact_array, scale_array

    # Legacy wrappers retained so older driver utilities do not fail if called.
    def select_active_anchor_indices(self, number_of_chains: int) -> NDArray[np.int64]:
        return np.arange(min(len(self.anchors), int(number_of_chains)), dtype=np.int64)

    def compute_anchor_allocation(
        self,
        number_of_chains: int,
        active_anchor_indices: Sequence[int] | None = None,
    ) -> NDArray[np.int64]:
        allocation, _, _, _ = self.compute_role_allocation(number_of_chains)
        return allocation

    def _active_restart_subdomains(
        self,
        rank: int,
    ) -> list[int]:
        return self.active_subdomain_indices(
            rank
        ).tolist()

    def _draw_perturbation(self, component: np.ndarray, subdomain: int, scale: float) -> np.ndarray:
        component = np.asarray(component, dtype=float)
        if self.restart_basis is None:
            return scale * self.rng.standard_normal(component.shape)

        basis = self.restart_basis[subdomain]
        noise = self.rng.standard_normal(basis.shape[1])
        perturbation = scale * (basis @ noise)
        return np.asarray(perturbation).reshape(component.shape)

    def propose_around_anchor(
        self,
        anchor: Anchor,
        *,
        rank: int,
        scale: float,
        exact_copy: bool = False,
    ) -> Theta:
        trial = copy_theta(anchor.theta)
        if exact_copy or scale == 0.0:
            return trial

        for subdomain in self._active_restart_subdomains(rank):
            trial[subdomain] = (
                np.asarray(anchor.theta[subdomain], dtype=float)
                + self._draw_perturbation(anchor.theta[subdomain], subdomain, scale)
            )
        return trial

    @staticmethod
    def evaluate_theta(
        theta: Sequence[np.ndarray],
        *,
        field_builder: FieldBuilder,
        simulation_fn: SimulationFunction,
        phi_fn: PhiFunction,
        inputdata: Any,
        rank: int,
        datafiletype: Any,
    ) -> tuple[Any, Any, float]:
        """
        Evaluate one state using an absolute archive potential.

        ``phi_fn`` receives the parameter state, the field payload returned by
        ``field_builder``, and the simulated data. This allows archive quality
        to include both the likelihood and the absolute global-prior
        correction, rather than a transition-specific prior ratio.
        """
        theta_copy = copy_theta(theta)
        field_payload = field_builder(theta_copy)

        simulation_field = (
            field_payload.simulation_field
            if isinstance(field_payload, BuiltField)
            else field_payload
        )

        simuldata = simulation_fn(
            inputdata,
            simulation_field,
            int(rank),
            datafiletype,
        )
        phi = float(phi_fn(theta_copy, field_payload, simuldata))
        return field_payload, simuldata, phi

    def _random_reach(
        self,
        *,
        start_theta: Sequence[np.ndarray],
        start_field: Any,
        start_simuldata: Any,
        start_phi: float,
        target_phi: float,
        rank: int,
        field_builder: FieldBuilder,
        simulation_fn: SimulationFunction,
        phi_fn: PhiFunction,
        inputdata: Any,
        datafiletype: Any,
    ) -> tuple[Theta, Any, Any, float, int, float, bool]:
        """Greedy random search until the restart threshold is reached."""
        best_theta = copy_theta(start_theta)
        best_field = start_field
        best_simuldata = start_simuldata
        best_phi = float(start_phi)
        scale = float(self.random_reach_initial_scale)

        for attempt in range(1, self.random_reach_attempts + 1):
            trial = copy_theta(best_theta)
            for subdomain in self._active_restart_subdomains(rank):
                trial[subdomain] = (
                    np.asarray(best_theta[subdomain], dtype=float)
                    + self._draw_perturbation(best_theta[subdomain], subdomain, scale)
                )

            field_payload, simuldata, phi = self.evaluate_theta(
                trial,
                field_builder=field_builder,
                simulation_fn=simulation_fn,
                phi_fn=phi_fn,
                inputdata=inputdata,
                rank=rank,
                datafiletype=datafiletype,
            )

            if phi < best_phi:
                best_theta = trial
                best_field = field_payload
                best_simuldata = simuldata
                best_phi = phi

            if best_phi <= target_phi:
                return (
                    best_theta,
                    best_field,
                    best_simuldata,
                    best_phi,
                    attempt,
                    scale,
                    True,
                )

            scale *= self.random_reach_decay

        return (
            best_theta,
            best_field,
            best_simuldata,
            best_phi,
            self.random_reach_attempts,
            scale,
            False,
        )

    def restart_one_chain(
        self,
        *,
        anchor_index: int,
        rank: int,
        exact_anchor_copy: bool,
        role: str,
        restart_scale: float,
        field_builder: FieldBuilder,
        simulation_fn: SimulationFunction,
        phi_fn: PhiFunction,
        inputdata: Any,
        datafiletype: Any,
    ) -> RestartResult:
        if not self.anchors:
            raise RuntimeError("Cannot restart: anchor set is empty.")

        anchor = self.anchors[int(anchor_index)]
        target_phi = anchor.phi + self.restart_phi_tolerance

        if exact_anchor_copy:
            # IMPORTANT FOR MPI SAFETY:
            # Even an exact anchor restart must enter the normal forward-model
            # evaluation path. Some GLAM forward solvers contain MPI collectives,
            # so allowing exact-copy ranks to skip simulation_fn while other ranks
            # evaluate perturbed restarts can deadlock the population at exchange.
            #
            # We still keep the archive lightweight: simuldata is NOT stored in
            # Anchor. It is reconstructed here only for the newly installed chain
            # state and returned transiently through RestartResult.
            theta_copy = copy_theta(anchor.theta)
            field_payload, simuldata, phi = self.evaluate_theta(
                theta_copy,
                field_builder=field_builder,
                simulation_fn=simulation_fn,
                phi_fn=phi_fn,
                inputdata=inputdata,
                rank=rank,
                datafiletype=datafiletype,
            )
            return RestartResult(
                theta=theta_copy,
                field=field_payload,
                simuldata=simuldata,
                phi=phi,
                anchor_index=int(anchor_index),
                anchor_phi=float(anchor.phi),
                exact_anchor_copy=True,
                accepted_under_threshold=(phi <= target_phi),
                attempts=1,
                final_scale=0.0,
                role=str(role),
                source_anchor_id=int(anchor.anchor_id),
            )

        scale = float(restart_scale)
        best_theta: Theta | None = None
        best_field: Any = None
        best_simuldata: Any = None
        best_phi = np.inf
        tried_phis: list[float] = []

        for attempt in range(1, self.max_restart_attempts + 1):
            trial = self.propose_around_anchor(anchor, rank=rank, scale=scale)
            field_payload, simuldata, phi = self.evaluate_theta(
                trial,
                field_builder=field_builder,
                simulation_fn=simulation_fn,
                phi_fn=phi_fn,
                inputdata=inputdata,
                rank=rank,
                datafiletype=datafiletype,
            )

            tried_phis.append(float(phi))
            if phi < best_phi:
                best_theta = trial
                best_field = field_payload
                best_simuldata = simuldata
                best_phi = phi

            if phi <= target_phi:
                return RestartResult(
                    theta=trial,
                    field=field_payload,
                    simuldata=simuldata,
                    phi=phi,
                    anchor_index=int(anchor_index),
                    anchor_phi=float(anchor.phi),
                    exact_anchor_copy=False,
                    accepted_under_threshold=True,
                    attempts=attempt,
                    final_scale=scale,
                    role=str(role),
                    source_anchor_id=int(anchor.anchor_id),
                )

            scale *= self.restart_scale_decay

        if best_theta is None:
            # Every perturbed trial had a non-finite potential. Fall back to the anchor itself (it was evaluated
            # finite when it entered the archive); if that is also non-finite, say exactly what was seen.
            theta_copy = copy_theta(anchor.theta)
            field_payload, simuldata, phi = self.evaluate_theta(
                theta_copy, field_builder=field_builder, simulation_fn=simulation_fn, phi_fn=phi_fn,
                inputdata=inputdata, rank=rank, datafiletype=datafiletype,
            )
            if not np.isfinite(phi):
                last = [float(np.ravel(b)[0]) if np.size(b) else float("nan") for b in anchor.theta]
                raise RuntimeError(
                    "Restart search produced no evaluated state: rank %d, anchor %d (stored phi %r), scale %g, "
                    "%d trial potentials %r, re-evaluated anchor phi %r, first entry of each anchor block %r"
                    % (rank, int(anchor_index), anchor.phi, restart_scale, len(tried_phis), tried_phis[:6], phi, last)
                )
            self.restart_fallbacks = getattr(self, "restart_fallbacks", 0) + 1
            return RestartResult(
                theta=theta_copy, field=field_payload, simuldata=simuldata, phi=phi,
                anchor_index=int(anchor_index), anchor_phi=float(anchor.phi), exact_anchor_copy=True,
                accepted_under_threshold=(phi <= target_phi), attempts=len(tried_phis), final_scale=0.0,
                role=str(role), source_anchor_id=int(anchor.anchor_id),
            )

        if self.random_reach_attempts == 0:

            if best_phi <= target_phi:
                return RestartResult(
                    theta=best_theta,
                    field=best_field,
                    simuldata=best_simuldata,
                    phi=float(best_phi),
                    anchor_index=int(anchor_index),
                    anchor_phi=float(anchor.phi),
                    exact_anchor_copy=False,
                    accepted_under_threshold=True,
                    attempts=self.max_restart_attempts,
                    final_scale=scale,
                    role=str(role),
                    source_anchor_id=int(anchor.anchor_id),
                )

            theta_fallback = copy_theta(anchor.theta)

            fallback_field, fallback_simuldata, fallback_phi = self.evaluate_theta(
                theta_fallback,
                field_builder=field_builder,
                simulation_fn=simulation_fn,
                phi_fn=phi_fn,
                inputdata=inputdata,
                rank=rank,
                datafiletype=datafiletype,
            )

            return RestartResult(
                theta=theta_fallback,
                field=fallback_field,
                simuldata=fallback_simuldata,
                phi=float(fallback_phi),
                anchor_index=int(anchor_index),
                anchor_phi=float(anchor.phi),
                exact_anchor_copy=True,
                accepted_under_threshold=(fallback_phi <= target_phi),
                attempts=self.max_restart_attempts + 1,
                final_scale=0.0,
                role=str(role),
                source_anchor_id=int(anchor.anchor_id),
            )

        (
            reach_theta,
            reach_field,
            reach_simuldata,
            reach_phi,
            reach_attempts,
            reach_scale,
            success,
        ) = self._random_reach(
            start_theta=best_theta,
            start_field=best_field,
            start_simuldata=best_simuldata,
            start_phi=best_phi,
            target_phi=target_phi,
            rank=rank,
            field_builder=field_builder,
            simulation_fn=simulation_fn,
            phi_fn=phi_fn,
            inputdata=inputdata,
            datafiletype=datafiletype,
        )

        total_attempts = self.max_restart_attempts + reach_attempts

        # ------------------------------------------------------------------
        # Safe anchor fallback.
        #
        # If neither the bounded restart search nor random reach found a state
        # satisfying the restart-quality threshold, do not install the poor
        # failed state. Re-evaluate the source anchor through the normal
        # forward-model path and restart exactly from that anchor.
        #
        # IMPORTANT: we still call evaluate_theta() because the forward solver
        # may contain MPI collectives and Anchor does not store simuldata.
        # ------------------------------------------------------------------
        if not success:

            theta_fallback = copy_theta(anchor.theta)

            fallback_field, fallback_simuldata, fallback_phi = self.evaluate_theta(
                theta_fallback,
                field_builder=field_builder,
                simulation_fn=simulation_fn,
                phi_fn=phi_fn,
                inputdata=inputdata,
                rank=rank,
                datafiletype=datafiletype,
            )

            return RestartResult(
                theta=theta_fallback,
                field=fallback_field,
                simuldata=fallback_simuldata,
                phi=float(fallback_phi),
                anchor_index=int(anchor_index),
                anchor_phi=float(anchor.phi),
                exact_anchor_copy=True,
                accepted_under_threshold=(fallback_phi <= target_phi),
                attempts=total_attempts + 1,
                final_scale=0.0,
                role=str(role),
                source_anchor_id=int(anchor.anchor_id),
            )

        return RestartResult(
            theta=reach_theta,
            field=reach_field,
            simuldata=reach_simuldata,
            phi=float(reach_phi),
            anchor_index=int(anchor_index),
            anchor_phi=float(anchor.phi),
            exact_anchor_copy=False,
            accepted_under_threshold=True,
            attempts=total_attempts,
            final_scale=reach_scale,
            role=str(role),
            source_anchor_id=int(anchor.anchor_id),
        )

    def restart_population_mpi(
        self,
        *,
        rank: int,
        number_of_chains: int,
        comm: Any,
        field_builder: FieldBuilder,
        simulation_fn: SimulationFunction,
        phi_fn: PhiFunction,
        inputdata: Any,
        datafiletype: Any,
    ) -> RestartResult:
        if rank == 0:
            allocation, roles, exact_flags, restart_scales = (
                self.compute_role_allocation(number_of_chains)
            )
            payload = {
                "anchors": [anchor.clone() for anchor in self.anchors],
                "allocation": allocation,
                "roles": roles.tolist(),
                "exact_flags": exact_flags,
                "restart_scales": restart_scales,
                "allocation_diagnostics": dict(self.last_allocation_diagnostics),
            }
        else:
            payload = None

        payload = comm.bcast(payload, root=0)
        # rebuild_anchor_set_mpi already synchronized self.anchors on all ranks.
        self.last_allocation_diagnostics = dict(payload["allocation_diagnostics"])

        anchor_index = int(payload["allocation"][rank])
        role = str(payload["roles"][rank])
        exact_copy = bool(payload["exact_flags"][rank])
        restart_scale = float(payload["restart_scales"][rank])

        result = self.restart_one_chain(
            anchor_index=anchor_index,
            rank=rank,
            exact_anchor_copy=exact_copy,
            role=role,
            restart_scale=restart_scale,
            field_builder=field_builder,
            simulation_fn=simulation_fn,
            phi_fn=phi_fn,
            inputdata=inputdata,
            datafiletype=datafiletype,
        )

        self.current_chain_role = str(result.role)
        self.current_source_anchor_id = int(result.source_anchor_id)
        self.current_restart_phi = float(result.phi)
        self.current_restart_anchor_phi = float(result.anchor_phi)
        self.initialize_phase_best(result.theta, result.phi, result.simuldata)

        local_restart_diag = {
            "role": str(result.role),
            "success": int(result.accepted_under_threshold),
            "attempts": int(result.attempts),
            "final_scale": float(result.final_scale),
            "used_random_reach": int(result.attempts > self.max_restart_attempts),
        }
        gathered = comm.gather(local_restart_diag, root=0)
        if rank == 0:
            summary: dict[str, Any] = {}
            for role_name in ("exploit", "frontier", "explore"):
                rows = [row for row in gathered if row["role"] == role_name]
                prefix = role_name
                if rows:
                    summary[f"{prefix}_restart_success_rate"] = float(
                        np.mean([row["success"] for row in rows])
                    )
                    summary[f"{prefix}_restart_attempts_mean"] = float(
                        np.mean([row["attempts"] for row in rows])
                    )
                    summary[f"{prefix}_restart_final_scale_mean"] = float(
                        np.mean([row["final_scale"] for row in rows])
                    )
                else:
                    summary[f"{prefix}_restart_success_rate"] = np.nan
                    summary[f"{prefix}_restart_attempts_mean"] = np.nan
                    summary[f"{prefix}_restart_final_scale_mean"] = np.nan
            summary["random_reach_count"] = int(
                np.sum([row["used_random_reach"] for row in gathered])
            )
            summary["restart_attempts_max"] = int(max(row["attempts"] for row in gathered))
        else:
            summary = None
        summary = comm.bcast(summary, root=0)
        self.last_restart_diagnostics = dict(summary)
        return result

    # ------------------------------------------------------------------
    # Anchor-set snapshots for robust restart
    # ------------------------------------------------------------------
    _GLOBAL_STATE_KEYS = (
        "anchors", "best_anchor", "successful_anchors", "successful_region_ids",
        "successful_region_aliases", "successful_region_merge_history",
        "frontier_anchor_id", "frontier_start_phi", "frontier_previous_phi",
        "frontier_best_phi", "frontier_rounds", "frontier_status",
        "frontier_cooldown_until", "_next_anchor_id", "round_index",
        "phase_index", "cycle", "phase_start_iteration", "phase_stop_iteration",
        "last_archive_diagnostics", "last_allocation_diagnostics",
        "last_restart_diagnostics", "last_equilibrium_diagnostics",
    )
    _STATE_FILE_RE = re.compile(r"^anchor_state_ite(\d+)_(global|rank\d+)\.pkl$")

    def get_global_state(self, comm_size: int) -> dict[str, Any]:
        """Schedule + archive + frontier/successful-region state (rank 0 view)."""
        state = {key: copy.deepcopy(getattr(self, key))
                 for key in self._GLOBAL_STATE_KEYS}
        state["phase"] = self.phase.name
        state["numsubd"] = int(self.numsubd)
        state["comm_size"] = int(comm_size)
        state["format_version"] = 1
        return state

    def set_global_state(self, state: dict[str, Any]) -> None:
        for key in self._GLOBAL_STATE_KEYS:
            setattr(self, key, copy.deepcopy(state[key]))
        self.phase = Phase[state["phase"]]

    def get_rank_state(self) -> dict[str, Any]:
        """Per-chain state: restart role/lineage and the manager RNG stream."""
        return {
            "round_index": int(self.round_index),
            "current_chain_role": str(self.current_chain_role),
            "current_source_anchor_id": int(self.current_source_anchor_id),
            "current_restart_phi": float(self.current_restart_phi),
            "current_restart_anchor_phi": float(self.current_restart_anchor_phi),
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
        }

    def set_rank_state(self, state: dict[str, Any]) -> None:
        self.current_chain_role = str(state["current_chain_role"])
        self.current_source_anchor_id = int(state["current_source_anchor_id"])
        self.current_restart_phi = float(state["current_restart_phi"])
        self.current_restart_anchor_phi = float(state["current_restart_anchor_phi"])
        self.rng.bit_generator.state = state["rng_state"]

    @staticmethod
    def _atomic_pickle(path: str, obj: Any) -> None:
        tmp = path + ".tmp"
        with open(tmp, "wb") as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)

    def _state_generations(self, directory: str, kind: str) -> set[int]:
        found: set[int] = set()
        if not os.path.isdir(directory):
            return found
        for name in os.listdir(directory):
            match = self._STATE_FILE_RE.match(name)
            if match and match.group(2) == kind:
                found.add(int(match.group(1)))
        return found

    def save_anchor_state(
        self,
        directory: str,
        *,
        iteration: int,
        rank: int,
        comm: Any,
        keep: int = 3,
    ) -> None:
        """Collective. Save the anchor set and manager state for ``iteration``.

        Call after the chains were assigned to their new anchors. Each rank
        writes ``..._rank###.pkl``; rank 0 writes ``..._global.pkl``. Files are
        written atomically, and older generations are pruned only after every
        rank has finished writing, so the newest ``keep`` generations are
        always complete.
        """
        os.makedirs(directory, exist_ok=True)
        tag = f"anchor_state_ite{int(iteration):09d}"
        self._atomic_pickle(
            os.path.join(directory, f"{tag}_rank{int(rank):03d}.pkl"),
            self.get_rank_state(),
        )
        if rank == 0:
            self._atomic_pickle(
                os.path.join(directory, f"{tag}_global.pkl"),
                self.get_global_state(comm.Get_size()),
            )
        comm.Barrier()

        kind = "global" if rank == 0 else f"rank{int(rank):03d}"
        generations = sorted(self._state_generations(directory, kind))
        for old in generations[:-max(int(keep), 1)]:
            for suffix in {kind, f"rank{int(rank):03d}"}:
                path = os.path.join(
                    directory, f"anchor_state_ite{old:09d}_{suffix}.pkl")
                if os.path.exists(path):
                    os.remove(path)

    def load_anchor_state(
        self,
        directory: str,
        *,
        max_iteration: int,
        rank: int,
        comm: Any,
    ) -> int | None:
        """Collective. Restore the newest complete snapshot with
        ``iteration <= max_iteration``; return its iteration, or None.

        ``max_iteration`` should be the last iteration every chain completed
        before the interruption, so the restored schedule is never ahead of
        the restored chains. A generation is usable only if rank 0's global
        file and every rank's own file exist.
        """
        own = self._state_generations(directory, f"rank{int(rank):03d}")
        gathered = comm.gather(own, root=0)
        chosen = None
        if rank == 0:
            common = self._state_generations(directory, "global")
            for generations in gathered:
                common &= generations
            usable = [i for i in common if i <= int(max_iteration)]
            chosen = max(usable) if usable else None
        chosen = comm.bcast(chosen, root=0)
        if chosen is None:
            return None

        tag = f"anchor_state_ite{int(chosen):09d}"
        payload = None
        if rank == 0:
            with open(os.path.join(directory, f"{tag}_global.pkl"), "rb") as h:
                payload = pickle.load(h)
        payload = comm.bcast(payload, root=0)
        if payload["numsubd"] != self.numsubd or \
                payload["comm_size"] != comm.Get_size():
            raise RuntimeError(
                "Anchor-state snapshot does not match this run: saved "
                f"numsubd={payload['numsubd']}, comm_size={payload['comm_size']}; "
                f"current numsubd={self.numsubd}, comm_size={comm.Get_size()}."
            )
        self.set_global_state(payload)
        with open(os.path.join(directory, f"{tag}_rank{int(rank):03d}.pkl"), "rb") as h:
            rank_state = pickle.load(h)
        if int(rank_state["round_index"]) != int(self.round_index):
            raise RuntimeError(
                f"Anchor-state rank file for rank {rank} is from a different "
                f"round than the global file (iteration {chosen})."
            )
        self.set_rank_state(rank_state)
        return int(chosen)

    def monitoring_row(
        self,
        *,
        iteration: int,
        number_of_chains: int,
    ) -> dict[str, Any]:
        """Return one flat CSV-ready row for the most recent GLAM exchange."""
        row: dict[str, Any] = {
            "iteration": int(iteration),
            "number_of_chains": int(number_of_chains),
        }
        row.update(self.last_archive_diagnostics)
        row.update(self.last_equilibrium_diagnostics)
        row.update(self.last_allocation_diagnostics)
        row.update(self.last_restart_diagnostics)

        if self.relative_error_fn is not None and self.anchors:
            best = min(self.anchors, key=lambda anchor: anchor.phi)
            row["best_anchor_re"] = float(self.relative_error_fn(best.theta))
            frontier = next(
                (
                    anchor
                    for anchor in self.anchors
                    if self.frontier_anchor_id is not None
                    and anchor.anchor_id == self.frontier_anchor_id
                ),
                None,
            )
            row["frontier_re"] = (
                np.nan
                if frontier is None
                else float(self.relative_error_fn(frontier.theta))
            )
        return row

    def append_monitoring_csv(
        self,
        filename: str | Path,
        *,
        iteration: int,
        number_of_chains: int,
    ) -> None:
        """Append the most recent exchange diagnostics to a CSV file."""
        append_csv_row(
            filename,
            self.monitoring_row(
                iteration=iteration,
                number_of_chains=number_of_chains,
            ),
        )

    def append_abridged_monitoring_csv(
        self,
        filename: str | Path,
        *,
        iteration: int,
        number_of_chains: int,
    ) -> None:
        """Append compact archive geometry, phi, and successful-region diagnostics."""
        full = self.monitoring_row(
            iteration=iteration,
            number_of_chains=number_of_chains,
        )
        fields = [
            "iteration", "round_index", "phase", "anchor_count",
            "successful_region_count", "new_successful_regions_this_round",
            "successful_region_updates_this_round",
            "successful_regions_merged_this_round",
            "successful_region_count_before_merge",
            "successful_region_count_after_merge",
            "successful_region_creation_distance",
            "successful_region_merge_distance",
            "anchor_cov_trace", "anchor_cov_largest_eigenvalue",
            "anchor_cov_largest_eigenvalue_fraction",
            "anchor_effective_rank", "logdet_Ck",
            "distance_min", "distance_mean", "distance_median", "distance_max",
            "anchor_phi_best", "anchor_phi_q25", "anchor_phi_median",
            "anchor_phi_mean", "anchor_phi_std", "anchor_phi_q75",
            "anchor_phi_max", "anchor_phi_gap_median", "anchor_phi_gap_max",
            "successful_phi_best", "successful_phi_median", "successful_phi_max",
            "successful_distance_min", "successful_distance_median",
            "frontier_active", "frontier_phi", "frontier_event",
            "frontier_relative_quality_gap", "frontier_reseeded",
            "equilibrium_observation_count",
            "equilibrium_positive_drift_fraction",
            "equilibrium_median_drift",
            "equilibrium_median_restart_phi",
            "equilibrium_median_end_phi",
            "equilibrium_pressure",
            "equilibrium_anchor_phi",
            "effective_exploit_fraction",
            "effective_frontier_fraction",
            "effective_explore_fraction",
            "effective_explore_distance_bias",
            "restart_attempts_max", "archive_exchange_wall_seconds_local",
            "restart_wall_seconds_local",
        ]
        append_csv_row(filename, {key: full.get(key, np.nan) for key in fields})

    def close_phase_and_restart_mpi(
        self,
        *,
        current_theta: Sequence[np.ndarray],
        current_phi: float,
        iteration: int,
        rank: int,
        number_of_chains: int,
        comm: Any,
        field_builder: FieldBuilder,
        simulation_fn: SimulationFunction,
        phi_fn: PhiFunction,
        inputdata: Any,
        datafiletype: Any,
        current_simuldata: Any | None = None,
        state_directory: str | None = None,
        state_keep: int = 3,
    ) -> tuple[RestartResult | None, list[np.ndarray]]:
        """
        Rebuild the archive, advance the schedule, and restart the population.

        During the initial burn-in this performs the first archive exchange.
        When the requested number of cycles has finished, no restart is made;
        ``None`` is returned together with an empty subdomain schedule.

        If ``state_directory`` is given, the completed anchor set and the
        manager state are saved (collectively) once the chains have been
        assigned to their new anchors, so an interrupted run can resume with
        ``load_anchor_state``.
        """
        t_archive = time.perf_counter()
        self.rebuild_anchor_set_mpi(
            current_theta=current_theta, current_phi=current_phi,
            current_simuldata=current_simuldata, rank=rank, comm=comm,
        )
        self.last_archive_diagnostics["archive_exchange_wall_seconds_local"] = float(
            time.perf_counter() - t_archive
        )

        self.advance_phase(next_iteration=int(iteration) + 1)

        if self.phase is Phase.FINISHED:
            if state_directory is not None:
                self.save_anchor_state(
                    state_directory, iteration=int(iteration), rank=rank,
                    comm=comm, keep=state_keep,
                )
            return None, []

        t_restart = time.perf_counter()
        result = self.restart_population_mpi(
            rank=rank, number_of_chains=number_of_chains, comm=comm,
            field_builder=field_builder, simulation_fn=simulation_fn, phi_fn=phi_fn,
            inputdata=inputdata, datafiletype=datafiletype,
        )
        self.last_restart_diagnostics["restart_wall_seconds_local"] = float(
            time.perf_counter() - t_restart
        )
        list_subd = self.subdomain_schedule(rank)
        if state_directory is not None:
            self.save_anchor_state(
                state_directory, iteration=int(iteration), rank=rank,
                comm=comm, keep=state_keep,
            )
        return result, list_subd


# ---------------------------------------------------------------------------
# Prototype splice-in sketch: stochastic diversity + exploit/frontier/explore
# ---------------------------------------------------------------------------
#
# manager = GlobalLocalAnchorManager(
#     numsubd=info_subdomains.numsubd,
#
#     initial_burnin_length=200,
#     global_stage_length=150,
#     local_stage_length=150,
#     minimum_global_stage_length=50,
#     minimum_local_stage_length=50,
#     global_stage_decay=0.80,
#     local_stage_decay=0.80,
#     disable_local_after_iteration=3000,
#     num_cycles=100,
#
#     # Hard archive screening.
#     quality_tolerance=15.0,
#     minimum_anchor_distance=2.0,
#     max_anchors=2 * NC,
#
#     # Stochastic soft-diversity archive.
#     archive_selection_mode="diversity",
#     covariance_regularization=1.0e-4,
#     diversity_alpha=0.70,
#     logdet_gain_scale=1.0,
#     minimum_diversity_score=0.20,
#     archive_selection_temperature=0.10,
#
#     # Exploit / frontier / explore chain fractions.
#     exploit_fraction=0.25,
#     frontier_fraction=0.40,
#     explore_fraction=0.35,
#
#     # Role-specific restart radii.
#     restart_scale=0.35,  # legacy fallback/default; role scales below are used
#     exploit_restart_scale=0.20,
#     frontier_restart_scale=0.35,
#     exploration_restart_scale=0.35,
#     restart_basis=None,
#
#     # Pilot-derived frontier target and persistent campaign controls.
#     frontier_target_phi=PHI_TARGET,
#     frontier_minimum_distance=2.0,
#     successful_region_distance=3.5,
#     successful_region_merge_distance=2.5,
#     frontier_max_rounds=3,
#     frontier_minimum_improvement=2.0,
#     frontier_cooldown_rounds=3,
#     frontier_relative_quality_tolerance=50.0,
#     frontier_reseed_interval=3,
#     frontier_quality_weight=0.60,
#     frontier_distance_weight=0.40,
#     preserve_frontier_exact=True,
#
#     # Keep one exact global-best copy; every other exploit/explore chain is
#     # perturbed. The frontier optionally receives one exact copy.
#     preserve_exact_anchor=True,
#
#     # Existing restart-quality safeguards.
#     restart_phi_tolerance=15.0,
#     max_restart_attempts=3,
#     restart_scale_decay=0.90,
#     random_reach_attempts=10,
#     random_reach_initial_scale=1.00,
#     random_reach_decay=0.97,
#
#     # Optional synthetic-problem diagnostic. The callback accepts theta and
#     # returns relative error. Leave None for real inverse problems.
#     relative_error_fn=None,
#
#     seed=int(nseed % 1000 + rank),
# )
#
# manager.initialize_phase_best(theta, error_glob, simuldata)
#
# # At each exchange, close_phase_and_restart_mpi() rebuilds the archive,
# # updates/promotes/retires the frontier campaign, allocates exploit/frontier/
# # explore roles, and performs role-specific restarts.
# result, list_subd = manager.close_phase_and_restart_mpi(...)
#
# # After the restart collective has completed, rank 0 can append one monitoring
# # row containing candidate filtering, archive geometry, frontier progress,
# # role allocation, and restart diagnostics.
# if rank == 0:
#     manager.append_monitoring_csv(
#         "anchor_round_summary.csv",
#         iteration=ite,
#         number_of_chains=NC,
#     )

