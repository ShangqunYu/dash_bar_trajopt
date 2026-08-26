"""Functional Bayesian optimization of control laws, via the vendored bofus.

Instead of searching a fixed parameter box (see :mod:`.search`), the optimizer
searches function space directly: candidates are RBF mixtures from phase
``s in [0, 1)`` to the four normalized joint targets, compared through their
ambient RKHS distance by a GP surrogate, with new candidates proposed by
expected improvement over both basis-point locations and amplitudes.

The search runs in stages that double the basis size: each stage splits every
atom in two (leaving the function unchanged) and doubles the observation
budget, so early stages explore a coarse basis cheaply and later stages refine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, Float

from dash_mjlab.trajopt.bar_env import ControlLaw
from dash_mjlab.trajopt.bofus import acquisition, gp, kernels, rkhs

N_JOINTS = 4


def as_control_law(f: rkhs.RBFMixture) -> ControlLaw:
  """Wrap a (k, m, d=1) mixture as a phase -> joint-targets control law."""

  def control_law(s: Float[np.ndarray, "... 1"]) -> Float[np.ndarray, "... 4"]:
    return np.clip(np.asarray(f(jnp.asarray(s))), -1.0, 1.0)

  return control_law


@dataclass
class FunctionalResult:
  best: rkhs.RBFMixture
  best_cost: float
  functions: rkhs.RBFMixture  # observation axis leading: (N, k, m, 1)
  costs: Float[np.ndarray, " N"]
  n_init: int


def _expand_to(
  fs: rkhs.RBFMixture, ys: Float[Array, " o"], size: int
) -> tuple[rkhs.RBFMixture, Float[Array, " n"]]:
  """Grow the observation axis to size, filling new slots inert."""

  def expand(z: Float[Array, " o ..."], fill: float) -> Float[Array, " n ..."]:
    o, *rest = z.shape
    return jnp.full((size, *rest), fill).at[:o].set(z)

  fills = rkhs.RBFMixture(l=1.0, x=0.0, a=0.0)
  return jax.tree.map(expand, fs, fills), expand(ys, jnp.nan)


def minimize_functional(
  objective: Callable[[ControlLaw], float],
  n_init: int = 8,
  m: int = 8,
  seed: int = 0,
  seeds: rkhs.RBFMixture | None = None,
  l_range: tuple[float, float] = (0.05**2, 0.5**2),
  a_range: tuple[float, float] = (-1.5, 1.5),
  on_step: Callable[[int, float], None] | None = None,
) -> FunctionalResult:
  """Staged EI loop doubling the basis size, each stage doubling the observations.

  Starts from an LHS design of ``n_init`` mixtures with the coarsest basis and
  runs stages up to ``m`` atoms; total evaluations are ``n_init * 2**n_stages``.
  ``seeds`` (batch shape ``(s, k, m0, 1)``, ``s <= n_init``) replaces the first
  ``s`` design points and sets the starting basis size ``m0``.
  """
  l_range = (jnp.asarray(l_range[0]), jnp.asarray(l_range[1]))
  x_range = (jnp.asarray(0.0), jnp.asarray(1.0))
  a_range = (jnp.asarray(a_range[0]), jnp.asarray(a_range[1]))
  key = jr.key(seed)

  # basis-size schedule: m0 doubling up to m
  m0 = 1 if seeds is None else seeds.a.shape[-1]
  schedule = [m0 << s for s in range((m // m0 - 1).bit_length())] + [m]

  # evaluated LHS design at the coarsest basis, seeds replacing the first rows
  key, key_init = jr.split(key)
  fs = rkhs.RBFMixture.from_lhs(
    key_init, (n_init, N_JOINTS, m0, 1), l_range, x_range, a_range
  )
  if seeds is not None:
    n_seeds = seeds.a.shape[0]
    fs = jax.tree.map(lambda z, sd: z.at[:n_seeds].set(sd), fs, seeds)

  def evaluate(i: int) -> float:
    cost = objective(as_control_law(jax.tree.map(lambda z: z[i], fs)))
    if on_step is not None:
      on_step(i, cost)
    return cost

  ys = jnp.full(n_init, jnp.nan)
  for i in range(n_init):
    ys = ys.at[i].set(evaluate(i))

  i = n_init
  for m_stage in schedule:
    # grow both buffers at the stage boundary, so each stage compiles once
    stage_end = 2 * i
    fs = fs if m_stage == fs.l.shape[-2] else fs.split()
    fs, ys = _expand_to(fs, ys, stage_end)
    while i < stage_end:
      surrogate = gp.GaussianProcess.fit(fs, ys, profile=kernels.matern52)
      key, key_ei = jr.split(key)
      candidate = acquisition.optimize_expected_improvement(
        key_ei, surrogate, l_range, x_range, a_range
      )
      fs = jax.tree.map(lambda z, c, i=i: z.at[i].set(c[0]), fs, candidate)
      ys = ys.at[i].set(evaluate(i))
      i += 1

  best = int(jnp.nanargmin(ys))
  return FunctionalResult(
    best=jax.tree.map(lambda z: z[best], fs),
    best_cost=float(ys[best]),
    functions=fs,
    costs=np.asarray(ys),
    n_init=n_init,
  )
