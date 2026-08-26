"""Functional Bayesian optimization of control laws, via the vendored bofus.

Instead of searching a fixed parameter box (see :mod:`.search`), the optimizer
searches function space directly: candidates are RBF mixtures from phase
``s in [0, 1)`` to the four normalized joint targets, compared through their
ambient RKHS distance by a GP surrogate, with new candidates proposed by
expected improvement over both basis-point locations and amplitudes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Float

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


def minimize_functional(
  objective: Callable[[ControlLaw], float],
  n_init: int = 16,
  n_iter: int = 48,
  m: int = 4,
  seed: int = 0,
  seeds: rkhs.RBFMixture | None = None,
  l_range: tuple[float, float] = (0.05**2, 0.5**2),
  a_range: tuple[float, float] = (-1.5, 1.5),
  on_step: Callable[[int, float], None] | None = None,
) -> FunctionalResult:
  """LHS design of ``n_init`` mixtures, then ``n_iter`` rounds of fit + EI.

  ``seeds`` (batch shape ``(s, k, m, 1)``, ``s <= n_init``) replaces the first
  ``s`` LHS points with known-good starting mixtures.
  """
  l_range = (jnp.asarray(l_range[0]), jnp.asarray(l_range[1]))
  x_range = (jnp.asarray(0.0), jnp.asarray(1.0))
  a_range = (jnp.asarray(a_range[0]), jnp.asarray(a_range[1]))
  key = jr.key(seed)

  # one padded buffer for the whole run, so fit/predict compile once
  n_total = n_init + n_iter
  key, key_init = jr.split(key)
  init = rkhs.RBFMixture.from_lhs(
    key_init, (n_init, N_JOINTS, m, 1), l_range, x_range, a_range
  )
  if seeds is not None:
    n_seeds = seeds.a.shape[0]
    init = jax.tree.map(lambda z, sd: z.at[:n_seeds].set(sd), init, seeds)
  fills = rkhs.RBFMixture(l=1.0, x=0.0, a=0.0)
  fs = jax.tree.map(
    lambda z, fill: jnp.full((n_total, *z.shape[1:]), fill).at[:n_init].set(z),
    init,
    fills,
  )
  ys = jnp.full(n_total, jnp.nan)

  def evaluate(i: int) -> float:
    cost = objective(as_control_law(jax.tree.map(lambda z: z[i], fs)))
    if on_step is not None:
      on_step(i, cost)
    return cost

  for i in range(n_init):
    ys = ys.at[i].set(evaluate(i))

  for i in range(n_init, n_total):
    surrogate = gp.GaussianProcess.fit(fs, ys, profile=kernels.matern52)
    key, key_ei = jr.split(key)
    candidate = acquisition.optimize_expected_improvement(
      key_ei, surrogate, l_range, x_range, a_range
    )
    fs = jax.tree.map(lambda z, c, i=i: z.at[i].set(c[0]), fs, candidate)
    ys = ys.at[i].set(evaluate(i))

  best = int(jnp.nanargmin(ys))
  return FunctionalResult(
    best=jax.tree.map(lambda z: z[best], fs),
    best_cost=float(ys[best]),
    functions=fs,
    costs=np.asarray(ys),
    n_init=n_init,
  )
