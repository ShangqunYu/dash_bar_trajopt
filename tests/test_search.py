import jax.numpy as jnp
import numpy as np

from dash_mjlab.trajopt import search
from dash_mjlab.trajopt.bofus import rkhs


def test_control_law_shape_and_clipping():
  f = rkhs.RBFMixture(
    l=jnp.full((4, 2, 1), 0.1),
    x=jnp.zeros((4, 2, 1)),
    a=jnp.full((4, 2), 5.0),
  )
  law = search.as_control_law(f)
  out = law(np.linspace(0, 1, 7)[:, None])
  assert out.shape == (7, 4)
  assert np.all(out <= 1.0) and np.all(out >= -1.0)


def test_minimize_functional_improves_on_synthetic_objective():
  """Cost is distance of the law from a fixed pose profile; BO must beat the
  worst initial sample and return the argmin of everything it evaluated."""
  s = np.linspace(0, 1, 32)[:, None]
  target = np.array([0.5, -0.5, 0.25, 0.0]) * np.sin(np.pi * s)

  def objective(law):
    return float(np.mean((law(s) - target) ** 2))

  # schedule is [1, 2]: 4 init evals, then two doubling stages -> 16 total
  result = search.minimize_functional(objective, n_init=4, m=2, seed=0)
  assert result.costs.shape == (16,)
  assert np.isfinite(result.costs).all()
  assert result.best_cost == result.costs.min()
  # costs round-trip through a float32 buffer
  replayed = objective(search.as_control_law(result.best))
  assert abs(replayed - result.best_cost) < 1e-6
