"""Render the video for one logged evaluation, rebuilding its law from params."""

import argparse
import json
from pathlib import Path

import jax.numpy as jnp

from dash_mjlab.trajopt import BarAngleTrajOptEnv, search
from dash_mjlab.trajopt.bofus import rkhs


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("run_dir", type=Path)
  parser.add_argument("i", type=int)
  args = parser.parse_args()

  lines = (args.run_dir / "log.jsonl").read_text().splitlines()
  header = json.loads(lines[0])
  record = next(r for r in map(json.loads, lines[1:]) if r["i"] == args.i)

  f = rkhs.RBFMixture(**{k: jnp.asarray(v) for k, v in record["params"].items()})
  env = BarAngleTrajOptEnv(num_spokes=header.get("num_spokes", 3))
  cost = env.evaluate(
    search.as_control_law(f),
    header["target_angle"],
    video_path=str(args.run_dir / f"eval_{args.i}.mp4"),
  )
  print(f"eval {args.i}: logged cost {record['cost']:.4f}, rendered at {cost:.4f}")


if __name__ == "__main__":
  main()
