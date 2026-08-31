"""Run the functional BO on the wheel task, logging everything the dashboard needs.

Writes ``<out>/log.jsonl`` (header line, then one record per rollout — flushed,
so a dashboard can be built mid-run) and one video per running-best rollout as
``<out>/eval_<i>.mp4``, each rendered in a parallel subprocess as soon as the
best improves. Build the page with ``scripts/make_fbo_dashboard.py``.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from dash_mjlab.trajopt import BarAngleTrajOptEnv, search
from dash_mjlab.trajopt.example import two_phase_push

PHASE = np.linspace(0.0, 1.0, 101)[:, None]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-angle", type=float, default=-0.5)
    parser.add_argument("--n-init", type=int, default=8)
    parser.add_argument("--m", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-spokes", type=int, default=1)
    parser.add_argument("--arms", choices=("right", "both"), default="right")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.out is None:
        angle_deg = (args.target_angle) * 180 / np.pi + 90
        angle_str = f"{angle_deg:.5g}".replace(".", "p").replace("-", "m")
        arms_tag = "-both-arms" if args.arms == "both" else ""
        args.out = Path(
            f"outputs/turn-{args.num_spokes}-spoke-to-{angle_str}{arms_tag}"
        )
    args.out.mkdir(parents=True, exist_ok=True)

    env = BarAngleTrajOptEnv(num_spokes=args.num_spokes, arms=args.arms)
    n_joints = len(env.active_joints)

    def hold(s):
        return np.zeros((*s.shape[:-1], n_joints))

    hold_cost, hold_traj = env.evaluate(hold, args.target_angle, return_trajectory=True)
    hold_cost = float(hold_cost)
    # The demo push is a 4-column right-arm law; it has no both-arms analogue.
    demo_cost = (
        float(env.evaluate(two_phase_push, args.target_angle))
        if args.arms == "right"
        else float("nan")
    )
    n_stages = (args.m - 1).bit_length() + 1
    print(
        f"target {args.target_angle} rad, {args.num_spokes} spokes, {args.arms} arms, "
        f"hold {hold_cost:.4f}, demo {demo_cost:.4f}"
    )
    # More spokes eventually crowd the parked hands; a spawn already in contact
    # silently voids the reaching term for every candidate.
    if bool(hold_traj["touched"]):
        print("WARNING: the spawn pose already touches the wheel at this spoke count.")

    log = open(args.out / "log.jsonl", "w")
    header = {
        "target_angle": args.target_angle,
        "n_init": args.n_init,
        "m": args.m,
        "seed": args.seed,
        "num_spokes": args.num_spokes,
        "arms": args.arms,
        "hold_cost": hold_cost,
        "demo_cost": demo_cost,
        "total_evals": args.n_init * 2**n_stages,
        "phase": PHASE[:, 0].round(2).tolist(),
    }
    log.write(json.dumps(header) + "\n")
    log.flush()

    # per-rollout record: cost, wall time, and the law sampled on the phase grid
    t0 = time.time()
    t_prev = t0
    best = np.inf
    renders: list[subprocess.Popen] = []

    def on_step(i: int, cost: float, f) -> None:
        nonlocal t_prev, best
        now = time.time()
        is_best = cost < best
        best = min(best, cost)
        curve = search.as_control_law(f)(PHASE)
        record = {
            "i": i,
            "cost": cost,
            "t": now - t_prev,
            "curve": curve.round(3).tolist(),
            # exact mixture params: BO proposals diverge across CPU types, so the
            # seed alone cannot rebuild a law evaluated on another node
            "params": {
                "l": np.asarray(f.l).tolist(),
                "x": np.asarray(f.x).tolist(),
                "a": np.asarray(f.a).tolist(),
            },
        }
        log.write(json.dumps(record) + "\n")
        log.flush()
        t_prev = now
        print(
            f"[{now - t0:7.1f}s] eval {i:3d}  cost={cost:.4f}  best={best:.4f}",
            flush=True,
        )
        # render each new best in parallel with the search, so the dashboard has
        # its clickable videos while the run is still going; the LHS init is not
        # worth watching, so only BO proposals get a video
        if is_best and i >= args.n_init:
            renders.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        str(Path(__file__).parent / "render_eval.py"),
                        str(args.out),
                        str(i),
                    ]
                )
            )

    result = search.minimize_functional(
        lambda law: float(env.evaluate(law, args.target_angle)),
        n_init=args.n_init,
        m=args.m,
        seed=args.seed,
        n_joints=n_joints,
        on_step=on_step,
    )
    log.close()
    print(
        f"\nbest cost {result.best_cost:.4f} rad (eval {int(np.argmin(result.costs))})"
    )
    for p in renders:
        p.wait()


if __name__ == "__main__":
    main()
