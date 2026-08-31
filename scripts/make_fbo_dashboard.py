"""Build the BO-progress dashboard from a run directory produced by run_fbo.py.

Works on a partial log too (the page shows an in-progress banner). Videos are
embedded base64, so keep the running-best count modest.
"""

import argparse
import base64
import json
from pathlib import Path

TEMPLATE = Path(__file__).parent / "fbo_dashboard_template.html"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("-o", "--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or args.run_dir / "fbo_progress.html"

    lines = (args.run_dir / "log.jsonl").read_text().splitlines()
    header = json.loads(lines[0])
    evals = [json.loads(line) for line in lines[1:]]
    costs = [e["cost"] for e in evals]

    # running-best indices; snapshot at most 6 of their curves, first to final
    pareto = [i for i, c in enumerate(costs) if c == min(costs[: i + 1])]
    snapshots = sorted(set(pareto[:: max(1, len(pareto) // 5)]) | {pareto[-1]})
    # a log without curves (e.g. parsed from plain stdout) still gets a page
    snapshots = [i for i in snapshots if "curve" in evals[i]]

    data = {
        "costs": costs,
        "times": [e["t"] for e in evals],
        "nSeeds": 0,
        "nInit": header["n_init"],
        "holdCost": round(header["hold_cost"], 4),
        "demoCost": round(header["demo_cost"], 4),
        "bestCost": min(costs),
        "bestIndex": costs.index(min(costs)),
        "targetAngle": header["target_angle"],
        "maxAtoms": header["m"],
        "phase": header["phase"],
        "snapshots": snapshots,
        "snapshotCurves": [evals[i]["curve"] for i in snapshots],
        "pareto": pareto,
        "totalTime": sum(e["t"] for e in evals),
        "totalEvals": header["total_evals"],
        "partial": len(evals) < header["total_evals"],
    }
    videos = {
        str(i): base64.b64encode((args.run_dir / f"eval_{i}.mp4").read_bytes()).decode()
        for i in pareto
        if (args.run_dir / f"eval_{i}.mp4").exists()
    }

    html = TEMPLATE.read_text()
    html = html.replace("__DATA__", json.dumps(data))
    html = html.replace("__VIDEOS__", json.dumps(videos))
    out.write_text(html)
    print(
        f"{out} ({len(html) / 1e6:.1f} MB, {len(videos)} videos, partial={data['partial']})"
    )


if __name__ == "__main__":
    main()
