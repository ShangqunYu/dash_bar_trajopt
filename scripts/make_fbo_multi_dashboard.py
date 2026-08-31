"""One dashboard for the whole sweep: toggles for arms / spokes / target angle
swap in the matching run's full per-run page (rendered by make_fbo_dashboard).

Self-contained like the per-run pages: every run is embedded base64, so the
file is roughly the sum of the per-run dashboards.
"""

import argparse
import base64
import json
from pathlib import Path

from make_fbo_dashboard import build_html

SHELL = """<!doctype html>
<html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>FBO Sweep Dashboard</title>
<style>
  :root { color-scheme: light dark; }
  body { margin: 0; height: 100vh; display: flex; flex-direction: column;
    font-family: system-ui, sans-serif; background: light-dark(#f9f9f7, #0d0d0d);
    color: light-dark(#0b0b0b, #fff); }
  #bar { display: flex; flex-wrap: wrap; gap: 18px; align-items: center;
    padding: 10px 18px; border-bottom: 1px solid light-dark(#e1e0d9, #2c2c2a); }
  .group { display: flex; gap: 4px; align-items: center; }
  .group .label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em;
    color: #898781; margin-right: 6px; }
  button { font: inherit; font-size: 13px; padding: 4px 12px; border-radius: 999px;
    border: 1px solid light-dark(rgba(0,0,0,0.15), rgba(255,255,255,0.2));
    background: transparent; color: inherit; cursor: pointer; }
  button.on { background: light-dark(#0b0b0b, #fff); color: light-dark(#fff, #0b0b0b);
    border-color: transparent; }
  button.missing { opacity: 0.35; }
  #status { font-size: 12.5px; color: #898781; margin-left: auto; }
  iframe { flex: 1; border: 0; width: 100%; }
  #empty { flex: 1; display: none; place-content: center; color: #898781; }
</style>
</head>
<body>
  <div id="bar">
    <div class="group" id="g-arms"><span class="label">arms</span></div>
    <div class="group" id="g-spokes"><span class="label">spokes</span></div>
    <div class="group" id="g-target"><span class="label">target</span></div>
    <span id="status"></span>
  </div>
  <iframe id="frame"></iframe>
  <div id="empty">no run for this combination</div>
<script>
const RUNS = __RUNS__;  // key "arms|spokes|target" -> base64 page
const OPTIONS = __OPTIONS__;
const state = { arms: OPTIONS.arms[0], spokes: OPTIONS.spokes[0], target: OPTIONS.target[0] };
const key = () => `${state.arms}|${state.spokes}|${state.target}`;
const decode = b64 => new TextDecoder().decode(Uint8Array.from(atob(b64), c => c.charCodeAt(0)));

function makeGroup(id, dim, label) {
  const g = document.getElementById(id);
  OPTIONS[dim].forEach(v => {
    const b = document.createElement("button");
    b.textContent = label(v);
    b.dataset.value = v;
    b.onclick = () => { state[dim] = v; render(); };
    g.appendChild(b);
  });
}
makeGroup("g-arms", "arms", v => v === "both" ? "2 arms" : "1 arm");
makeGroup("g-spokes", "spokes", v => v);
makeGroup("g-target", "target", v => `${v} rad`);

function render() {
  for (const dim of ["arms", "spokes", "target"])
    document.querySelectorAll(`#g-${dim} button`).forEach(b => {
      b.classList.toggle("on", b.dataset.value == state[dim]);
      const probe = { ...state, [dim]: b.dataset.value };
      b.classList.toggle("missing",
        RUNS[`${probe.arms}|${probe.spokes}|${probe.target}`] === undefined);
    });
  const page = RUNS[key()];
  document.getElementById("frame").style.display = page ? "" : "none";
  document.getElementById("empty").style.display = page ? "none" : "grid";
  if (page) document.getElementById("frame").srcdoc = decode(page);
  document.getElementById("status").textContent =
    `${state.arms === "both" ? 2 : 1} arm(s) · ${state.spokes} spokes · target ${state.target} rad`;
}
render();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outputs_dir", type=Path, nargs="?", default=Path("outputs"))
    parser.add_argument("-o", "--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or args.outputs_dir / "fbo_sweep.html"

    runs = {}
    for log in sorted(args.outputs_dir.glob("*/log.jsonl")):
        header = json.loads(log.read_text().split("\n", 1)[0])
        page = build_html(log.parent)
        k = f"{header.get('arms', 'right')}|{header.get('num_spokes', 3)}|{header['target_angle']}"
        runs[k] = base64.b64encode(page.encode()).decode()
        print(f"{log.parent.name}: {k} ({len(page) / 1e6:.1f} MB)")

    arms, spokes, targets = ({k.split("|")[i] for k in runs} for i in range(3))
    options = {
        "arms": [a for a in ("right", "both") if a in arms],
        "spokes": sorted(spokes, key=float),
        "target": sorted(targets, key=float),
    }
    html = SHELL.replace("__RUNS__", json.dumps(runs)).replace(
        "__OPTIONS__", json.dumps(options)
    )
    out.write_text(html)
    print(f"{out} ({len(html) / 1e6:.1f} MB, {len(runs)} runs)")


if __name__ == "__main__":
    main()
