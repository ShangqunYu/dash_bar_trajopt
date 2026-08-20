"""Decimate Dash visual meshes.

The STLs exported from CAD are ~80k triangles each (~48 MB total), which bloats
the repo and slows MuJoCo model compilation. They are visual-only geometry, so
a heavy reduction is invisible in the viz.

Usage:
  uv run --with fast-simplification python scripts/decimate_meshes.py \
      --src /path/to/DASH_URDF/assets
"""

import argparse
from pathlib import Path

import trimesh

DEFAULT_DST = Path(__file__).parent.parent / "src/dash_mjlab/robots/xmls/assets"


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--src", type=Path, required=True, help="Directory of raw STLs.")
  parser.add_argument("--dst", type=Path, default=DEFAULT_DST)
  parser.add_argument(
    "--target-faces",
    type=int,
    default=8000,
    help="Face budget per mesh. Actual counts land higher when the simplifier "
    "cannot collapse further without degenerate triangles.",
  )
  args = parser.parse_args()

  args.dst.mkdir(parents=True, exist_ok=True)
  total_before = total_after = 0

  for src in sorted(args.src.glob("*.stl")):
    mesh = trimesh.load(src)
    assert isinstance(mesh, trimesh.Trimesh), f"{src.name} is not a single mesh"
    before = len(mesh.faces)
    if before > args.target_faces:
      mesh = mesh.simplify_quadric_decimation(face_count=args.target_faces)
    after = len(mesh.faces)

    dst = args.dst / src.name
    mesh.export(dst)
    total_before += src.stat().st_size
    total_after += dst.stat().st_size
    print(f"{src.name:24s} {before:7d} -> {after:6d} faces")

  print(f"\n{total_before / 1e6:.1f} MB -> {total_after / 1e6:.1f} MB")


if __name__ == "__main__":
  main()
