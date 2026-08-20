"""Exports a Dash scene to standalone MJCF. Wraps ``mjlab.scripts.export_scene``."""

from mjlab.scripts.export_scene import main

import dash_mjlab.tasks  # noqa: F401  Registers the Dash tasks.

if __name__ == "__main__":
  main()
