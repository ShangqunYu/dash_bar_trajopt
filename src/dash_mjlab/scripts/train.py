"""Train a Dash policy. Wraps ``mjlab.scripts.train``."""

from mjlab.scripts.train import main

import dash_mjlab.tasks  # noqa: F401  Registers the Dash tasks.

if __name__ == "__main__":
  main()
