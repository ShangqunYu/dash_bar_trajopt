"""Play a trained Dash policy. Wraps ``mjlab.scripts.play``."""

from mjlab.scripts.play import main

import dash_mjlab.tasks  # noqa: F401  Registers the Dash tasks.

if __name__ == "__main__":
  main()
