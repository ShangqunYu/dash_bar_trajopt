"""List registered tasks, including Dash's. Wraps ``mjlab.scripts.list_envs``."""

from mjlab.scripts.list_envs import main

import dash_mjlab.tasks  # noqa: F401  Registers the Dash tasks.

if __name__ == "__main__":
  main()
