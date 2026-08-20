"""Dash humanoid robot assets and tasks for mjlab."""

from pathlib import Path

DASH_MJLAB_SRC_PATH = Path(__file__).parent
"""Root of the installed ``dash_mjlab`` package, used to locate MJCF assets."""

__all__ = ["DASH_MJLAB_SRC_PATH"]
