"""CLI entry points.

These are thin wrappers around mjlab's scripts. mjlab's versions only import
``mjlab.tasks``, so Dash tasks would not be in the registry; importing
``dash_mjlab.tasks`` here registers them before the CLI parses a task name.
"""
