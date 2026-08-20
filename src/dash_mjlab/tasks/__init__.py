"""Dash task registrations.

Importing this package walks every subpackage, which runs each
``register_mjlab_task`` call and populates mjlab's task registry.
"""

from mjlab.utils.lab_api.tasks.importer import import_packages

_BLACKLIST_PKGS = ["utils", ".mdp"]

import_packages(__name__, _BLACKLIST_PKGS)
