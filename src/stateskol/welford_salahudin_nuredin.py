"""Importable wrapper for the applicant-named Welford implementation."""

import importlib.util
from pathlib import Path

_PACKAGE_DIR = Path(
    next(iter(importlib.util.find_spec("stateskol").submodule_search_locations))
)
_IMPLEMENTATION = _PACKAGE_DIR / "welford (Salahudin Nuredin).py"

_spec = importlib.util.spec_from_file_location(
    "stateskol_welford_impl",
    _IMPLEMENTATION,
)

if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load {_IMPLEMENTATION}")

_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

welford = _module.welford
