from __future__ import annotations

"""
Compatibility facade for analyzer public API.

Implementation lives in `third_set.analyzer_core.orchestrator`.
This module re-exports analyzer symbols to preserve import paths.
"""

from importlib import import_module

_impl = import_module("third_set.analyzer_core.orchestrator")

__all__ = [name for name in dir(_impl) if not name.startswith("__")]
globals().update({name: getattr(_impl, name) for name in __all__})
