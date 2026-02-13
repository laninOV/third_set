from __future__ import annotations

"""
Compatibility facade for Sofascore client API.

Implementation lives in `third_set.sofascore_client.client_impl`.
This module re-exports existing symbols to keep imports stable.
"""

from importlib import import_module

_impl = import_module("third_set.sofascore_client.client_impl")

__all__ = [name for name in dir(_impl) if not name.startswith("__")]
globals().update({name: getattr(_impl, name) for name in __all__})
