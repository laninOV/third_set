from __future__ import annotations

"""
Compatibility facade for CLI entrypoints.

Implementation lives in `third_set.cli_commands.main`.
This module keeps `python -m third_set.cli` and legacy imports stable.
"""

from importlib import import_module

_impl = import_module("third_set.cli_commands.main")

__all__ = [name for name in dir(_impl) if not name.startswith("__")]
globals().update({name: getattr(_impl, name) for name in __all__})


if __name__ == "__main__":
    raise SystemExit(main())
