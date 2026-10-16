"""Bridge to xsy_utils/spatial-exp (experiment code kept out of the core package).

Every call site guards on ``d4rt_enabled(config)``, which is False unless the config carries
``d4rt_params: {enable: true}`` -- so without it nothing here is imported or run.
"""

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1] / "xsy_utils" / "spatial-exp"


def d4rt_enabled(config) -> bool:
    raw = getattr(config, "d4rt_params", None) or {}
    return bool(dict(raw).get("enable", False))


def d4rt():
    """-> the ``d4rt_lingbot.hooks`` module."""
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from d4rt_lingbot import hooks

    return hooks
