"""Top-level public API for the ``pras`` package."""

from importlib.metadata import PackageNotFoundError, version

from pras.api import process_structure
from pras.process import ProcessOptions

try:
    __version__ = version("pras")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__", "process_structure", "ProcessOptions"]
