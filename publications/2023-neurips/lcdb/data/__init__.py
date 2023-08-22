"""Sub-package for data loading utilities.
"""
from ._base import load_task
from ._sklearn import load_from_sklearn

__all__ = ["load_task", "load_from_sklearn"]
