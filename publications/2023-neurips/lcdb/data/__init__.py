"""Sub-package for data loading utilities.
"""
from ._base import load_task
from ._sklearn import load_from_sklearn
from ._split import random_split_from_array

__all__ = ["load_task", "load_from_sklearn", "random_split_from_array"]
