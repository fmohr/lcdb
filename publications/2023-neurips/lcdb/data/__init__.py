"""Sub-package for data loading utilities.
"""
from ._sklearn import load_from_sklearn
from ._split import random_split_from_array

__all__ = ["load_from_sklearn", "random_split_from_array"]
