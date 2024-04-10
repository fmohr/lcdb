"""Sub-package for sklearn models.
"""
from ._majority import MajorityWorkflow
from ._majority_with_pp import MajorityWorkflowWithPreprocessing
from ._mean import MeanWorkflow
from ._median import MedianWorkflow
from ._knn import KNNWorkflow
from ._libsvm import LibSVMWorkflow
from ._liblinear import LibLinearWorkflow
from ._randomforest import RandomForestWorkflow
from ._random import RandomWorkflow

__all__ = [
    "MajorityWorkflow",
    "MajorityWorkflowWithPreprocessing",
    "MeanWorkflow",
    "MedianWorkflow",
    "KNNWorkflow",
    "LibSVMWorkflow",
    "LibLinearWorkflow",
    "RandomForestWorkflow",
    "RandomWorkflow"
]
