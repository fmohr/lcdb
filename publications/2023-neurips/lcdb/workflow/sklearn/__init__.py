"""Sub-package for sklearn models.
"""
from ._discriminant_analysis import LDAWorkflow, QDAWorkflow
from ._linear_model import LRWorkflow, RidgeWorkflow, PAWorkflow, PerceptronWorkflow
from ._naive_bayes import GaussianNBWorkflow
from ._majority import MajorityWorkflow
from ._majority_with_pp import MajorityWorkflowWithPreprocessing
from ._mean import MeanWorkflow
from ._median import MedianWorkflow
from ._knn import KNNWorkflow
from ._libsvm import LibSVMWorkflow
from ._liblinear import LibLinearWorkflow
from ._trees_ensemble import TreesEnsembleWorkflow
from ._random import RandomWorkflow
from ._tree import DTWorkflow

__all__ = [
    "LDAWorkflow",
    "QDAWorkflow",
    "LRWorkflow",
    "RidgeWorkflow",
    "PAWorkflow",
    "PerceptronWorkflow",
    "GaussianNBWorkflow",
    "MajorityWorkflow",
    "MajorityWorkflowWithPreprocessing",
    "MeanWorkflow",
    "MedianWorkflow",
    "KNNWorkflow",
    "LibSVMWorkflow",
    "LibLinearWorkflow",
    "DTWorkflow",
    "TreesEnsembleWorkflow",
    "RandomWorkflow"
]
