import ConfigSpace
import torch.nn.functional as F
import torch.nn as nn
import os
from ._base import PytorchWorkflow
from .._util import unserialize_config_space

import numpy as np


class SimpleMLPClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, num_hidden_units: int):
        """Simple Multi-Layer Perceptron (MLP) workflow with fully connected layers and ReLU activation.

        Args:
            input_dim (int): the input is assumed to be a vector of this dimensionality.
            num_classes (int): the number of classes to predict.
        """
        super(SimpleMLPClassifier, self).__init__()
        
        self.layer1 = nn.Linear(input_dim, num_hidden_units)
        self.layer2 = nn.Linear(num_hidden_units, num_hidden_units)
        self.layer3 = nn.Linear(num_hidden_units, num_classes)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.layer3(x), dim=1)
        return x


class SimpleTorchMLPWorkflow(PytorchWorkflow):
    
    def __init__(self, X_train, y_train, hyperparams):
        super().__init__(SimpleMLPClassifier(input_dim=X_train.shape[1], num_classes=len(np.unique(y_train)), **hyperparams))

    def update_summary(self):
        pass

    @staticmethod
    def get_config_space() -> ConfigSpace.ConfigurationSpace:
        path = os.path.abspath(__file__)
        path = path[:path.rfind("/") + 1]
        return unserialize_config_space(path + "_simple_mlp_cs.json")