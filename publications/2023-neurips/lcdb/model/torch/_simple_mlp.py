import torch
import torch.nn.functional as F
import torch.nn as nn


class SimpleMLPClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        """Simple Multi-Layer Perceptron (MLP) model with fully connected layers and ReLU activation.

        Args:
            input_dim (int): the input is assumed to be a vector of this dimensionality.
            num_classes (int): the number of classes to predict.
        """
        super(SimpleMLPClassifier, self).__init__()
        
        self.layer1 = nn.Linear(input_dim, 50)
        self.layer2 = nn.Linear(50, 50)
        self.layer3 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.layer3(x), dim=1)
        return x
