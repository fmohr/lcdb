# %%
# pip install torch torchvision lightning torchmetrics

from typing import Any, Optional
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import lightning.pytorch as pl
import torchmetrics as tm
import pandas as pd
from lightning.pytorch.callbacks import EarlyStopping, StochasticWeightAveraging

map_nn_activations = {"relu": "ReLU", "sigmoid": "Sigmoid", "identity": "Identity"}
map_nn_optimizer = {"adam": "Adam", "sgd": "SGD"}


class HistoryTracker(pl.Callback):
    def __init__(self):
        self.training = False

        self.train_history = []
        self.valid_history = []
        self.test_history = []

    @property
    def history(self):
        return {
            "train": self.train_history,
            "valid": self.valid_history,
            "test": self.test_history,
        }

    @property
    def history_as_dataframe(self):
        df = pd.DataFrame(self.train_history)
        return df

    def on_train_start(self, trainer, pl_module):
        self.training = True

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ) -> None:
        if self.training:
            logs = dict(
                batch_idx=batch_idx,
                epoch_idx=trainer.current_epoch,
                **{k: v.item() for k, v in outputs.items()},
            )
            self.train_history.append(logs)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ) -> None:
        if self.training:
            logs = dict(
                batch_idx=batch_idx,
                epoch_idx=trainer.current_epoch,
                **{k: v.item() for k, v in outputs.items()},
            )
            self.train_history.append(logs)


class ResNetMLPBlock(nn.Module):
    def __init__(
        self,
        num_hidden_units: int = 32,
        num_layers: int = 4,
        activation: str = "relu",
        skip_co: bool = False,
        batch_norm: bool = False,
        dropout_rate=0.0,
    ):
        """Simple Multi-Layer Perceptron (MLP) Net with fully connected layers and ReLU activation."""
        super(ResNetMLPBlock, self).__init__()

        self.input_dim = num_hidden_units
        self.output_dim = num_hidden_units

        self.skip_co = skip_co
        self.batch_norm = batch_norm

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.Linear(num_hidden_units, num_hidden_units))

            if self.batch_norm:
                self.layers.append(nn.BatchNorm1d(num_hidden_units))

            if activation in map_nn_activations:
                self.layers.append(getattr(nn, map_nn_activations[activation])())

            if dropout_rate > 0.0:
                self.layers.append(nn.Dropout(p=dropout_rate))

    def forward(self, x):
        for i, l in enumerate(self.layers):
            if self.skip_co and i >= 1:
                x = l(x) + x
            else:
                x = l(x)
        return x


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, block):
        super(MLPClassifier, self).__init__()

        self.input_dim = input_dim
        self.output_dim = num_classes

        # Network
        self.net = nn.Sequential(
            nn.Linear(input_dim, block.input_dim),
            block,
            nn.Linear(block.output_dim, num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


class LightningClassifier(pl.LightningModule):
    def __init__(
        self,
        net: nn.Module,
        optimizer: str = "adam",
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()

        # Network
        self.net = net

        # Loss configuration
        self.label_smoothing = label_smoothing

        # Optimizer
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Metrics
        self.m_accuracy = tm.Accuracy(
            task="multiclass", num_classes=self.net.output_dim, threshold=0.5
        )

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        optimizer_class = getattr(torch.optim, map_nn_optimizer[self.optimizer])
        base_optimizer = optimizer_class(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return base_optimizer

    def generic_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(
            y_hat, y, reduction="mean", label_smoothing=self.label_smoothing
        )
        acc = self.m_accuracy(y_hat, y)
        return {"loss": loss, "acc": acc}

    def training_step(self, train_batch, batch_idx):
        out = self.generic_step(train_batch, batch_idx)
        for k, v in out.items():
            self.log(f"train_{k}", v)
        return out

    def validation_step(self, val_batch, batch_idx):
        out = self.generic_step(val_batch, batch_idx)
        for k, v in out.items():
            self.log(f"valid_{k}", v)
        return out

    def test_step(self, val_batch, batch_idx):
        out = self.generic_step(val_batch, batch_idx)
        for k, v in out.items():
            self.log(f"test_{k}", v)
        return out


# Hyperparameters
# - batch_size: int
# - learning_rate: float
# - weight_decay: float
# - label_smoothing: float
# - num_hidden_units: int
# - num_layers: int
# - activation: cat
# - skip_co: cat
# - batch_norm: cat
# - dropout_rate: float
# - early_stopping_patience: int
# _ swa: cat
# - swa_lrs: float
# - swa_epoch_start: int

# Data
dataset = MNIST("", train=True, download=True, transform=transforms.ToTensor())
mnist_train, mnist_val = random_split(dataset, [55000, 5000])

train_loader = DataLoader(mnist_train, batch_size=32)
val_loader = DataLoader(mnist_val, batch_size=32)

# model
pl_classifier = LightningClassifier(
    net=MLPClassifier(
        input_dim=28 * 28,
        num_classes=10,
        block=ResNetMLPBlock(
            num_hidden_units=32,
            num_layers=4,
            activation="relu",
            skip_co=True,
            batch_norm=True,
            dropout_rate=0.0,
        ),
    ),
    learning_rate=1e-3,
    weight_decay=0.1,
    label_smoothing=0.1,
)

# training
cb = HistoryTracker()
early_stopping = EarlyStopping(monitor="valid_loss", mode="min", patience=3)
swa = StochasticWeightAveraging(swa_lrs=0.05, swa_epoch_start=1,)

trainer = pl.Trainer(
    max_epochs=10, logger=None, accelerator="cpu", callbacks=[cb, early_stopping, swa]
)
trainer.fit(pl_classifier, train_loader, val_loader)

# %%

df = cb.history_as_dataframe


import matplotlib.pyplot as plt

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(df["loss"], label="train")
plt.ylabel("Loss")
plt.subplot(2, 1, 2)
plt.plot(df["acc"], label="train")
plt.ylabel("Accuracy")
plt.show()
