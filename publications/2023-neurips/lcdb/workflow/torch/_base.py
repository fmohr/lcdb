
import numpy as np
import torch
import torch.nn
from torch.utils.data import DataLoader, TensorDataset

from .._base_workflow import BaseWorkflow


class PytorchWorkflow(BaseWorkflow):
    def __init__(self, torch_module: torch.nn.Module) -> None:
        super().__init__()
        self.torch_module = torch_module
        
        # Collect number of parameters in the workflow
        self.summary["num_parameters"] = sum(
            p.numel() for p in torch_module.parameters()
        )
        self.summary["num_parameters_train"] = sum(
            p.numel() for p in torch_module.parameters() if p.requires_grad
        )

    def fit(self, dataset_train, dataset_test) -> "PytorchWorkflow":

        # TODO: should collect the following metadata
        # - total training time
        # - training time per epoch
        # - inference time per epoch
        # - loss and scores per epoch

        X_train, y_train = dataset_train
        X_test, y_test = dataset_test

        # TODO: the ordering of these labels does currently not (necessarily) conincide with the Bernoulli encoding in TF
        self.labels = sorted(np.unique(y_train))
        self.itos = {i: l for i, l in enumerate(self.labels)}

        # Convert to PyTorch tensors
        X_train = torch.from_numpy(X_train).float()
        X_test = torch.from_numpy(X_test).float()
        y_train = torch.from_numpy(y_train).long()
        y_test = torch.from_numpy(y_test).long()

        dataset_train = TensorDataset(X_train, y_train)
        dataset_test = TensorDataset(X_test, y_test)

        # Setup optimization strategy
        batch_size = 8
        learning_rate = 0.001
        num_epochs = 100

        data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        data_loader_test = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.torch_module.parameters(), lr=learning_rate)
        loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

        loss_list = np.zeros((num_epochs,))
        accuracy_list = np.zeros((num_epochs,))
        epoch_list = []

        for epoch in range(num_epochs):
            
            epoch_list.append(epoch)

            # Training set loop
            for batch_idx, (batch_input, batch_target) in enumerate(data_loader_train):

                # Forward pass
                y_pred = self.torch_module(batch_input)
                loss = loss_fn(y_pred, batch_target) / len(batch_input)

                # Zero gradients
                optimizer.zero_grad()

                # Backward pass
                loss.backward()

                # Update weights
                optimizer.step()
            
            # Test set loop
            n = 0
            for batch_idx, (batch_input, batch_target) in enumerate(data_loader_test):
                
                n += len(batch_input)
                # Forward pass
                y_pred = self.torch_module(batch_input)
                loss = loss_fn(y_pred, batch_target)

                loss_list[epoch] += loss.item()

                correct = (torch.argmax(y_pred, dim=1) == batch_target).type(torch.FloatTensor).sum().item()
                accuracy_list[epoch] += correct

            accuracy_list[epoch] /= n
            loss_list[epoch] /= n

        self.summary["iter_curve"] = {
            "loss": loss_list.round(4).tolist(),
            "accuracy": accuracy_list.round(4).tolist(),
            "epoch": epoch_list
        }

        return self

    def predict(self, X):
        preds = self.predict_proba(X).argmax(dim=1).int().numpy()
        return [self.itos[i] for i in preds]

    def predict_proba(self, X):
        with torch.no_grad():
            return self.torch_module.forward(torch.from_numpy(X).float())