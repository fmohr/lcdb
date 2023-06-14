import numpy as np
import torch
import torch.nn as nn
import tqdm
from lcdb.data import load_from_sklearn, random_split_from_array
from lcdb.model.torch import SimpleMLPClassifier
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

# Load data from Sklearn
X, y, metadata = load_from_sklearn("iris")

input_dim = X.shape[1]
num_classes = len(np.unique(y))

# Preprocessing
# Scale data to have mean 0 and variance 1
# which is importance for convergence of the neural network
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data set into training and testing
X_train, X_test, y_train, y_test = random_split_from_array(
    X_scaled, y, train_size=0.8, random_state=42
)

# Convert to PyTorch tensors
X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()
y_train = torch.from_numpy(y_train).long()
y_test = torch.from_numpy(y_test).long()

data_train = TensorDataset(X_train, y_train)
data_test = TensorDataset(X_test, y_test)

# Build the model
model = SimpleMLPClassifier(input_dim, num_classes)

# Setup optimization strategy
batch_size = 8
learning_rate = 0.001
num_epochs = 100

data_loader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True)
data_loader_test = DataLoader(data_test, batch_size=batch_size, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss(reduction="sum")


# Training loop

loss_list = np.zeros((num_epochs,))
accuracy_list = np.zeros((num_epochs,))

for epoch in tqdm.trange(num_epochs):

    # Training set loop
    for batch_idx, (batch_input, batch_target) in enumerate(data_loader_train):

        # Forward pass
        y_pred = model(batch_input)
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
        y_pred = model(batch_input)
        loss = loss_fn(y_pred, batch_target)

        loss_list[epoch] += loss.item()

        correct = (torch.argmax(y_pred, dim=1) == batch_target).type(torch.FloatTensor).sum().item()
        accuracy_list[epoch] += correct

    accuracy_list[epoch] /= n
    loss_list[epoch] /= n


# Print metrics on test set
print(accuracy_list)
print(loss_list)

# plt.figure(figsize=(10, 10))
# plt.plot([0, 1], [0, 1], "k--")

# # One hot encoding
# enc = OneHotEncoder()
# Y_onehot = enc.fit_transform(y_test[:, np.newaxis]).toarray()

# with torch.no_grad():
#     y_pred = model(X_test).numpy()
#     fpr, tpr, threshold = roc_curve(Y_onehot.ravel(), y_pred.ravel())

# plt.plot(fpr, tpr, label="AUC = {:.3f}".format(auc(fpr, tpr)))
# plt.xlabel("False positive rate")
# plt.ylabel("True positive rate")
# plt.title("ROC curve")
# plt.legend()
# plt.show()
