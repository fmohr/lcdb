from lcdb.data import load_task, random_split_from_array
from lcdb.model.torch import SimpleMLPClassifier, PytorchModel
from sklearn.preprocessing import StandardScaler

task_name = "sklearn.iris"

# Load data from Sklearn
(X, y), metadata = load_task(task_name)

print("Task: ", metadata["name"])
print("Type: ", metadata["type"])

# Preprocessing
# Scale data to have mean 0 and variance 1
# which is importance for convergence of the neural network
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data set into training and testing
X_train, X_test, y_train, y_test = random_split_from_array(
    X_scaled, y, train_size=0.8, random_state=42
)

# Define Pytorch Module
torch_module = SimpleMLPClassifier(metadata["input_dimension"], metadata["num_classes"])

# Define Model which handles training and evaluation of Pytorch modules
model = PytorchModel(torch_module=torch_module)

# Train the model
model.fit(dataset_train=(X_train, y_train), dataset_test=(X_test, y_test))

# Collect all metadata which should be JSON Serializable
print(model.metadata)
