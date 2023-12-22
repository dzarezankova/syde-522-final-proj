#Trains basic perceptron on everything and find weights
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.model_selection
import sklearn.preprocessing
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden, output_size):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden, hidden)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden, hidden)
        self.relu3 = nn.ReLU()
        self.layer4 = nn.Linear(hidden, hidden)
        self.relu4 = nn.ReLU()
        self.output_layer = nn.Linear(hidden, output_size)

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.relu3(self.layer3(x))
        x = self.relu4(self.layer4(x))
        return self.output_layer(x)

data = pd.read_csv("data_data/mlp_data.csv").dropna()

X = (data.loc[:, data.columns != 'NextLapTime']).to_numpy()
Y = (data.loc[:, data.columns == 'NextLapTime']).to_numpy()

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y.ravel(), test_size=0.2, shuffle=True,)
X_train, X_eval, Y_train, Y_eval = sklearn.model_selection.train_test_split(X_train, Y_train, test_size=0.25, shuffle=True,)

scaler = StandardScaler()

# Fit on training set only
X_train = scaler.fit_transform(X_train)

# Apply transform to both the training set and the test set
X_test = scaler.transform(X_test)

X_eval = scaler.transform(X_eval)

# Convert arrays to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(Y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(Y_test, dtype=torch.float32).view(-1, 1)
X_eval = torch.tensor(X_eval, dtype=torch.float32)
y_eval = torch.tensor(Y_eval, dtype=torch.float32).view(-1, 1)


train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=, shuffle=True)

# Model initialization
input_size = X_train.shape[1]  # Number of input features
hidden1 = 128
hidden2 = 64
output_size = 1  # For regression, the output size is typically 1

model = MLP(input_size, hidden1, hidden2, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10

train_losses = []
test_losses = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Calculate average training loss for the epoch
    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Evaluate on the testing set
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test).item()
    test_losses.append(test_loss)

    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}")

# Plotting Training and Testing Losses
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Testing Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Testing Loss Over Epochs')
plt.legend()
plt.show()
