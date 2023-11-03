import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

import torch
from torch.utils.data import Dataset


class ZcDataset(Dataset):

    def __init__(self, input_dict):
        self.x_train = []
        self.y_train = []
        for key, value in input_dict.items():
            self.x_train.append(value['zc_score'])
            self.y_train.append(value['gt'])
        self.x_train = torch.tensor(self.x_train).float()
        self.y_train = torch.tensor(self.y_train).float()

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]

# Define the MLP model as a PyTorch module
class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(18, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Define the input dictionary
with open('./checkpoints/mq-bench-layerwise-zc.json', 'r') as f:
    input_dict = json.load(f)

# Create an instance of the custom dataset
dataset = ZcDataset(input_dict)

# Split the data into training and testing sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, test_size])

# Create data loaders for batch processing
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Create an instance of the MLP model and define the loss function and optimizer
mlp_model = MLP()
loss_function = nn.MSELoss()
optimizer = optim.Adam(mlp_model.parameters(), lr=0.002)

# Move the model to GPU if available
if torch.cuda.is_available():
    mlp_model.cuda()

# Train the model
num_epochs = 100000
print('Start training...')
for epoch in range(num_epochs):
    mlp_model.train()
    for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
        if torch.cuda.is_available():
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
        optimizer.zero_grad()
        y_pred = mlp_model(batch_x)
        loss = loss_function(y_pred, batch_y)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print('Training completed!')

# Evaluate the model
mlp_model.eval()
y_pred = []
y_true = []
print('Start evaluation...')
for batch_x, batch_y in test_loader:
    if torch.cuda.is_available():
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
    with torch.no_grad():
        batch_pred = mlp_model(batch_x)
        y_pred.extend(batch_pred.cpu().numpy())
        y_true.extend(batch_y.cpu().numpy())

# Calculate the mean squared error (MSE) loss between the predicted and actual values
mse_loss = mean_squared_error(y_true, y_pred)

print('MSE Loss:', mse_loss)