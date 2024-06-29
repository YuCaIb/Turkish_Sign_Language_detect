import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam, NAdam
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from sklearn.model_selection import train_test_split

folder_names = np.load('model/folder_names.npy')
X = np.load('model/X.npy')
y = np.load('model/y.npy')
X.shape
y.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
writer = SummaryWriter('logs_torch')  # Specify a log directory (replace with your desired path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load your data (assuming X_train, X_test, y_train, y_test are NumPy arrays)
X_train = torch.from_numpy(X_train).float().to(device)
y_train = torch.from_numpy(y_train).long().to(device)  # Assuming categorical labels (long for integer labels)
X_test = torch.from_numpy(X_test).float().to(device)
y_test = torch.from_numpy(y_test).long().to(device)

# Define the dataset and dataloader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # Match training behavior
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# Define the LSTM model (no changes to architecture)
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(1662, 128, batch_first=True)
        self.lstm2 = nn.LSTM(128, 256, batch_first=True)
        self.lstm3 = nn.LSTM(256, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, folder_names.shape[0])

    def forward(self, x):
        x = F.relu(self.lstm1(x)[0])
        x = F.relu(self.lstm2(x)[0])
        x = F.relu(self.lstm3(x)[0])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)  # Softmax for categorical labels
        return x


# Define optimizer and loss function (no changes)
model = LSTMModel().to(device)
optimizer = NAdam(model.parameters(), lr= 0.001)
criterion = nn.CrossEntropyLoss()  # Categorical cross entropy loss

# Define training loop (similar structure to Keras fit)
epochs = 500
for epoch in range(epochs):
    for i, (data, target) in enumerate(train_loader):
        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log loss and other metrics
        writer.add_scalar('Loss/train', loss.item(), epoch)  # Log loss with tag 'Loss/train'

        # Calculate accuracy (optional)
        predicted = torch.argmax(output, dim=1)
        accuracy = (predicted == target).sum().item() / target.size(0)
        writer.add_scalar('Accuracy/train', accuracy, epoch)  # Log accuracy with tag 'Accuracy/train'
        # Print training progress (optional)
        if i % 100 == 0:
            print(f'Epoch: {epoch + 1}/{epochs}, Step: {i + 1}/{len(train_loader)}, Loss: {loss.item():.4f}')

# Evaluate on test set (optional)
with torch.no_grad():
    total, correct = 0, 0
    for data, target in test_loader:
        output = model(data)
        predicted = torch.argmax(output, dim=1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')