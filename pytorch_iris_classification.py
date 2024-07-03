import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Preprocessing
data = pd.read_csv('iris-flower-pytorch/iris.csv')
print(data.describe())

# Check if CUDA is available
print(torch.cuda.is_available())

X = data.drop(['species'], axis=1)
y = data['species']

# Transforms species data from strings to vectors
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
onehot_encoder = OneHotEncoder(sparse_output=False)
y_oneHot = onehot_encoder.fit_transform(y_encoded.reshape(-1, 1))

X_train, X_test, Y_train, Y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Converting to torch tensor
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
Y_train = torch.LongTensor(Y_train)
Y_test = torch.LongTensor(Y_test)

class NeuralNetwork(nn.Module):
    def __init__(self, n_input, n_output):
        np.random.seed(2)
        super(NeuralNetwork, self).__init__()
        self.input = nn.Linear(n_input, 128)
        self.hidden = nn.Linear(128, 64)
        self.output = nn.Linear(64, n_output)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.input(x))
        out = self.relu(self.hidden(out))
        out = self.output(out)
        return out

input_dim = 4
output_dim = 3
neural_network = NeuralNetwork(input_dim, output_dim)
learning_rate = 0.01

# Optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(neural_network.parameters(), learning_rate)

def train(neural_network, optimizer, criterion, X_train, X_test, Y_train, Y_test, num_epochs, train_losses, test_losses):
    for epoch in range(num_epochs):
        neural_network.train()  # Set the network to training mode
        optimizer.zero_grad()  # Clear out gradients from last backpropagation

        # Forward pass
        output_train = neural_network(X_train)

        # Calculate the loss
        loss_train = criterion(output_train, Y_train)

        # Backward propagation
        loss_train.backward()

        # Update the weights
        optimizer.step()
        
        neural_network.eval()  # Set the network to evaluation mode
        with torch.no_grad():  # Do not track gradients during evaluation
            output_test = neural_network(X_test)
            loss_test = criterion(output_test, Y_test)

        train_losses[epoch] = loss_train.item()
        test_losses[epoch] = loss_test.item()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {loss_train.item():.4f}, Test Loss: {loss_test.item():.4f}")

num_epochs = 1000
train_losses = np.zeros(num_epochs)
test_losses = np.zeros(num_epochs)

train(neural_network, optimizer, criterion, X_train, X_test, Y_train, Y_test, num_epochs, train_losses, test_losses)

# Accuracy calculation
with torch.no_grad():
    predictions_train = neural_network(X_train)
    predictions_test = neural_network(X_test)

def getAccuracy(pred_arr, original_arr):
    final_pred = np.argmax(pred_arr.numpy(), axis=1)
    original_arr = original_arr.numpy()
    count = (final_pred == original_arr).sum()
    return count / len(final_pred)

training_accuracy = getAccuracy(predictions_train, Y_train)
testing_accuracy = getAccuracy(predictions_test, Y_test)

print("Training Accuracy",training_accuracy*100,"%")
print("Testing Accuracy",testing_accuracy*100,"%")

plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.legend()
plt.show()
