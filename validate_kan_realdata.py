import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from models.KAN import Model

class ConfigClass:
    def __init__(self):
        self.seq_len = 50  # Input sequence length
        self.pred_len = 10  # Prediction length
        self.features = 'M'  # Multivariate features
        self.enc_in = 6  # Number of input channels

# Create configuration
configs = ConfigClass()

# Load real datasets
data_files = [
    'dataset/AAPLh.csv',
    'dataset/NVDAh.csv',
    'dataset/TSLAh.csv'
]

def create_sequences(data, seq_len, pred_len):
    """
    Generate sequences of length `seq_len` and targets of length `pred_len`.
    """
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

dataset = []
for file in data_files:
    df = pd.read_csv(file)
    df['date'] = pd.to_datetime(df['date'])
    
    # Drop 'date' column for features
    features = df.drop(columns=['date']).values.astype(np.float32)
    
    # Create sequences
    X, y = create_sequences(features, seq_len=configs.seq_len, pred_len=configs.pred_len)
    
    # Split into train/val/test
    train_len = int(len(X) * 0.7)
    val_len = int(len(X) * 0.2)
    test_len = len(X) - train_len - val_len
    
    X_train, y_train = X[:train_len], y[:train_len]
    X_val, y_val = X[train_len:train_len+val_len], y[train_len:train_len+val_len]
    X_test, y_test = X[train_len+val_len:], y[train_len+val_len:]
    
    # Convert to tensors
    dataset.append((
        torch.from_numpy(X_train), torch.from_numpy(y_train),
        torch.from_numpy(X_val), torch.from_numpy(y_val),
        torch.from_numpy(X_test), torch.from_numpy(y_test)
    ))


# Initialize the KAN model
model = Model(configs)

def train_model(model, X_train, y_train, X_val, y_val, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        total_loss = 0
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X_train[:, :configs.seq_len])
        
        # Compute loss (predicting the next part of the sequence)
        loss = criterion(outputs[:, -configs.pred_len:], y_train[:, -configs.pred_len:])
        
        # Add regularization loss
        reg_loss = model.regularization_loss()
        total_loss = loss + reg_loss
        
        # Backward pass and optimize
        total_loss.backward()
        optimizer.step()
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss.item():.4f}')

def demonstrate_prediction(model, X_test, y_test):
    """
    Demonstrate model's prediction capability
    """
    # Make prediction
    with torch.no_grad():
        prediction = model(X_test[:, :configs.seq_len])
    
    # Plot original and predicted
    plt.figure(figsize=(15, 5))
    for i in range(configs.enc_in):
        plt.subplot(1, configs.enc_in, i+1)
        plt.plot(np.arange(configs.seq_len), X_test[0, :, i], label='Original')
        plt.plot(np.arange(configs.seq_len, configs.seq_len + configs.pred_len), 
                 prediction[0, -configs.pred_len:, i].numpy(), label='Predicted', linestyle='--')
        plt.title(f'Channel {i} Prediction')
        plt.xlabel('Time Steps')
        plt.ylabel('Value')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

# Training loop
for X_train, y_train, X_val, y_val, X_test, y_test in dataset:
    train_model(model, X_train, y_train, X_val, y_val, epochs=10)
    
    # Evaluate on test set
    with torch.no_grad():
        test_pred = model(X_test[:, :configs.seq_len])
        test_loss = nn.MSELoss()(test_pred[:, -configs.pred_len:], y_test)
    
    print(f"Test Loss: {test_loss:.4f}")
    
    # Visualize predictions
    demonstrate_prediction(model, X_test, y_test)

# Plot KAN activations
def plot_kan_activations(model):
    """
    Plot activation functions for each KAN layer
    """
    plt.figure(figsize=(15, 5 * len(model.kans)))
    
    for i, kan in enumerate(model.kans):
        plt.subplot(len(model.kans), 1, i+1)
        
        # Get the first layer's base activation function
        for j, (name, param) in enumerate(kan.named_parameters()):
            if 'base_fn' in name:
                x = torch.linspace(-1, 1, 100)
                y = param(x)
                plt.plot(x.detach(), y.detach(), label=f'Channel {i} Base Activation')
                break
        
        plt.title(f'KAN Activation for Channel {i}')
        plt.xlabel('Input')
        plt.ylabel('Activation')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

# Run the notebook
print("Starting KAN model analysis on real datasets...")

# Train and evaluate the model
for X_train, y_train, X_val, y_val, X_test, y_test in dataset:
    train_model(model, X_train, y_train, X_val, y_val, epochs=10)
    
    # Evaluate on test set
    with torch.no_grad():
        test_pred = model(X_test[:, :configs.seq_len])
        test_loss = nn.MSELoss()(test_pred[:, -configs.pred_len:], y_test)
    
    print(f"Test Loss: {test_loss:.4f}")
    
    # Visualize predictions
    demonstrate_prediction(model, X_test, y_test)

# Visualize KAN activations
plot_kan_activations(model)

print("KAN Model Analysis Complete!")