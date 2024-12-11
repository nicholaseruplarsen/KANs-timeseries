import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# First, import your custom KAN model
from KAN import Model

class ConfigClass:
    def __init__(self):
        self.seq_len = 50  # Input sequence length
        self.pred_len = 10  # Prediction length
        self.features = 'M'  # Multivariate features
        self.enc_in = 3  # Number of input channels

# Create configuration
configs = ConfigClass()

# Generate synthetic financial data for testing
def generate_synthetic_financial_data(n_samples, seq_len, n_channels):
    """
    Generate synthetic financial time series data
    
    Args:
    - n_samples: Number of sample sequences
    - seq_len: Length of each sequence
    - n_channels: Number of financial features
    
    Returns:
    - Tensor of shape [n_samples, seq_len, n_channels]
    """
    # Simulate different financial indicators
    data = np.zeros((n_samples, seq_len, n_channels))
    
    for i in range(n_channels):
        # Different patterns for each channel
        trend = np.linspace(0, 10, seq_len)
        seasonality = 5 * np.sin(np.linspace(0, 4*np.pi, seq_len))
        noise = np.random.normal(0, 1, seq_len)
        
        for j in range(n_samples):
            # Create slightly different patterns for each sample
            sample_trend = trend * (1 + 0.1 * np.random.rand())
            sample_seasonality = seasonality * (1 + 0.2 * np.random.rand() - 0.1)
            sample_noise = noise * np.random.rand()
            
            data[j, :, i] = sample_trend + sample_seasonality + sample_noise
    
    return torch.tensor(data, dtype=torch.float32)

# Generate test data
n_samples = 100
test_data = generate_synthetic_financial_data(n_samples, configs.seq_len + configs.pred_len, configs.enc_in)

# Create Dataset and DataLoader
dataset = TensorDataset(test_data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize the KAN model
model = Model(configs)

# Optional: Custom plotting function for KAN layers
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

# Training loop (minimal example)
def train_model(model, dataloader, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x = batch[0]
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(x[:, :configs.seq_len])
            
            # Compute loss (predicting the next part of the sequence)
            loss = criterion(outputs[:, -configs.pred_len:], x[:, -configs.pred_len:])
            
            # Add regularization loss
            reg_loss = model.regularization_loss()
            total_loss = loss + reg_loss
            
            # Backward pass and optimize
            total_loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss.item():.4f}')

# Run training
print("Starting model training...")
train_model(model, dataloader)

# Visualize KAN activations
plot_kan_activations(model)

# Prediction demonstration
def demonstrate_prediction(model, data):
    """
    Demonstrate model's prediction capability
    """
    # Select a sample sequence
    sample_input = data[0:1, :configs.seq_len]
    
    # Make prediction
    with torch.no_grad():
        prediction = model(sample_input)
    
    # Plot original and predicted
    plt.figure(figsize=(15, 5))
    for i in range(configs.enc_in):
        plt.subplot(1, configs.enc_in, i+1)
        plt.plot(np.arange(configs.seq_len), sample_input[0, :, i].numpy(), label='Original')
        plt.plot(np.arange(configs.seq_len, configs.seq_len + configs.pred_len), 
                 prediction[0, -configs.pred_len:, i].numpy(), label='Predicted', linestyle='--')
        plt.title(f'Channel {i} Prediction')
        plt.xlabel('Time Steps')
        plt.ylabel('Value')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

# Demonstrate prediction
demonstrate_prediction(model, test_data)

print("KAN Model Analysis Complete!")