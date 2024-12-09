import torch
import torch.nn as nn
from efficient_kan import KAN as BaseKAN

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = 1 if configs.features == 'S' else configs.enc_in
        
        # Create a KAN for each channel
        self.kans = nn.ModuleList([
            BaseKAN(
                layers_hidden=[self.seq_len, 256, 128, self.pred_len],
                grid_size=5,
                spline_order=3,
                scale_noise=0.1,
                scale_base=1.0,
                scale_spline=1.0,
                base_activation=nn.SiLU,
                grid_eps=0.02,
                grid_range=[-1, 1]
            ) for _ in range(self.channels)
        ])

    def forward(self, x):
        batch_size = x.shape[0]
        
        # Handle input shape - don't squeeze yet
        if len(x.shape) == 2:  # [Batch, Input length]
            x = x.unsqueeze(-1)  # Add channel dimension: [Batch, Input length, 1]
            
        # Initialize output with zeros keeping all dimensions
        output = torch.zeros(batch_size, self.seq_len + self.pred_len, self.channels).to(x.device)
        
        # Copy input sequence to first part of output 
        output[:, :self.seq_len] = x
        
        # Generate predictions for each channel
        for i in range(self.channels):
            channel_data = x[:, :, i]  # [Batch, Input length]
            pred = self.kans[i](channel_data)  # [Batch, pred_len]
            output[:, -self.pred_len:, i] = pred
            
        # Important: Don't squeeze the output - keep all dimensions
        return output  # Always return [batch_size, seq_len + pred_len, channels]

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute total regularization loss across all KANs
        """
        return sum(
            kan.regularization_loss(regularize_activation, regularize_entropy)
            for kan in self.kans
        )