import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) with configurable architecture.
    
    Args:
        input_dim (int): Input feature dimension
        output_dim (int): Output feature dimension (default: 128)
        hidden_dims (list): List of hidden layer dimensions (default: [128, 128])
        activation (nn.Module): Activation function (default: nn.ReLU)
        use_layernorm (bool): Whether to use layer normalization (default: False)
    """
    
    def __init__(self, input_dim, output_dim=128, hidden_dims=[128, 128], 
                 activation=nn.ReLU, use_layernorm=False):
        super(MLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation())
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through the MLP.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        return self.mlp(x) 