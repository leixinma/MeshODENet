import torch
import torch.nn as nn
import torch_scatter
from torch_geometric.nn.conv import MessagePassing
from .mlp import MLP
from .processor import ProcessorLayer
import enum


class NodeType(enum.IntEnum):
    """Node type enumeration for boundary conditions."""
    NORMAL = 0
    WALL_BOUNDARY = 1
    SIZE = 2


def normalize(to_normalize, mean_vec, std_vec):
    """Normalize tensor using mean and standard deviation."""
    return (to_normalize - mean_vec) / std_vec


class NeuralODEFunc(MessagePassing):
    """
    Neural ODE Function for physics-based mesh simulation.
    
    This module defines the dynamics function for the ODE solver, combining
    graph neural networks with physics constraints to simulate mesh deformation.
    
    Args:
        node_input_dim (int): Input dimension for node features
        hidden_dim (int): Hidden dimension for embeddings
        output_dim (int): Output dimension (velocity dimension)
        edge_size (int): Edge feature dimension
        node_size (int): Node feature dimension
        num_layers (int): Number of processor layers
        **kwargs: Additional arguments for MessagePassing
    """
    
    def __init__(self, node_input_dim, hidden_dim, output_dim, edge_size, 
                 node_size, num_layers=1, **kwargs):
        super(NeuralODEFunc, self).__init__(**kwargs)
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Encoders for node velocity and edge features
        self.node_encoder = MLP(node_input_dim, hidden_dim)
        self.edge_encoder = MLP(edge_size, hidden_dim)
        
        # Graph processor layers
        self.processor = nn.ModuleList()
        for _ in range(self.num_layers):
            self.processor.append(ProcessorLayer(hidden_dim))
        
        # Decoder for final velocity prediction
        self.decoder = MLP(hidden_dim + node_size, output_dim, use_layernorm=False)
    
    def forward(self, t, state, initial_edge_index, initial_mesh_edge_attr, 
                other_features, node_type, stats):
        """
        Forward pass defining the ODE dynamics.
        
        Args:
            t (torch.Tensor): Current time (not used in autonomous system)
            state (tuple): Current state (position, velocity)
            initial_edge_index (torch.Tensor): Edge connectivity [2, E]
            initial_mesh_edge_attr (torch.Tensor): Static mesh edge features [E, F_mesh]
            other_features (torch.Tensor): Additional node features [N, F_other]
            node_type (torch.Tensor): Node type indicators [N, 1]
            stats (dict): Normalization statistics
            
        Returns:
            tuple: Time derivatives (dpos/dt, dvel/dt)
        """
        pos, vel = state
        
        # Normalize velocity for network input
        vel_norm = normalize(vel, stats["vel_mean"], stats["vel_std"])
        h_node = self.node_encoder(vel_norm)
        
        # Compute current edge features from positions
        edge_index = initial_edge_index.long()
        u_i = pos[edge_index[0]]  # Source positions
        u_j = pos[edge_index[1]]  # Target positions
        u_ij = u_i - u_j
        u_ij_norm = torch.norm(u_ij, p=2, dim=1, keepdim=True)
        
        # Create edge features and normalize
        edge_attr_unnorm = torch.cat((u_ij, u_ij_norm), dim=-1).float()
        edge_attr = normalize(edge_attr_unnorm, stats["edge_mean"], stats["edge_std"])
        
        # Encode combined edge features
        h_edge = self.edge_encoder(torch.cat([edge_attr, initial_mesh_edge_attr], dim=-1))
        
        # Process through graph neural network layers
        for processor in self.processor:
            h_node, h_edge = processor(h_node, edge_index, h_edge)
        
        # Decode to velocity changes
        decoder_input = torch.cat([h_node, other_features], dim=-1)
        velocity_change_norm = self.decoder(decoder_input)
        
        # Denormalize velocity changes
        velocity_change = velocity_change_norm * stats["vel_std"]
        
        # Apply boundary conditions
        # Boundary nodes have zero velocity and acceleration
        dvel_dt = torch.where(
            node_type == NodeType.WALL_BOUNDARY, 
            torch.zeros_like(velocity_change), 
            velocity_change
        )
        
        dpos_dt = torch.where(
            node_type == NodeType.WALL_BOUNDARY, 
            torch.zeros_like(vel), 
            vel
        )
        
        return dpos_dt, dvel_dt 