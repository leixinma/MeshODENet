import torch
import torch.nn as nn
from torchdiffeq import odeint
from .ode_function import NeuralODEFunc, normalize, NodeType


class NeuralODEProcessor(nn.Module):
    """
    Neural ODE Processor for integrating the ODE function over time.
    
    Args:
        ode_func (NeuralODEFunc): The ODE function defining dynamics
        rtol (float): Relative tolerance for ODE solver (default: 1e-5)
        atol (float): Absolute tolerance for ODE solver (default: 1e-6)
    """
    
    def __init__(self, ode_func, rtol=1e-5, atol=1e-6):
        super(NeuralODEProcessor, self).__init__()
        self.ode_func = ode_func
        self.rtol = rtol
        self.atol = atol
    
    def forward(self, initial_node_state, initial_edge_index, initial_mesh_edge_attr,
                other_features, node_type, stats, t_span):
        """
        Integrate the ODE function over the specified time span.
        
        Args:
            initial_node_state (tuple): Initial (position, velocity)
            initial_edge_index (torch.Tensor): Edge connectivity [2, E]
            initial_mesh_edge_attr (torch.Tensor): Static mesh edge features [E, F_mesh]
            other_features (torch.Tensor): Additional node features [N, F_other]
            node_type (torch.Tensor): Node type indicators [N, 1]
            stats (dict): Normalization statistics
            t_span (torch.Tensor): Time points for integration
            
        Returns:
            tuple: Integrated positions and velocities over time
        """
        with torch.cuda.amp.autocast():
            states = odeint(
                func=lambda t, state: self.ode_func(
                    t, state, initial_edge_index, initial_mesh_edge_attr,
                    other_features, node_type, stats
                ),
                y0=initial_node_state,
                t=t_span,
                rtol=self.rtol,
                atol=self.atol,
                method='rk4',
                options={'step_size': 0.1}
            )
        
        node_pos, node_vel = states
        return node_pos, node_vel


class MeshODENet(nn.Module):
    """
    MeshODENet: Neural ODE-based Graph Neural Network for mesh simulation.
    
    This model combines graph neural networks with neural ODEs to simulate
    physics-based mesh deformation over time.
    
    Args:
        hidden_dim (int): Hidden dimension for embeddings
        output_dim (int): Output dimension (velocity dimension)
        node_size (int): Node feature dimension
        edge_size (int): Edge feature dimension
        num_layers (int): Number of processor layers (default: 1)
        rtol (float): Relative tolerance for ODE solver (default: 1e-5)
        atol (float): Absolute tolerance for ODE solver (default: 1e-6)
    """
    
    def __init__(self, hidden_dim, output_dim, node_size, edge_size, 
                 num_layers=1, rtol=1e-5, atol=1e-6):
        super(MeshODENet, self).__init__()
        
        # Create the ODE function
        ode_func = NeuralODEFunc(
            node_input_dim=output_dim,  # Velocity dimension
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            edge_size=edge_size,
            node_size=node_size,
            num_layers=num_layers
        )
        
        # Create the ODE processor
        self.neural_ode_processor = NeuralODEProcessor(ode_func, rtol, atol)
    
    def forward(self, data, stats, t_span):
        """
        Forward pass through the MeshODENet.
        
        Args:
            data (dict): Input data containing:
                - velocity: Node velocities [T, N, 3]
                - world_pos: Node positions [T, N, 3]
                - edge_index: Edge connectivity [T, 2, E]
                - youngs: Material properties [T, N, 1]
                - one_hot_node_type: Node type one-hot [T, N, num_types]
                - mesh_edge_attr: Mesh edge features [T, E, F_mesh]
                - node_type: Node type indicators [T, N, 1]
            stats (dict): Normalization statistics
            t_span (torch.Tensor): Time points for integration
            
        Returns:
            tuple: Predicted positions and velocities over time
        """
        # Extract initial conditions and features
        initial_velocity = data['velocity'][0]
        initial_pos = data['world_pos'][0]
        initial_edge_index = data['edge_index'][0]
        youngs = data['youngs'][0]
        one_hot_node_type = data['one_hot_node_type'][0]
        initial_mesh_edge_attr = data['mesh_edge_attr'][0]
        node_type = data['node_type'][0]
        
        # Prepare additional node features (normalized)
        other_features = torch.cat([
            normalize(youngs, stats['youngs_mean'], stats['youngs_std']),
            one_hot_node_type
        ], dim=-1)
        
        # Normalize mesh edge attributes
        initial_mesh_edge_attr = normalize(
            initial_mesh_edge_attr, 
            stats['mesh_edge_mean'], 
            stats['mesh_edge_std']
        )
        
        # Set up initial state
        initial_node_state = (initial_pos, initial_velocity)
        
        # Integrate over time using Neural ODE
        output_pos, output_vel = self.neural_ode_processor(
            initial_node_state=initial_node_state,
            initial_edge_index=initial_edge_index,
            initial_mesh_edge_attr=initial_mesh_edge_attr,
            other_features=other_features,
            node_type=node_type,
            stats=stats,
            t_span=t_span
        )
        
        return output_pos, output_vel
    
    def loss(self, pred_pos, data):
        """
        Compute loss between predicted and target positions.
        
        Args:
            pred_pos (torch.Tensor): Predicted positions [T, N, 3]
            data (dict): Data containing target positions
            
        Returns:
            torch.Tensor: Root mean square error loss
        """
        target_pos = data['world_pos']
        error = torch.sum((target_pos - pred_pos) ** 2, dim=1)
        loss = torch.sqrt(torch.mean(error))
        return loss 