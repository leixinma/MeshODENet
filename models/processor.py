import torch
import torch.nn as nn
import torch_scatter
from torch_geometric.nn.conv import MessagePassing
from .mlp import MLP


class ProcessorLayer(MessagePassing):
    """
    Graph Neural Network Processor Layer using Message Passing.
    
    This layer updates both node and edge features through message passing
    mechanism, where edges are updated based on connected node features
    and nodes are updated based on aggregated edge messages.
    
    Args:
        out_channels (int): Output dimension for both node and edge features
        **kwargs: Additional arguments for MessagePassing
    """
    
    def __init__(self, out_channels, **kwargs):
        super(ProcessorLayer, self).__init__(**kwargs)
        
        # MLP for updating edge features
        # Input: concatenated features of two connected nodes + edge features
        self.edge_mlp = self._build_mlp(3 * out_channels, out_channels)
        
        # MLP for updating node features  
        # Input: current node features + aggregated edge messages
        self.node_mlp = self._build_mlp(2 * out_channels, out_channels)
    
    def _build_mlp(self, input_dim, output_dim, use_layernorm=False):
        """
        Build MLP with specified input and output dimensions.
        
        Args:
            input_dim (int): Input feature dimension
            output_dim (int): Output feature dimension
            use_layernorm (bool): Whether to use layer normalization
            
        Returns:
            MLP: Configured MLP module
        """
        return MLP(input_dim=input_dim, output_dim=output_dim, use_layernorm=use_layernorm)
    
    def forward(self, x, edge_index, edge_attr, size=None):
        """
        Forward pass through the processor layer.
        
        Args:
            x (torch.Tensor): Node features [N, F_node]
            edge_index (torch.Tensor): Edge indices [2, E]
            edge_attr (torch.Tensor): Edge features [E, F_edge]
            size (tuple, optional): Size of the bipartite graph
            
        Returns:
            tuple: Updated node features [N, F_node] and edge features [E, F_edge]
        """
        # Get source and target node features
        x_i = x[edge_index[0]]  # Source node features
        x_j = x[edge_index[1]]  # Target node features
        
        # Update edge features using MLP
        edge_input = torch.cat([x_i, x_j, edge_attr], dim=1)
        updated_edges = self.edge_mlp(edge_input) + edge_attr  # Residual connection
        
        # Perform message passing to update node features
        aggregated_messages = self.propagate(edge_index, x=x, edge_attr=updated_edges, size=size)
        
        # Update node features using MLP
        node_input = torch.cat([x, aggregated_messages], dim=1)
        updated_nodes = x + self.node_mlp(node_input)  # Residual connection
        
        return updated_nodes, updated_edges
    
    def message(self, edge_attr):
        """
        Define the message function.
        
        Args:
            edge_attr (torch.Tensor): Edge features
            
        Returns:
            torch.Tensor: Messages to be passed
        """
        return edge_attr
    
    def aggregate(self, inputs, index, dim_size=None):
        """
        Aggregate messages for each node.
        
        Args:
            inputs (torch.Tensor): Messages to aggregate
            index (torch.Tensor): Target node indices
            dim_size (int, optional): Number of nodes
            
        Returns:
            torch.Tensor: Aggregated messages
        """
        return torch_scatter.scatter(inputs, index, dim=0, reduce='sum') 