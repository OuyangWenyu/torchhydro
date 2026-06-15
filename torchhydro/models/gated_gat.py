import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from typing import Optional, Tuple, Union

class GatedGATConv(nn.Module):
    """
    Gated Graph Attention Network Layer.
    
    Wraps a standard GATConv layer and adds a gating mechanism to the output.
    This allows the model to learn how much of the aggregated neighborhood information
    should be incorporated, effectively relaxing the constraint that attention weights sum to 1
    (by scaling the whole output).
    
    Formula:
        h_agg = GATConv(x, edge_index)
        g = sigmoid(Linear(x || h_agg))  or just Linear(x)
        h_out = g * h_agg + (1 - g) * x  (Gated Residual)
        
    Alternatively, if we just want to gate the attention output:
        h_out = g * h_agg
        
    Here we implement a Gated Residual connection:
    h_out = gate * GAT(x) + (1 - gate) * x
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        **kwargs
    ):
        super().__init__()
        self.gat = GATConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            concat=concat,
            dropout=dropout,
            add_self_loops=add_self_loops,
            edge_dim=edge_dim,
            bias=bias,
            **kwargs
        )
        
        # Calculate actual output dimension of GAT
        self.gat_out_channels = out_channels * heads if concat else out_channels
        
        # Gate mechanism
        # We compute a gate based on the input node features and the aggregated features
        # Gate dim should match output dim
        self.gate_linear = nn.Linear(in_channels + self.gat_out_channels, self.gat_out_channels)
        
        # If input and output dims don't match, we need a projection for the residual x
        if in_channels != self.gat_out_channels:
            self.residual_proj = nn.Linear(in_channels, self.gat_out_channels)
        else:
            self.residual_proj = None
            
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 1. Standard GAT aggregation
        h_agg = self.gat(x, edge_index, edge_attr=edge_attr)
        
        # 2. Compute Gate
        # Concatenate input x and aggregated h_agg
        # Note: x might need projection if we want to use it in residual, but for gate calc we use raw x
        # We assume x and h_agg have same number of nodes (dim 0)
        
        # If x has different dim than expected (e.g. due to heads), we might need to be careful
        # But usually in_channels is correct.
        
        gate_input = torch.cat([x, h_agg], dim=-1)
        gate = torch.sigmoid(self.gate_linear(gate_input))
        
        # 3. Apply Gated Residual
        # h_out = gate * h_agg + (1 - gate) * x_proj
        
        if self.residual_proj is not None:
            x_res = self.residual_proj(x)
        else:
            x_res = x
            
        out = gate * h_agg + (1.0 - gate) * x_res
        
        return out
