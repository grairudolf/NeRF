"""
Neural Radiance Field (NeRF) network architecture.

References:
    Tancik et al., "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis" (2020)
    
The radiance field is an implicit neural representation that maps 3D coordinates (x, y, z)
and viewing direction (θ, φ) to color (RGB) and volume density (σ).

Architecture:
    - Input: Encoded position (positional encoding of x, y, z) + encoded view direction
    - Hidden layers: 8 layers of 256 units with ReLU activations
    - Skip connections: Concatenate positional encoding at layer 4
    - Outputs: Density σ at layer 8, then RGB from final layer with different input
    
This two-stream approach (one for density/geometry, another for appearance) allows the
network to learn that density is view-independent while RGB depends on viewing angle.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from .encoding import PositionalEncoding


class NeRFNetwork(nn.Module):
    """
    Multi-layer perceptron implementing the neural radiance field.
    
    Maps encoded position and view direction to (RGB, density).
    
    Args:
        input_dim_pos (int): Dimension of encoded position features
        input_dim_dir (int): Dimension of encoded view direction features
        hidden_dim (int): Number of hidden units per layer
        num_layers (int): Number of hidden layers
        skip_layers (list): Indices where positional encoding is concatenated
        use_view_direction (bool): Whether to condition on viewing direction
    """
    
    def __init__(
        self,
        input_dim_pos: int,
        input_dim_dir: int = 24,  # Encoded view direction
        hidden_dim: int = 256,
        num_layers: int = 8,
        skip_layers: Optional[list] = None,
        use_view_direction: bool = True,
    ):
        super().__init__()
        
        self.input_dim_pos = input_dim_pos
        self.input_dim_dir = input_dim_dir
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.skip_layers = skip_layers or [4]
        self.use_view_direction = use_view_direction
        
        # First layer: processes position
        self.fc_pos_input = nn.Linear(input_dim_pos, hidden_dim)
        
        # Position processing layers (with skip connections)
        self.pos_layers = nn.ModuleList()
        for i in range(1, num_layers):
            layer_input_dim = hidden_dim
            if i in self.skip_layers:
                # Skip connection: concatenate original positional encoding
                layer_input_dim += input_dim_pos
            
            self.pos_layers.append(nn.Linear(layer_input_dim, hidden_dim))
        
        # Density output (view-independent)
        self.fc_density = nn.Linear(hidden_dim, 1)
        
        # View-dependent RGB branch
        if use_view_direction:
            # Input: position features + encoded view direction
            self.fc_rgb_input = nn.Linear(hidden_dim + input_dim_dir, hidden_dim)
            self.fc_rgb_output = nn.Linear(hidden_dim, 3)
        else:
            # Input: position features only
            self.fc_rgb_input = nn.Linear(hidden_dim, hidden_dim)
            self.fc_rgb_output = nn.Linear(hidden_dim, 3)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
    
    def forward(
        self,
        pos_encoded: torch.Tensor,
        dir_encoded: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the radiance field network.
        
        Args:
            pos_encoded (torch.Tensor): Encoded position features, shape (..., input_dim_pos)
            dir_encoded (torch.Tensor): Encoded view direction, shape (..., input_dim_dir)
                                       Only required if use_view_direction=True
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - rgb: RGB color prediction, shape (..., 3), range [0, 1]
                - density: Volume density (σ), shape (..., 1), range [0, ∞)
        """
        # Store input for skip connections
        x_pos = pos_encoded
        
        # Process through position stream
        x = self.relu(self.fc_pos_input(x_pos))
        
        for i, layer in enumerate(self.pos_layers):
            if i + 1 in self.skip_layers:
                # Skip connection: concatenate original positional encoding
                x = torch.cat([x, x_pos], dim=-1)
            
            x = self.relu(layer(x))
        
        # Compute density (view-independent)
        density = self.softplus(self.fc_density(x))
        
        # View-dependent RGB branch
        if self.use_view_direction:
            assert dir_encoded is not None, "dir_encoded required when use_view_direction=True"
            x_rgb = torch.cat([x, dir_encoded], dim=-1)
        else:
            x_rgb = x
        
        x_rgb = self.relu(self.fc_rgb_input(x_rgb))
        rgb = self.sigmoid(self.fc_rgb_output(x_rgb))
        
        return rgb, density


class HierarchicalNeRF(nn.Module):
    """
    Hierarchical NeRF with coarse and fine networks.
    
    As per the original paper, we use a two-stage approach:
    1. Coarse network: Sampled along rays with stratified sampling
    2. Fine network: Sampled with importance weighting based on coarse density
    
    Args:
        coarse_network (NeRFNetwork): Coarse network for initial sampling
        fine_network (NeRFNetwork): Fine network for refinement
    """
    
    def __init__(self, coarse_network: NeRFNetwork, fine_network: NeRFNetwork):
        super().__init__()
        self.coarse = coarse_network
        self.fine = fine_network
    
    def forward_coarse(
        self,
        pos_encoded: torch.Tensor,
        dir_encoded: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward through coarse network."""
        return self.coarse(pos_encoded, dir_encoded)
    
    def forward_fine(
        self,
        pos_encoded: torch.Tensor,
        dir_encoded: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward through fine network."""
        return self.fine(pos_encoded, dir_encoded)
