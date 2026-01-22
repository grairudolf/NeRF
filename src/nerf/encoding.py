"""
Positional encoding (Fourier feature encoding) for NeRF.

References:
    Tancik et al., "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis" (2020)
    https://www.matthewtancik.com/nerf
    
The positional encoding maps continuous coordinates (x, y, z, direction) to higher-dimensional
Fourier features, enabling the neural network to learn fine details (addressing the vanishing
gradient problem for high-frequency signals).

Equation: γ(p) = (sin(2^0 * π * p), cos(2^0 * π * p), ..., sin(2^(L-1) * π * p), cos(2^(L-1) * π * p))
"""

import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    """
    Fourier positional encoding for continuous input coordinates.
    
    Encodes scalars or vectors using sinusoidal basis functions at different frequencies,
    enabling the network to learn high-frequency details.
    
    Args:
        num_freqs (int): Number of frequency levels L. Total output dimension: 2*L*input_dim
        include_input (bool): Whether to include the raw input in the output
        log_space (bool): If True, frequencies are 2^0, 2^1, ..., 2^(L-1)
                         If False, frequencies are 0, 1, ..., L-1
    """
    
    def __init__(self, num_freqs: int, include_input: bool = True, log_space: bool = True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.log_space = log_space
        
        # Pre-compute frequency scales
        if log_space:
            freq_bands = torch.linspace(0.0, num_freqs - 1, num_freqs)
            self.register_buffer('freq_bands', 2.0 ** freq_bands * np.pi)
        else:
            self.register_buffer('freq_bands', torch.linspace(0.0, num_freqs - 1, num_freqs) * np.pi)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input tensor using Fourier features.
        
        Args:
            x (torch.Tensor): Input tensor of shape (..., D) where D is the input dimension
            
        Returns:
            torch.Tensor: Encoded tensor of shape (..., D + 2*D*L) if include_input=True,
                         else (..., 2*D*L)
        """
        # Expand input for broadcasting with frequency bands
        # x: (..., D) -> (..., D, 1)
        x_expanded = x.unsqueeze(-1)
        
        # Apply frequency scaling: multiply by freq_bands
        # x_expanded * freq_bands -> (..., D, L)
        scaled = x_expanded * self.freq_bands
        
        # Compute sin and cos
        sin_features = torch.sin(scaled)
        cos_features = torch.cos(scaled)
        
        # Interleave sin and cos: (..., D, 2*L)
        encoded = torch.cat([sin_features, cos_features], dim=-1)
        
        # Flatten last two dimensions: (..., 2*D*L)
        *batch_shape, d, freq_dim = encoded.shape
        encoded = encoded.view(*batch_shape, d * freq_dim)
        
        # Concatenate with input if requested
        if self.include_input:
            return torch.cat([x, encoded], dim=-1)
        else:
            return encoded
    
    def get_output_dim(self, input_dim: int) -> int:
        """
        Compute output dimension for a given input dimension.
        
        Args:
            input_dim (int): Dimension of input
            
        Returns:
            int: Output dimension after encoding
        """
        encoded_dim = 2 * input_dim * self.num_freqs
        if self.include_input:
            return input_dim + encoded_dim
        else:
            return encoded_dim


class IdentityEncoding(nn.Module):
    """
    Identity encoding (no-op) for testing or when positional encoding is not needed.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
    
    def get_output_dim(self, input_dim: int) -> int:
        return input_dim
