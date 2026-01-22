import torch
import numpy as np

def positional_encoding(positions, num_freqs=10):
    """
    Apply positional encoding to input coordinates.

    Args:
        positions: A tensor of shape (..., 3) representing the (x, y, z) coordinates.
        num_freqs: The number of frequency bands to use for encoding.

    Returns:
        A tensor of shape (..., 6 * num_freqs) representing the encoded positions.
    """
    frequencies = 2.0 ** torch.arange(0, num_freqs, dtype=positions.dtype, device=positions.device)
    encoded = []
    
    for freq in frequencies:
        for func in [torch.sin, torch.cos]:
            encoded.append(func(positions * freq))
    
    return torch.cat(encoded, dim=-1)