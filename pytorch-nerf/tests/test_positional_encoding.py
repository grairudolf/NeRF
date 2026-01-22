import torch
import numpy as np
import pytest
from nerf.positional_encoding import positional_encoding

def test_positional_encoding():
    # Test for 1D input
    input_1d = torch.tensor([[0.0], [1.0], [2.0]])
    encoded_1d = positional_encoding(input_1d, num_freqs=10)
    assert encoded_1d.shape == (3, 20), "1D positional encoding shape mismatch"

    # Test for 3D input
    input_3d = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    encoded_3d = positional_encoding(input_3d, num_freqs=10)
    assert encoded_3d.shape == (3, 60), "3D positional encoding shape mismatch"

    # Test for view direction input
    view_dir = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    encoded_view_dir = positional_encoding(view_dir, num_freqs=10)
    assert encoded_view_dir.shape == (2, 60), "View direction positional encoding shape mismatch"

    # Test for numerical stability
    input_stability = torch.tensor([[1e-10], [1e10]])
    encoded_stability = positional_encoding(input_stability, num_freqs=10)
    assert not torch.isnan(encoded_stability).any(), "Positional encoding contains NaN values"

if __name__ == "__main__":
    pytest.main([__file__])