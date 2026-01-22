import torch
import torch.nn.functional as F
import numpy as np
import pytest
from nerf.renderer import render_rays
from nerf.models import NeRFModel

def test_render_rays():
    # Setup a simple NeRF model
    model = NeRFModel()
    model.eval()  # Set the model to evaluation mode

    # Define test inputs
    rays_o = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)  # Origin of the ray
    rays_d = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)  # Direction of the ray
    view_dirs = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32)  # View direction
    num_samples = 64  # Number of samples along the ray

    # Render the rays
    rgb, depth = render_rays(model, rays_o, rays_d, view_dirs, num_samples)

    # Check the output shapes
    assert rgb.shape == (1, 3), "RGB output shape is incorrect"
    assert depth.shape == (1,), "Depth output shape is incorrect"

    # Check that RGB values are in the expected range
    assert torch.all(rgb >= 0) and torch.all(rgb <= 1), "RGB values are out of bounds"

def test_render_rays_with_invalid_input():
    model = NeRFModel()
    model.eval()

    rays_o = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    rays_d = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)  # Invalid direction
    view_dirs = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32)
    num_samples = 64

    with pytest.raises(ValueError):
        render_rays(model, rays_o, rays_d, view_dirs, num_samples)

def test_render_rays_with_different_samples():
    model = NeRFModel()
    model.eval()

    rays_o = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    rays_d = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    view_dirs = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32)

    for num_samples in [32, 64, 128]:
        rgb, depth = render_rays(model, rays_o, rays_d, view_dirs, num_samples)
        assert rgb.shape == (1, 3), "RGB output shape is incorrect for num_samples={}".format(num_samples)
        assert depth.shape == (1,), "Depth output shape is incorrect for num_samples={}".format(num_samples)

# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])