"""
Unit tests for NeRF components.

References:
    Tancik et al., "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis" (2020)
"""

import torch
import numpy as np
import pytest


def test_positional_encoding():
    """Test positional encoding dimensions and values."""
    from src.nerf.encoding import PositionalEncoding
    
    encoder = PositionalEncoding(num_freqs=10, include_input=True)
    
    # Test output dimension
    input_dim = 3
    output_dim = encoder.get_output_dim(input_dim)
    assert output_dim == input_dim + 2 * input_dim * 10  # 3 + 60 = 63
    
    # Test forward pass
    x = torch.randn(10, 3)
    encoded = encoder(x)
    assert encoded.shape == (10, 63)
    
    # Test that encoding is differentiable
    encoded.sum().backward()
    assert encoder.freq_bands.grad is None  # Buffers don't have gradients


def test_nerf_network():
    """Test NeRF network architecture."""
    from src.nerf.encoding import PositionalEncoding
    from src.nerf.networks import NeRFNetwork
    
    pos_encoder = PositionalEncoding(num_freqs=10, include_input=True)
    dir_encoder = PositionalEncoding(num_freqs=4, include_input=True)
    
    pos_dim = pos_encoder.get_output_dim(3)
    dir_dim = dir_encoder.get_output_dim(3)
    
    network = NeRFNetwork(pos_dim, dir_dim)
    
    # Test forward pass
    pos_encoded = pos_encoder(torch.randn(10, 3))
    dir_encoded = dir_encoder(torch.randn(10, 3))
    
    rgb, density = network(pos_encoded, dir_encoded)
    
    assert rgb.shape == (10, 3)
    assert density.shape == (10, 1)
    assert torch.all(rgb >= 0) and torch.all(rgb <= 1)  # RGB in [0, 1]
    assert torch.all(density >= 0)  # Density >= 0


def test_ray_generation():
    """Test ray generation."""
    from src.nerf.rays import get_rays
    
    height, width = 64, 64
    focal = 50.0
    pose = torch.eye(4)
    
    rays_o, rays_d = get_rays(height, width, focal, pose)
    
    assert rays_o.shape == (height, width, 3)
    assert rays_d.shape == (height, width, 3)
    
    # Check that directions are normalized
    norms = torch.norm(rays_d, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


def test_stratified_sampling():
    """Test stratified sampling."""
    from src.nerf.rays import stratified_sample
    
    batch_size = 10
    rays_o = torch.randn(batch_size, 3)
    rays_d = torch.randn(batch_size, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    
    sample_points, depths = stratified_sample(rays_o, rays_d, 2.0, 6.0, 32, perturb=True)
    
    assert sample_points.shape == (batch_size, 32, 3)
    assert depths.shape == (batch_size, 32)
    assert torch.all(depths >= 2.0) and torch.all(depths <= 6.0)


def test_hierarchical_sampling():
    """Test hierarchical importance sampling."""
    from src.nerf.rays import hierarchical_sample, stratified_sample
    
    batch_size = 10
    rays_o = torch.randn(batch_size, 3)
    rays_d = torch.randn(batch_size, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    
    # First get coarse samples
    _, depths_coarse = stratified_sample(rays_o, rays_d, 2.0, 6.0, 32, perturb=False)
    
    # Create weights (simulate density)
    weights = torch.ones(batch_size, 32)
    
    sample_points, depths_fine, depths_all = hierarchical_sample(
        rays_o, rays_d, depths_coarse, weights, 64
    )
    
    assert sample_points.shape == (batch_size, 64, 3)
    assert depths_fine.shape == (batch_size, 64)
    assert depths_all.shape == (batch_size, 32 + 64)
    
    # Check that depths_all is sorted
    for i in range(batch_size):
        assert torch.all(depths_all[i, :-1] <= depths_all[i, 1:])


def test_volume_rendering():
    """Test volume rendering (alpha compositing)."""
    from src.nerf.rendering import volume_rendering
    
    batch_size = 10
    num_samples = 32
    
    rgb = torch.rand(batch_size, num_samples, 3)
    density = torch.rand(batch_size, num_samples, 1)
    depths = torch.linspace(2.0, 6.0, num_samples).unsqueeze(0).expand(batch_size, -1)
    rays_d = torch.randn(batch_size, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    
    render_dict = volume_rendering(rgb, density, depths, rays_d)
    
    assert render_dict['color'].shape == (batch_size, 3)
    assert render_dict['depth'].shape == (batch_size, 1)
    assert render_dict['acc'].shape == (batch_size, 1)
    assert render_dict['weights'].shape == (batch_size, num_samples)
    
    # Check value ranges
    assert torch.all(render_dict['color'] >= 0) and torch.all(render_dict['color'] <= 1)
    assert torch.all(render_dict['acc'] >= 0) and torch.all(render_dict['acc'] <= 1)
    
    # Check that weights sum to <= 1 (accumulated alpha)
    assert torch.all(torch.sum(render_dict['weights'], dim=-1) <= 1.0 + 1e-5)


def test_differentiability():
    """Test that the entire pipeline is differentiable."""
    from src.nerf.encoding import PositionalEncoding
    from src.nerf.networks import NeRFNetwork
    from src.nerf.rays import stratified_sample
    from src.nerf.rendering import volume_rendering
    
    # Setup
    pos_encoder = PositionalEncoding(num_freqs=5)
    dir_encoder = PositionalEncoding(num_freqs=2)
    pos_dim = pos_encoder.get_output_dim(3)
    dir_dim = dir_encoder.get_output_dim(3)
    network = NeRFNetwork(pos_dim, dir_dim, hidden_dim=64, num_layers=4)
    
    # Forward pass
    batch_size = 5
    rays_o = torch.randn(batch_size, 3, requires_grad=True)
    rays_d = torch.randn(batch_size, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    
    sample_points, depths = stratified_sample(rays_o, rays_d, 2.0, 6.0, 16)
    
    pos_encoded = pos_encoder(sample_points.reshape(-1, 3))
    pos_encoded = pos_encoded.reshape(batch_size, 16, -1)
    
    dirs_broadcast = rays_d.unsqueeze(1).expand(batch_size, 16, 3).reshape(-1, 3)
    dir_encoded = dir_encoder(dirs_broadcast)
    dir_encoded = dir_encoded.reshape(batch_size, 16, -1)
    
    rgb, density = network(
        pos_encoded.reshape(-1, pos_dim),
        dir_encoded.reshape(-1, dir_dim)
    )
    
    rgb = rgb.reshape(batch_size, 16, 3)
    density = density.reshape(batch_size, 16, 1)
    
    render_dict = volume_rendering(rgb, density, depths, rays_d)
    loss = render_dict['color'].sum()
    
    # Check gradient flow
    loss.backward()
    assert rays_o.grad is not None
    assert torch.any(network.fc_pos_input.weight.grad != 0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
