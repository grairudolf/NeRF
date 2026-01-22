import torch
import numpy as np

def stratified_sampling(z_vals, num_samples):
    """Perform stratified sampling along the z-values."""
    z_samples = np.zeros_like(z_vals)
    for i in range(len(z_vals) - 1):
        z_min, z_max = z_vals[i], z_vals[i + 1]
        z_samples[i] = np.random.uniform(z_min, z_max, num_samples)
    return z_samples

def hierarchical_sampling(weights, z_vals, num_samples):
    """Perform hierarchical sampling based on the weights."""
    # Compute the cumulative distribution function (CDF)
    cdf = np.cumsum(weights)
    cdf = cdf / cdf[-1]  # Normalize to [0, 1]
    
    # Sample uniform random values
    u = np.random.uniform(0, 1, num_samples)
    
    # Inverse CDF sampling
    z_samples = np.interp(u, cdf, z_vals)
    return z_samples

def sample_points(ray_origins, ray_directions, near, far, num_samples, stratified=True):
    """Sample points along rays."""
    z_vals = np.linspace(near, far, num_samples)
    if stratified:
        z_samples = stratified_sampling(z_vals, num_samples)
    else:
        z_samples = np.random.uniform(near, far, num_samples)
    
    # Compute 3D points
    points = ray_origins[:, None, :] + ray_directions[:, None, :] * z_samples[..., None]
    return points, z_samples