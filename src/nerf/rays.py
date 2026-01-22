"""
Ray generation and sampling for neural volumetric rendering.

References:
    Tancik et al., "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis" (2020)

This module implements the classical ray-tracing pipeline:
1. Generate rays from camera intrinsics/extrinsics
2. Stratified sampling: Divide ray interval [near, far] into uniform bins and sample randomly
3. Hierarchical sampling: Importance sample based on coarse network density predictions
"""

import torch
import numpy as np
from typing import Tuple, Optional


def get_rays(
    height: int,
    width: int,
    focal: float,
    pose: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate rays from camera intrinsics and extrinsics.
    
    Uses a simple pinhole camera model with focal length and pose matrix.
    
    Args:
        height (int): Image height in pixels
        width (int): Image width in pixels
        focal (float): Focal length (assuming square pixels and principal point at image center)
        pose (torch.Tensor): Camera pose (4x4 extrinsic matrix), shape [4, 4]
                            Format: [R | t] where R is rotation, t is translation
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - rays_o: Ray origins (camera center), shape [height, width, 3]
            - rays_d: Ray directions (normalized), shape [height, width, 3]
    """
    # Create pixel coordinates
    # meshgrid creates: x ∈ [0, width), y ∈ [0, height)
    x = torch.arange(width, dtype=torch.float32)
    y = torch.arange(height, dtype=torch.float32)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    
    # Normalize pixel coordinates to [-1, 1] and account for focal length
    # Camera space: x_cam = (x_pixel - center_x) / focal
    #              y_cam = (y_pixel - center_y) / focal
    #              z_cam = 1 (depth direction)
    
    coords = torch.stack([
        (xx - width / 2.0) / focal,
        -(yy - height / 2.0) / focal,  # Negative because pixel y increases downward
        torch.ones_like(xx),
    ], dim=-1)  # Shape: [width, height, 3]
    
    # Transform from camera space to world space using pose
    # World coords = R^T @ camera_coords + t (inverse of extrinsic transformation)
    R = pose[:3, :3]  # Rotation matrix [3, 3]
    t = pose[:3, 3]   # Translation vector [3]
    
    # R_inv = R^T for orthogonal matrix (rotation)
    R_inv = R.T
    
    # Directions: rotate camera space directions to world space
    # Shape: [width, height, 3, 3] @ [width, height, 3, 1] -> [width, height, 3]
    dirs = torch.matmul(coords, R_inv.T)  # (N, 3) @ (3, 3) -> (N, 3)
    dirs = dirs / (torch.norm(dirs, dim=-1, keepdim=True) + 1e-8)  # Normalize
    
    # Origins: broadcast camera center in world space
    # Ray origin (camera center) in world space: -R^T @ t
    rays_o = -torch.matmul(R_inv, t.unsqueeze(-1)).squeeze(-1)  # [3]
    rays_o = rays_o.expand(*coords.shape[:-1], 3)  # [width, height, 3]
    
    return rays_o.transpose(0, 1), dirs.transpose(0, 1)  # Swap to [height, width, 3]


def stratified_sample(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near: float,
    far: float,
    num_samples: int,
    perturb: bool = True,
    lindisp: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Stratified sampling along rays.
    
    Divides the ray interval [near, far] into num_samples bins and samples uniformly
    (with optional perturbation) within each bin.
    
    Args:
        rays_o (torch.Tensor): Ray origins, shape [batch, 3]
        rays_d (torch.Tensor): Ray directions (normalized), shape [batch, 3]
        near (float): Near plane distance
        far (float): Far plane distance
        num_samples (int): Number of samples per ray
        perturb (bool): If True, sample randomly within bins; else sample at bin centers
        lindisp (bool): If True, sample linearly in disparity (1/depth); else linear in depth
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - sample_points: 3D coordinates of samples, shape [batch, num_samples, 3]
            - depths: Depth values along rays, shape [batch, num_samples]
    """
    batch_size = rays_o.shape[0]
    device = rays_o.device
    
    if lindisp:
        # Sample linearly in disparity (inverse depth)
        depths = torch.linspace(1.0 / far, 1.0 / near, num_samples, device=device)
        depths = depths.unsqueeze(0).expand(batch_size, -1)  # [batch, num_samples]
        
        if perturb:
            # Perturb by random amount within bin width
            bin_width = (1.0 / near - 1.0 / far) / num_samples
            depths = depths + (torch.rand_like(depths) - 0.5) * bin_width
        
        depths = 1.0 / depths  # Convert back to depth
    else:
        # Sample linearly in depth
        depths = torch.linspace(near, far, num_samples, device=device)
        depths = depths.unsqueeze(0).expand(batch_size, -1)  # [batch, num_samples]
        
        if perturb:
            # Perturb by random amount within bin width
            bin_width = (far - near) / num_samples
            depths = depths + (torch.rand_like(depths) - 0.5) * bin_width
        
        # Clamp to valid range
        depths = torch.clamp(depths, near, far)
    
    # Compute 3D points: P = O + d * D where d is depth
    # rays_o: [batch, 3], rays_d: [batch, 3], depths: [batch, num_samples]
    # Need to expand for broadcasting
    sample_points = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * depths.unsqueeze(-1)
    # Shape: [batch, num_samples, 3]
    
    return sample_points, depths


def hierarchical_sample(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    depths_coarse: torch.Tensor,
    weights: torch.Tensor,
    num_samples: int,
    perturb: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Importance sampling based on weights from coarse network.
    
    Uses probability distribution defined by weights to sample additional points
    where the network predicts high density.
    
    Args:
        rays_o (torch.Tensor): Ray origins, shape [batch, 3]
        rays_d (torch.Tensor): Ray directions, shape [batch, 3]
        depths_coarse (torch.Tensor): Depths from coarse sampling, shape [batch, num_coarse]
        weights (torch.Tensor): Weights for importance sampling, shape [batch, num_coarse]
        num_samples (int): Number of new samples
        perturb (bool): Whether to add random perturbation
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - sample_points: New 3D coordinates, shape [batch, num_samples, 3]
            - depths_new: New depth values, shape [batch, num_samples]
            - depths_all: Concatenated coarse + fine depths (sorted), shape [batch, num_coarse + num_samples]
    """
    batch_size = rays_o.shape[0]
    device = rays_o.device
    
    # Compute bin edges from coarse depths
    # If coarse depths are [d0, d1, d2, ...], edges are at midpoints plus near/far
    depths_sorted, _ = torch.sort(depths_coarse, dim=-1)
    
    # Compute bin edges (PDF is defined over bins between samples)
    bin_edges = torch.cat([
        depths_sorted[..., :1],  # Near edge
        (depths_sorted[..., :-1] + depths_sorted[..., 1:]) / 2.0,  # Midpoints
        depths_sorted[..., -1:],  # Far edge
    ], dim=-1)  # [batch, num_coarse + 1]
    
    # Normalize weights to form a probability distribution
    pdf = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-5)
    cdf = torch.cumsum(pdf, dim=-1)  # Cumulative density function
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)  # [batch, num_coarse + 1]
    
    # Inverse transform sampling
    uniform_samples = torch.rand(batch_size, num_samples, device=device)
    
    # Find indices where uniform samples fall in CDF
    indices = torch.searchsorted(cdf, uniform_samples, right=False) - 1
    indices = torch.clamp(indices, 0, cdf.shape[-1] - 2)
    
    # Linear interpolation within bins
    left_edges = bin_edges[torch.arange(batch_size).unsqueeze(-1), indices]
    right_edges = bin_edges[torch.arange(batch_size).unsqueeze(-1), indices + 1]
    
    left_cdf = cdf[torch.arange(batch_size).unsqueeze(-1), indices]
    right_cdf = cdf[torch.arange(batch_size).unsqueeze(-1), indices + 1]
    
    # Interpolation fraction
    cdf_width = right_cdf - left_cdf
    cdf_width = torch.where(
        cdf_width < 1e-5,
        torch.ones_like(cdf_width),
        cdf_width
    )
    t = (uniform_samples - left_cdf) / cdf_width
    t = torch.clamp(t, 0, 1)
    
    # Interpolated depths
    depths_new = left_edges + t * (right_edges - left_edges)
    
    if perturb:
        # Add small random perturbation
        depths_new = depths_new + torch.randn_like(depths_new) * 1e-3
    
    # Compute 3D points
    sample_points = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * depths_new.unsqueeze(-1)
    
    # Concatenate and sort all depths
    depths_all = torch.cat([depths_coarse, depths_new], dim=-1)
    depths_all, _ = torch.sort(depths_all, dim=-1)
    
    return sample_points, depths_new, depths_all


def get_directions_from_rays(rays_d: torch.Tensor) -> torch.Tensor:
    """
    Normalize ray directions (should already be normalized, but ensure it).
    
    Args:
        rays_d (torch.Tensor): Ray directions, shape [..., 3]
    
    Returns:
        torch.Tensor: Normalized directions, shape [..., 3]
    """
    return rays_d / (torch.norm(rays_d, dim=-1, keepdim=True) + 1e-8)
