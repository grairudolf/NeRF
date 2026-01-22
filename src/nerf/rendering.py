"""
Volumetric rendering and alpha compositing for NeRF.

References:
    Tancik et al., "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis" (2020)
    Max, "Optical Models for Direct Volume Rendering" (1995)

Classical volumetric rendering via ray marching:
    C(r) = ∫[tn to tf] T(t) * σ(r(t)) * c(r(t), d) dt
    where:
        - σ(r(t)) is density (opacity) at point r(t)
        - c(r(t), d) is color at point with view direction d
        - T(t) = exp(-∫[tn to t] σ(r(s)) ds) is transmittance

In practice, we discretize along rays (Riemann sum) and use differentiable alpha compositing:
    T_i = exp(-σ_i * δ_i)  where δ_i is step size
    α_i = 1 - exp(-σ_i * δ_i)
    C = Σ T_i * α_i * c_i + T_n * c_bg
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict


def volume_rendering(
    rgb: torch.Tensor,
    density: torch.Tensor,
    depths: torch.Tensor,
    rays_d: torch.Tensor,
    bg_color: Optional[torch.Tensor] = None,
    white_background: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Differentiable volume rendering via alpha compositing.
    
    Computes the color of rays by integrating over sample points along each ray.
    
    Args:
        rgb (torch.Tensor): Colors at sample points, shape [batch, num_samples, 3], range [0, 1]
        density (torch.Tensor): Density (σ) at sample points, shape [batch, num_samples, 1], range [0, ∞)
        depths (torch.Tensor): Depth values along rays, shape [batch, num_samples]
        rays_d (torch.Tensor): Ray directions (normalized), shape [batch, 3]
        bg_color (torch.Tensor): Background color, shape [3] or [batch, 3]. If None, use black or white
        white_background (bool): Use white background if bg_color is None
    
    Returns:
        Dict[str, torch.Tensor]:
            - 'color': Rendered color per ray, shape [batch, 3]
            - 'depth': Expected depth per ray, shape [batch, 1]
            - 'acc': Accumulated transmittance (opacity), shape [batch, 1]
            - 'weights': Per-sample weights for importance sampling, shape [batch, num_samples]
            - 'transmittance': Transmittance at each sample, shape [batch, num_samples]
    """
    batch_size, num_samples = density.shape[:2]
    device = density.device
    
    # Flatten depth samples to compute step sizes
    # Step size δ_i = ||depth_i+1 - depth_i|| / ||rays_d||
    # Since rays_d is normalized: step_size = depth_i+1 - depth_i
    
    # Compute differences between consecutive depths
    depth_diff = torch.diff(depths, dim=-1, prepend=depths[..., :1] - 1e6)
    depth_diff = torch.abs(depth_diff)  # Ensure positive
    depth_diff = torch.clamp(depth_diff, min=1e-3)  # Minimum step size
    
    # Compute alpha values: α_i = 1 - exp(-σ_i * δ_i)
    # density: [batch, num_samples, 1], depth_diff: [batch, num_samples]
    alpha = 1.0 - torch.exp(-density.squeeze(-1) * depth_diff)
    alpha = torch.clamp(alpha, 0.0, 1.0)  # Ensure valid range
    
    # Compute transmittance: T_i = exp(-Σ[j=0 to i-1] σ_j * δ_j)
    # Cumulative sum of density * step_size
    dists = density.squeeze(-1) * depth_diff  # [batch, num_samples]
    transmittance = torch.exp(-torch.cumsum(dists, dim=-1) + dists)  # Shift for proper indexing
    # Prepend T_0 = 1 (no prior occlusion)
    transmittance = torch.cat([torch.ones_like(transmittance[..., :1]), transmittance[..., :-1]], dim=-1)
    
    # Composite weights: w_i = T_i * α_i
    weights = transmittance * alpha  # [batch, num_samples]
    
    # Composite color: C = Σ w_i * c_i
    color = torch.sum(weights.unsqueeze(-1) * rgb, dim=1)  # [batch, 3]
    
    # Expected depth: Ê[z] = Σ w_i * z_i
    depth_map = torch.sum(weights * depths, dim=-1, keepdim=True)  # [batch, 1]
    
    # Accumulated alpha (total opacity)
    acc = torch.sum(weights, dim=-1, keepdim=True)  # [batch, 1]
    
    # Add background color: C_final = C_ray + (1 - acc) * C_bg
    if bg_color is None:
        if white_background:
            bg_color = torch.ones(3, device=device, dtype=rgb.dtype)
        else:
            bg_color = torch.zeros(3, device=device, dtype=rgb.dtype)
    
    # Handle batch vs non-batch background color
    if bg_color.dim() == 1:
        bg_color = bg_color.unsqueeze(0).expand(batch_size, -1)
    
    color = color + (1.0 - acc) * bg_color
    
    return {
        'color': color,
        'depth': depth_map,
        'acc': acc,
        'weights': weights,
        'transmittance': transmittance,
    }


def render_rays(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    network: nn.Module,
    pos_encoder,
    dir_encoder,
    near: float,
    far: float,
    num_samples_coarse: int,
    num_samples_fine: Optional[int] = None,
    perturb: bool = True,
    white_background: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Complete rendering pipeline: sampling -> network forward -> volume rendering.
    
    Implements the hierarchical NeRF approach:
    1. Stratified sampling along rays
    2. Forward through coarse network
    3. Importance sampling based on coarse density
    4. Forward through fine network
    5. Volume rendering
    
    Args:
        rays_o (torch.Tensor): Ray origins, shape [batch, 3]
        rays_d (torch.Tensor): Ray directions, shape [batch, 3]
        network: NeRF network (coarse and/or fine)
        pos_encoder: Positional encoder for coordinates
        dir_encoder: Positional encoder for view directions
        near (float): Near plane distance
        far (float): Far plane distance
        num_samples_coarse (int): Number of stratified samples
        num_samples_fine (int): Number of importance samples (if None, hierarchical sampling is skipped)
        perturb (bool): Whether to perturb samples during training
        white_background (bool): Use white background
    
    Returns:
        Dict[str, torch.Tensor]: Rendering results including coarse and fine outputs
    """
    batch_size = rays_o.shape[0]
    device = rays_o.device
    
    # ===== Coarse rendering =====
    # Stratified sampling
    sample_points_coarse, depths_coarse = stratified_sample(
        rays_o, rays_d, near, far, num_samples_coarse, perturb=perturb
    )
    
    # Encode coordinates and directions
    # sample_points_coarse: [batch, num_samples, 3]
    pos_encoded_coarse = pos_encoder(sample_points_coarse.reshape(-1, 3))
    pos_encoded_coarse = pos_encoded_coarse.reshape(batch_size, num_samples_coarse, -1)
    
    # Broadcast ray directions to all samples
    # rays_d: [batch, 3] -> [batch, num_samples, 3]
    dirs_broadcast = rays_d.unsqueeze(1).expand(batch_size, num_samples_coarse, 3)
    dirs_broadcast = dirs_broadcast.reshape(-1, 3)
    dir_encoded_coarse = dir_encoder(dirs_broadcast)
    dir_encoded_coarse = dir_encoded_coarse.reshape(batch_size, num_samples_coarse, -1)
    
    # Flatten for network forward pass
    pos_encoded_coarse_flat = pos_encoded_coarse.reshape(-1, pos_encoded_coarse.shape[-1])
    dir_encoded_coarse_flat = dir_encoded_coarse.reshape(-1, dir_encoded_coarse.shape[-1])
    
    # Forward through coarse network
    if hasattr(network, 'forward_coarse'):
        rgb_coarse, density_coarse = network.forward_coarse(pos_encoded_coarse_flat, dir_encoded_coarse_flat)
    else:
        rgb_coarse, density_coarse = network(pos_encoded_coarse_flat, dir_encoded_coarse_flat)
    
    rgb_coarse = rgb_coarse.reshape(batch_size, num_samples_coarse, 3)
    density_coarse = density_coarse.reshape(batch_size, num_samples_coarse, 1)
    
    # Volume rendering (coarse)
    render_dict_coarse = volume_rendering(
        rgb_coarse, density_coarse, depths_coarse, rays_d, white_background=white_background
    )
    
    results = {
        'color_coarse': render_dict_coarse['color'],
        'depth_coarse': render_dict_coarse['depth'],
        'acc_coarse': render_dict_coarse['acc'],
    }
    
    # ===== Fine rendering (hierarchical importance sampling) =====
    if num_samples_fine is not None and hasattr(network, 'forward_fine'):
        weights_coarse = render_dict_coarse['weights']
        
        # Importance sampling
        sample_points_fine, depths_fine, depths_all = hierarchical_sample(
            rays_o, rays_d, depths_coarse, weights_coarse, num_samples_fine, perturb=perturb
        )
        
        # Encode fine samples
        pos_encoded_fine = pos_encoder(sample_points_fine.reshape(-1, 3))
        pos_encoded_fine = pos_encoded_fine.reshape(batch_size, num_samples_fine, -1)
        
        dirs_broadcast_fine = rays_d.unsqueeze(1).expand(batch_size, num_samples_fine, 3)
        dirs_broadcast_fine = dirs_broadcast_fine.reshape(-1, 3)
        dir_encoded_fine = dir_encoder(dirs_broadcast_fine)
        dir_encoded_fine = dir_encoded_fine.reshape(batch_size, num_samples_fine, -1)
        
        # Flatten for network forward pass
        pos_encoded_fine_flat = pos_encoded_fine.reshape(-1, pos_encoded_fine.shape[-1])
        dir_encoded_fine_flat = dir_encoded_fine.reshape(-1, dir_encoded_fine.shape[-1])
        
        # Forward through fine network
        rgb_fine, density_fine = network.forward_fine(pos_encoded_fine_flat, dir_encoded_fine_flat)
        rgb_fine = rgb_fine.reshape(batch_size, num_samples_fine, 3)
        density_fine = density_fine.reshape(batch_size, num_samples_fine, 1)
        
        # Concatenate coarse and fine samples (already sorted by depths_all)
        num_total = num_samples_coarse + num_samples_fine
        rgb_all = torch.zeros(batch_size, num_total, 3, device=device, dtype=rgb_coarse.dtype)
        density_all = torch.zeros(batch_size, num_total, 1, device=device, dtype=density_coarse.dtype)
        
        # Sort indices to combine coarse and fine samples
        sorted_indices = torch.argsort(depths_all, dim=-1)
        
        # Stack coarse and fine
        rgb_combined = torch.cat([rgb_coarse, rgb_fine], dim=1)
        density_combined = torch.cat([density_coarse, density_fine], dim=1)
        
        # Reorder using sorted indices
        for i in range(batch_size):
            rgb_all[i] = rgb_combined[i, sorted_indices[i]]
            density_all[i] = density_combined[i, sorted_indices[i]]
        
        # Volume rendering (fine)
        render_dict_fine = volume_rendering(
            rgb_all, density_all, depths_all, rays_d, white_background=white_background
        )
        
        results.update({
            'color_fine': render_dict_fine['color'],
            'depth_fine': render_dict_fine['depth'],
            'acc_fine': render_dict_fine['acc'],
        })
    
    return results
