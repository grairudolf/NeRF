#!/usr/bin/env python3
"""
Quick-start example: Train a minimal NeRF and render novel views.

This script demonstrates the complete NeRF pipeline in ~100 lines of code.
"""

import torch
import numpy as np
from pathlib import Path

# ============================================================================
# 1. SETUP
# ============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# 2. CREATE MODEL
# ============================================================================

from src.nerf.encoding import PositionalEncoding
from src.nerf.networks import NeRFNetwork, HierarchicalNeRF
from src.nerf.trainer import NeRFTrainer

# Positional encoders
pos_encoder = PositionalEncoding(num_freqs=10, include_input=True)
dir_encoder = PositionalEncoding(num_freqs=4, include_input=True)

# Network dimensions
pos_dim = pos_encoder.get_output_dim(3)
dir_dim = dir_encoder.get_output_dim(3)

# Create coarse and fine networks
coarse_net = NeRFNetwork(
    input_dim_pos=pos_dim,
    input_dim_dir=dir_dim,
    hidden_dim=256,
    num_layers=8,
    skip_layers=[4],
)

fine_net = NeRFNetwork(
    input_dim_pos=pos_dim,
    input_dim_dir=dir_dim,
    hidden_dim=256,
    num_layers=8,
    skip_layers=[4],
)

# Combine into hierarchical model
model = HierarchicalNeRF(coarse_net, fine_net).to(device)

print(f"Model created:")
print(f"  Coarse network: {sum(p.numel() for p in coarse_net.parameters()):,} parameters")
print(f"  Fine network: {sum(p.numel() for p in fine_net.parameters()):,} parameters")

# Create trainer
trainer = NeRFTrainer(
    model=model,
    pos_encoder=pos_encoder,
    dir_encoder=dir_encoder,
    learning_rate=5e-4,
    device=device,
)

# ============================================================================
# 3. SYNTHETIC DATA
# ============================================================================

from src.nerf.rays import get_rays

# Simple synthetic scene parameters
height, width = 64, 64
focal = 50.0
near, far = 2.0, 6.0

# Generate random camera poses
num_train_views = 10
poses_train = []
for i in range(num_train_views):
    theta = 2 * np.pi * i / num_train_views
    radius = 4.0
    
    # Camera position on circle
    cam_pos = torch.tensor([
        radius * np.cos(theta),
        0.0,
        radius * np.sin(theta),
    ], dtype=torch.float32)
    
    # Look at origin
    forward = -cam_pos
    forward = forward / torch.norm(forward)
    
    up = torch.tensor([0.0, 1.0, 0.0])
    right = torch.cross(forward, up)
    right = right / torch.norm(right)
    up = torch.cross(right, forward)
    up = up / torch.norm(up)
    
    # Pose matrix
    pose = torch.eye(4)
    pose[:3, :3] = torch.stack([right, up, -forward])
    pose[:3, 3] = cam_pos
    poses_train.append(pose)

print(f"\nTraining data: {num_train_views} views at {height}x{width}")

# ============================================================================
# 4. TRAINING
# ============================================================================

print("\nTraining...")
num_train_steps = 100
batch_size = 256

for step in range(num_train_steps):
    # Random view
    pose_idx = np.random.randint(0, num_train_views)
    pose = poses_train[pose_idx].to(device)
    
    # Generate rays
    rays_o, rays_d = get_rays(height, width, focal, pose)
    rays_o = rays_o.to(device)
    rays_d = rays_d.to(device)
    
    # Create synthetic target (simple: gradient based on position)
    x_norm = torch.linspace(-1, 1, width)
    y_norm = torch.linspace(-1, 1, height)
    yy, xx = torch.meshgrid(y_norm, x_norm, indexing='ij')
    
    target_rgb = torch.stack([
        (xx + 1) / 2,  # Red channel
        torch.abs(yy),  # Green channel
        0.5 * torch.ones_like(xx),  # Blue channel
    ], dim=-1).to(device)
    
    # Random subset of rays
    indices = torch.randperm(height * width)[:batch_size]
    rays_o_batch = rays_o.reshape(-1, 3)[indices]
    rays_d_batch = rays_d.reshape(-1, 3)[indices]
    target_batch = target_rgb.reshape(-1, 3)[indices]
    
    # Training step
    loss_dict = trainer.train_step(
        rays_o_batch, rays_d_batch, target_batch,
        near=near, far=far,
        num_samples_coarse=16,
        num_samples_fine=32,
        perturb=True,
    )
    
    if (step + 1) % 20 == 0:
        print(f"  Step {step + 1:3d}: Loss coarse={loss_dict['loss_coarse']:.6f}, "
              f"fine={loss_dict['loss_fine']:.6f}")

print("Training complete!")

# ============================================================================
# 5. NOVEL VIEW SYNTHESIS
# ============================================================================

print("\nRendering novel views...")

from src.nerf.rendering import render_rays

# Generate novel camera poses (spiral)
num_novel_views = 4
poses_novel = []

for i in range(num_novel_views):
    theta = 2 * np.pi * i / num_novel_views
    radius = 4.0
    height_cam = 0.0
    
    cam_pos = torch.tensor([
        radius * np.cos(theta),
        height_cam,
        radius * np.sin(theta),
    ], dtype=torch.float32)
    
    forward = -cam_pos
    forward = forward / torch.norm(forward)
    up = torch.tensor([0.0, 1.0, 0.0])
    right = torch.cross(forward, up)
    right = right / torch.norm(right)
    up = torch.cross(right, forward)
    
    pose = torch.eye(4)
    pose[:3, :3] = torch.stack([right, up, -forward])
    pose[:3, 3] = cam_pos
    poses_novel.append(pose)

# Render
with torch.no_grad():
    for view_idx, pose in enumerate(poses_novel):
        pose = pose.to(device)
        
        # Get rays
        rays_o, rays_d = get_rays(height, width, focal, pose)
        rays_o = rays_o.to(device)
        rays_d = rays_d.to(device)
        
        # Render
        render_dict = render_rays(
            rays_o.reshape(-1, 3),
            rays_d.reshape(-1, 3),
            model,
            pos_encoder, dir_encoder,
            near, far,
            num_samples_coarse=16,
            num_samples_fine=32,
            perturb=False,
        )
        
        # Extract fine image
        color_fine = render_dict['color_fine'].reshape(height, width, 3)
        color_fine = torch.clamp(color_fine, 0, 1)
        
        print(f"  View {view_idx + 1}: RGB range [{color_fine.min():.3f}, {color_fine.max():.3f}]")

# ============================================================================
# 6. SAVE & VISUALIZATION
# ============================================================================

print("\nSaving model...")
output_dir = Path('./outputs')
output_dir.mkdir(exist_ok=True)

trainer.save_checkpoint(str(output_dir / 'model_final.pt'))

# Optional: visualize with matplotlib
try:
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, num_novel_views, figsize=(15, 4))
    
    with torch.no_grad():
        for ax_idx, pose in enumerate(poses_novel):
            pose = pose.to(device)
            rays_o, rays_d = get_rays(height, width, focal, pose)
            rays_o, rays_d = rays_o.to(device), rays_d.to(device)
            
            render_dict = render_rays(
                rays_o.reshape(-1, 3),
                rays_d.reshape(-1, 3),
                model, pos_encoder, dir_encoder,
                near, far, 16, 32, perturb=False
            )
            
            img = torch.clamp(render_dict['color_fine'].reshape(height, width, 3), 0, 1)
            axes[ax_idx].imshow(img.cpu().numpy())
            axes[ax_idx].set_title(f'View {ax_idx + 1}')
            axes[ax_idx].axis('off')
    
    plt.suptitle('NeRF Novel View Synthesis')
    plt.tight_layout()
    plt.savefig(str(output_dir / 'novel_views.png'), dpi=100, bbox_inches='tight')
    print(f"Visualization saved to {output_dir / 'novel_views.png'}")
    
except ImportError:
    print("Matplotlib not available for visualization")

print("\n✅ Quick-start complete!")
print(f"Model saved to {output_dir / 'model_final.pt'}")
