"""
Main training script for NeRF.

References:
    Tancik et al., "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis" (2020)

This script demonstrates the complete training pipeline:
    1. Initialize model and encoders
    2. Create dataset and dataloader
    3. Train with photometric loss
    4. Validate and visualize results
    5. Render novel views
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json
from typing import Optional

from src.nerf.encoding import PositionalEncoding
from src.nerf.networks import NeRFNetwork, HierarchicalNeRF
from src.nerf.rays import get_rays
from src.nerf.rendering import render_rays
from src.nerf.trainer import NeRFTrainer
from src.nerf.datasets import SyntheticNeRFDataset, compute_camera_poses


def create_model(
    num_freq_pos: int = 10,
    num_freq_dir: int = 4,
    hidden_dim: int = 256,
    num_layers: int = 8,
    device: str = 'cuda',
) -> tuple:
    """
    Create NeRF model with coarse and fine networks.
    
    Args:
        num_freq_pos (int): Frequency levels for position encoding
        num_freq_dir (int): Frequency levels for direction encoding
        hidden_dim (int): Hidden layer dimension
        num_layers (int): Number of hidden layers
        device (str): Device ('cuda' or 'cpu')
    
    Returns:
        Tuple of (model, pos_encoder, dir_encoder)
    """
    # Positional encoders
    pos_encoder = PositionalEncoding(num_freqs=num_freq_pos, include_input=True)
    dir_encoder = PositionalEncoding(num_freqs=num_freq_dir, include_input=True)
    
    # Compute encoded dimensions
    pos_dim = pos_encoder.get_output_dim(3)  # 3D position
    dir_dim = dir_encoder.get_output_dim(3)  # 3D direction
    
    # Coarse network
    coarse_network = NeRFNetwork(
        input_dim_pos=pos_dim,
        input_dim_dir=dir_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        skip_layers=[4],
        use_view_direction=True,
    )
    
    # Fine network (same architecture)
    fine_network = NeRFNetwork(
        input_dim_pos=pos_dim,
        input_dim_dir=dir_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        skip_layers=[4],
        use_view_direction=True,
    )
    
    # Hierarchical model
    model = HierarchicalNeRF(coarse_network, fine_network).to(device)
    
    return model, pos_encoder, dir_encoder


def train(
    data_dir: str,
    output_dir: str = './outputs',
    num_epochs: int = 20,
    batch_size: int = 4096,
    num_freq_pos: int = 10,
    num_freq_dir: int = 4,
    learning_rate: float = 5e-4,
    num_samples_coarse: int = 64,
    num_samples_fine: int = 128,
    checkpoint_every: int = 5,
    device: str = 'cuda',
    white_background: bool = False,
):
    """
    Train NeRF on a dataset.
    
    Args:
        data_dir (str): Path to dataset directory
        output_dir (str): Output directory for checkpoints and results
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        num_freq_pos (int): Position encoding frequency levels
        num_freq_dir (int): Direction encoding frequency levels
        learning_rate (float): Learning rate
        num_samples_coarse (int): Coarse network samples per ray
        num_samples_fine (int): Fine network samples per ray
        checkpoint_every (int): Save checkpoint every N epochs
        device (str): Device to train on
        white_background (bool): Use white background
    """
    
    # Setup output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("NeRF Training")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Position encoding freqs: {num_freq_pos}")
    print(f"Direction encoding freqs: {num_freq_dir}")
    print(f"Coarse samples: {num_samples_coarse}, Fine samples: {num_samples_fine}")
    print("=" * 80)
    
    # Create model
    model, pos_encoder, dir_encoder = create_model(
        num_freq_pos=num_freq_pos,
        num_freq_dir=num_freq_dir,
        device=device,
    )
    
    print(f"Model created")
    print(f"  Coarse network parameters: {sum(p.numel() for p in model.coarse.parameters()):,}")
    print(f"  Fine network parameters: {sum(p.numel() for p in model.fine.parameters()):,}")
    
    # Create trainer
    trainer = NeRFTrainer(
        model, pos_encoder, dir_encoder,
        learning_rate=learning_rate,
        device=device,
    )
    
    # Load dataset
    print("\nLoading dataset...")
    train_dataset = SyntheticNeRFDataset(
        data_dir, split='train', num_rays_per_batch=batch_size
    )
    val_dataset = SyntheticNeRFDataset(
        data_dir, split='val', num_rays_per_batch=batch_size * 4
    )
    
    print(f"  Train images: {len(train_dataset)}")
    print(f"  Val images: {len(val_dataset)}")
    print(f"  Image size: {train_dataset.height} x {train_dataset.width}")
    print(f"  Focal length: {train_dataset.focal:.2f}")
    
    # Training loop
    print("\nStarting training...")
    global_step = 0
    
    for epoch in range(num_epochs):
        # Training
        train_loss_sum = 0.0
        train_steps = 0
        
        for sample in tqdm(train_dataset, desc=f"Epoch {epoch + 1}/{num_epochs} [train]"):
            # Get rays for this image
            from src.nerf.rays import get_rays
            
            pose = sample['pose'].to(device)
            image = sample['image'].to(device)
            height, width = sample['height'], sample['width']
            focal = sample['focal']
            
            rays_o, rays_d = get_rays(height, width, focal, pose)
            
            # Subsample rays
            pixel_indices = torch.randperm(height * width)[:batch_size]
            rays_o_batch = rays_o.reshape(-1, 3)[pixel_indices].to(device)
            rays_d_batch = rays_d.reshape(-1, 3)[pixel_indices].to(device)
            rgb_batch = image.reshape(-1, 3)[pixel_indices].to(device)
            
            # Training step
            loss_dict = trainer.train_step(
                rays_o_batch, rays_d_batch, rgb_batch,
                near=train_dataset.near,
                far=train_dataset.far,
                num_samples_coarse=num_samples_coarse,
                num_samples_fine=num_samples_fine,
                perturb=True,
                white_background=white_background,
            )
            
            train_loss_sum += loss_dict['loss_total']
            train_steps += 1
            global_step += 1
        
        avg_train_loss = train_loss_sum / max(train_steps, 1)
        
        # Validation
        val_loss_sum = 0.0
        val_steps = 0
        
        with torch.no_grad():
            for sample in tqdm(val_dataset, desc=f"Epoch {epoch + 1}/{num_epochs} [val]"):
                pose = sample['pose'].to(device)
                image = sample['image'].to(device)
                height, width = sample['height'], sample['width']
                focal = sample['focal']
                
                rays_o, rays_d = get_rays(height, width, focal, pose)
                rays_o = rays_o.to(device)
                rays_d = rays_d.to(device)
                
                # Evaluate on full image
                render_dict = render_rays(
                    rays_o.reshape(-1, 3),
                    rays_d.reshape(-1, 3),
                    model,
                    pos_encoder, dir_encoder,
                    train_dataset.near, train_dataset.far,
                    num_samples_coarse, num_samples_fine,
                    perturb=False,
                    white_background=white_background,
                )
                
                # Compute PSNR
                color_fine = render_dict['color_fine'].reshape(height, width, 3)
                mse = torch.mean((color_fine - image) ** 2)
                psnr = -10.0 * torch.log10(mse + 1e-8)
                
                val_loss_sum += mse.item()
                val_steps += 1
        
        avg_val_loss = val_loss_sum / max(val_steps, 1)
        avg_val_psnr = -10.0 * np.log10(avg_val_loss + 1e-8)
        
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"  Train loss: {avg_train_loss:.4f}")
        print(f"  Val loss:   {avg_val_loss:.4f} (PSNR: {avg_val_psnr:.2f} dB)")
        
        # Save checkpoint
        if (epoch + 1) % checkpoint_every == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch + 1:03d}.pt"
            trainer.save_checkpoint(str(checkpoint_path))
            print(f"  Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_path = output_dir / "model_final.pt"
    trainer.save_checkpoint(str(final_path))
    print(f"\nTraining complete. Final model saved to {final_path}")
    
    return trainer, train_dataset


def render_novel_views(
    trainer: NeRFTrainer,
    dataset,
    output_dir: str = './outputs',
    num_views: int = 60,
    num_frames: int = 1,
):
    """
    Render novel views using trained model.
    """
    output_dir = Path(output_dir) / "novel_views"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nRendering novel views...")
    
    # Generate camera poses
    poses = compute_camera_poses(num_views, radius=4.0)
    
    with torch.no_grad():
        for i, pose in enumerate(poses):
            pose = pose.to(trainer.device)
            
            # Get rays
            rays_o, rays_d = get_rays(
                dataset.height, dataset.width,
                dataset.focal, pose
            )
            
            rays_o = rays_o.to(trainer.device)
            rays_d = rays_d.to(trainer.device)
            
            # Render
            render_dict = render_rays(
                rays_o.reshape(-1, 3),
                rays_d.reshape(-1, 3),
                trainer.model,
                trainer.pos_encoder,
                trainer.dir_encoder,
                dataset.near, dataset.far,
                num_samples_coarse=64,
                num_samples_fine=128,
                perturb=False,
            )
            
            # Reshape and convert to image
            color = render_dict['color_fine'].reshape(dataset.height, dataset.width, 3)
            color = torch.clamp(color, 0, 1)
            color_np = color.cpu().numpy()
            
            # Save
            img_path = output_dir / f"view_{i:03d}.png"
            
            # Use matplotlib to save
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(color_np)
            ax.axis('off')
            fig.savefig(img_path, bbox_inches='tight', dpi=100)
            plt.close(fig)
            
            if (i + 1) % 10 == 0:
                print(f"  Rendered {i + 1}/{num_views} views")
    
    print(f"Novel views saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train NeRF model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4096, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--no_white_bg', action='store_true', help='Disable white background')
    parser.add_argument('--render_only', action='store_true', help='Render novel views only')
    
    args = parser.parse_args()
    
    if not args.render_only:
        trainer, dataset = train(
            args.data_dir,
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=args.device,
            white_background=not args.no_white_bg,
        )
        
        render_novel_views(trainer, dataset, args.output_dir)
