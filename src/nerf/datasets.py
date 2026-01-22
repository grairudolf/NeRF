"""
Dataset utilities for NeRF training and novel view synthesis.

References:
    Tancik et al., "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis" (2020)

Supports:
    - Synthetic datasets (Blender scenes)
    - Real-world datasets (LLFF)
    - Custom datasets with flexible camera pose specifications
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import json


class SyntheticNeRFDataset(Dataset):
    """
    Synthetic NeRF dataset (Blender scenes).
    
    Loads pre-rendered images and camera poses from JSON metadata.
    
    Args:
        data_dir (str): Directory containing images and camera metadata
        split (str): 'train', 'val', or 'test'
        downsampling (int): Downsample images by this factor
        num_rays_per_batch (int): Number of rays to sample per batch
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        downsampling: int = 1,
        num_rays_per_batch: int = 4096,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.downsampling = downsampling
        self.num_rays_per_batch = num_rays_per_batch
        
        # Load metadata
        metadata_path = self.data_dir / f'transforms_{split}.json'
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.frames = self.metadata['frames']
        self.camera_angle_x = self.metadata['camera_angle_x']
        
        # Load all images
        self.images = []
        self.poses = []
        
        for frame in self.frames:
            # Load image
            img_path = self.data_dir / f"{frame['file_path']}.png"
            if not img_path.exists():
                img_path = self.data_dir / f"{frame['file_path']}.jpg"
            
            # Simple PNG/JPG loading (requires PIL)
            try:
                from PIL import Image
                img = Image.open(img_path).convert('RGB')
                img = np.array(img, dtype=np.float32) / 255.0
                
                # Downsample
                if downsampling > 1:
                    img = img[::downsampling, ::downsampling]
                
                self.images.append(torch.from_numpy(img))
                
                # Load pose
                pose = np.array(frame['transform_matrix'], dtype=np.float32)
                self.poses.append(torch.from_numpy(pose))
            except Exception as e:
                print(f"Failed to load {img_path}: {e}")
        
        self.height, self.width = self.images[0].shape[:2]
        
        # Compute focal length from camera angle
        self.focal = 0.5 * self.width / np.tan(0.5 * self.camera_angle_x)
        
        # Scene bounds
        self.near = 2.0
        self.far = 6.0
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a training example.
        
        For efficiency, samples random rays from the image instead of
        returning the full image.
        """
        image = self.images[idx]
        pose = self.poses[idx]
        
        # Randomly sample rays
        h, w = image.shape[:2]
        
        # Random pixel coordinates
        coords = torch.stack(
            torch.meshgrid(torch.arange(w), torch.arange(h), indexing='ij'),
            dim=-1
        ).reshape(-1, 2)
        
        # Sample subset
        n_pixels = h * w
        if self.num_rays_per_batch < n_pixels:
            sample_indices = torch.randperm(n_pixels)[:self.num_rays_per_batch]
            coords = coords[sample_indices]
        
        # Get pixel colors
        pixel_colors = image[coords[:, 1], coords[:, 0]]  # [num_rays, 3]
        
        return {
            'image': image,
            'pose': pose,
            'pixel_colors': pixel_colors,
            'pixel_coords': coords,
            'height': h,
            'width': w,
            'focal': self.focal,
            'near': self.near,
            'far': self.far,
        }


class RayDataset(Dataset):
    """
    Dataset of pre-computed rays and colors for batched training.
    
    Useful for caching rays from multiple images for efficient training.
    """
    
    def __init__(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        rgb: torch.Tensor,
        batch_size: int = 4096,
    ):
        """
        Args:
            rays_o (torch.Tensor): Ray origins [num_rays, 3]
            rays_d (torch.Tensor): Ray directions [num_rays, 3]
            rgb (torch.Tensor): Ground truth colors [num_rays, 3]
            batch_size (int): Batch size for training
        """
        assert rays_o.shape[0] == rays_d.shape[0] == rgb.shape[0]
        
        self.rays_o = rays_o
        self.rays_d = rays_d
        self.rgb = rgb
        self.batch_size = batch_size
        self.num_rays = rays_o.shape[0]
    
    def __len__(self) -> int:
        return (self.num_rays + self.batch_size - 1) // self.batch_size
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = idx * self.batch_size
        end = min(start + self.batch_size, self.num_rays)
        
        return {
            'rays_o': self.rays_o[start:end],
            'rays_d': self.rays_d[start:end],
            'rgb': self.rgb[start:end],
        }


def compute_camera_poses(num_poses: int, radius: float = 4.0) -> torch.Tensor:
    """
    Generate camera poses in a circular trajectory around origin.
    
    Useful for novel view synthesis and rendering videos.
    
    Args:
        num_poses (int): Number of poses
        radius (float): Circular orbit radius
    
    Returns:
        torch.Tensor: Camera poses [num_poses, 4, 4]
    """
    poses = []
    for i in range(num_poses):
        theta = 2.0 * np.pi * i / num_poses
        
        # Camera position on circle
        cam_pos = np.array([
            radius * np.cos(theta),
            0.0,  # Fixed height
            radius * np.sin(theta),
        ])
        
        # Look at origin
        look_at = np.array([0.0, 0.0, 0.0])
        forward = look_at - cam_pos
        forward = forward / np.linalg.norm(forward)
        
        # Up vector
        up = np.array([0.0, 1.0, 0.0])
        
        # Right vector
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        # Recompute up
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        # Construct extrinsic matrix [R | t]
        # The standard camera matrix is [R^T | -R^T * t] but in NeRF we use [R | t]
        R = np.stack([right, up, -forward], axis=0)  # 3x3
        t = cam_pos  # 3
        
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = t
        
        poses.append(pose)
    
    return torch.from_numpy(np.stack(poses)).float()
