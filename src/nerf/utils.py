"""
Utility functions for NeRF training, rendering, and visualization.

References:
    Tancik et al., "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis" (2020)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional, Dict
import json


def compute_psnr(mse: torch.Tensor) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) from MSE.
    
    PSNR = 20 * log10(MAX_I) - 10 * log10(MSE)
    where MAX_I = 1.0 for normalized images.
    
    Args:
        mse (torch.Tensor): Mean squared error
    
    Returns:
        float: PSNR in dB
    """
    if mse == 0:
        return float('inf')
    return -10.0 * torch.log10(mse).item()


def compute_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> float:
    """
    Compute Structural Similarity Index (SSIM) between two images.
    
    Args:
        img1 (torch.Tensor): First image [H, W, 3] in [0, 1]
        img2 (torch.Tensor): Second image [H, W, 3] in [0, 1]
        window_size (int): Gaussian window size for local comparison
    
    Returns:
        float: SSIM value in [-1, 1] (higher is better)
    """
    # This is a simplified SSIM; full implementation would use Gaussian weighting
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    
    mean1 = img1.mean()
    mean2 = img2.mean()
    var1 = ((img1 - mean1) ** 2).mean()
    var2 = ((img2 - mean2) ** 2).mean()
    cov = ((img1 - mean1) * (img2 - mean2)).mean()
    
    ssim = ((2 * mean1 * mean2 + c1) * (2 * cov + c2)) / \
           ((mean1 ** 2 + mean2 ** 2 + c1) * (var1 + var2 + c2))
    
    return ssim.item()


def visualize_rays(rays_o: torch.Tensor, rays_d: torch.Tensor, num_rays: int = 50):
    """
    Visualize ray origins and directions in 3D.
    
    Args:
        rays_o (torch.Tensor): Ray origins [H, W, 3]
        rays_d (torch.Tensor): Ray directions [H, W, 3]
        num_rays (int): Number of rays to visualize
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Sample subset of rays
    h, w = rays_o.shape[:2]
    indices = np.random.choice(h * w, num_rays, replace=False)
    
    for idx in indices:
        y, x = divmod(idx, w)
        o = rays_o[y, x].numpy()
        d = rays_d[y, x].numpy()
        
        ax.quiver(o[0], o[1], o[2], d[0], d[1], d[2],
                 length=1.0, alpha=0.6, arrow_length_ratio=0.3)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Ray Visualization ({num_rays} rays)')
    plt.show()


def visualize_depth_map(depth_map: torch.Tensor, mask: Optional[torch.Tensor] = None):
    """
    Visualize depth map with optional occlusion mask.
    
    Args:
        depth_map (torch.Tensor): Depth values [H, W] or [H, W, 1]
        mask (torch.Tensor): Optional occlusion mask [H, W]
    """
    depth = depth_map.squeeze().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if mask is not None:
        depth_masked = np.where(mask.cpu().numpy() > 0.5, depth, np.nan)
        im = ax.imshow(depth_masked, cmap='viridis')
    else:
        im = ax.imshow(depth, cmap='viridis')
    
    ax.set_title('Depth Map')
    plt.colorbar(im, ax=ax, label='Depth')
    plt.show()


def compose_with_background(
    rgb: torch.Tensor,
    acc: torch.Tensor,
    bg_color: torch.Tensor = None,
    white_background: bool = False,
) -> torch.Tensor:
    """
    Composite rendered image with background.
    
    Args:
        rgb (torch.Tensor): Rendered RGB [H, W, 3]
        acc (torch.Tensor): Accumulated alpha [H, W, 1]
        bg_color (torch.Tensor): Background color [3]
        white_background (bool): Use white background
    
    Returns:
        torch.Tensor: Composited image [H, W, 3]
    """
    if bg_color is None:
        if white_background:
            bg_color = torch.ones(3, device=rgb.device)
        else:
            bg_color = torch.zeros(3, device=rgb.device)
    
    return rgb + (1.0 - acc) * bg_color


def save_image(img: torch.Tensor, path: str, normalize: bool = True):
    """
    Save image to file.
    
    Args:
        img (torch.Tensor): Image tensor [H, W, 3] with values in [0, 1]
        path (str): Output file path
        normalize (bool): If True, normalize to [0, 255] before saving
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    img_np = img.cpu().numpy()
    
    if normalize:
        img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
    
    # Use PIL to save
    try:
        from PIL import Image
        Image.fromarray(img_np).save(path)
    except ImportError:
        # Fallback to matplotlib
        plt.imsave(path, img_np)
    
    print(f"Image saved to {path}")


def create_video_from_frames(
    frame_dir: str,
    output_path: str,
    fps: int = 30,
    frame_pattern: str = "*.png",
):
    """
    Create video from sequence of frames.
    
    Args:
        frame_dir (str): Directory containing frames
        output_path (str): Output video file path
        fps (int): Frames per second
        frame_pattern (str): Glob pattern for frame files
    """
    try:
        import cv2
    except ImportError:
        print("OpenCV not installed. Cannot create video.")
        return
    
    frame_dir = Path(frame_dir)
    frames = sorted(frame_dir.glob(frame_pattern))
    
    if not frames:
        print(f"No frames found matching pattern: {frame_pattern}")
        return
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(str(frames[0]))
    height, width = first_frame.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Write frames
    for frame_path in frames:
        frame = cv2.imread(str(frame_path))
        writer.write(frame)
    
    writer.release()
    print(f"Video saved to {output_path}")


def get_batch_rays(
    height: int,
    width: int,
    focal: float,
    poses: torch.Tensor,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate batched rays from multiple poses.
    
    Args:
        height (int): Image height
        width (int): Image width
        focal (float): Focal length
        poses (torch.Tensor): Camera poses [num_poses, 4, 4]
        batch_size (int): Batch size
    
    Returns:
        Tuple of (rays_o, rays_d, pose_indices) all concatenated
    """
    from src.nerf.rays import get_rays
    
    all_rays_o = []
    all_rays_d = []
    pose_indices = []
    
    for pose_idx, pose in enumerate(poses):
        rays_o, rays_d = get_rays(height, width, focal, pose)
        all_rays_o.append(rays_o.reshape(-1, 3))
        all_rays_d.append(rays_d.reshape(-1, 3))
        pose_indices.extend([pose_idx] * (height * width))
    
    rays_o = torch.cat(all_rays_o, dim=0)
    rays_d = torch.cat(all_rays_d, dim=0)
    pose_indices = torch.tensor(pose_indices)
    
    # Shuffle and batch
    indices = torch.randperm(rays_o.shape[0])
    
    return rays_o[indices], rays_d[indices], pose_indices[indices]


def save_config(config: Dict, path: str):
    """
    Save configuration to JSON file.
    
    Args:
        config (Dict): Configuration dictionary
        path (str): Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Config saved to {path}")


def load_config(path: str) -> Dict:
    """
    Load configuration from JSON file.
    
    Args:
        path (str): Config file path
    
    Returns:
        Dict: Configuration dictionary
    """
    with open(path, 'r') as f:
        return json.load(f)
