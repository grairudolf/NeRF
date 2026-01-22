"""
Training loop for NeRF with photometric MSE loss.

References:
    Tancik et al., "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis" (2020)

Loss function:
    L = MSE(rendered_color_coarse, ground_truth) + MSE(rendered_color_fine, ground_truth)
    
This encourages the network to predict colors that match the observed images while learning
a density field that encodes scene geometry.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Optional, Dict, Tuple, Any
from pathlib import Path
import json

from .encoding import PositionalEncoding
from .networks import NeRFNetwork, HierarchicalNeRF
from .rendering import render_rays


class NeRFTrainer:
    """
    Trainer for NeRF models with hierarchical sampling and photometric loss.
    
    Args:
        model (HierarchicalNeRF): NeRF model with coarse and fine networks
        pos_encoder (PositionalEncoding): Positional encoder for coordinates
        dir_encoder (PositionalEncoding): Positional encoder for view directions
        learning_rate (float): Learning rate for optimizer
        device (str): Device to train on ('cuda' or 'cpu')
    """
    
    def __init__(
        self,
        model: HierarchicalNeRF,
        pos_encoder: PositionalEncoding,
        dir_encoder: PositionalEncoding,
        learning_rate: float = 5e-4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.model = model.to(device)
        self.pos_encoder = pos_encoder
        self.dir_encoder = dir_encoder
        self.device = device
        
        # Optimizer: use Adam (as in original paper)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9999)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Training state
        self.global_step = 0
        self.train_losses = []
        self.val_losses = []
    
    def compute_loss(
        self,
        render_dict: Dict[str, torch.Tensor],
        target_image: torch.Tensor,
        use_fine: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute photometric MSE loss.
        
        Args:
            render_dict (Dict): Output from render_rays including coarse and fine colors
            target_image (torch.Tensor): Ground truth image, shape [batch, 3]
            use_fine (bool): Include fine loss if True
        
        Returns:
            Tuple[torch.Tensor, Dict[str, float]]:
                - loss: Total loss
                - loss_dict: Dictionary of individual loss components
        """
        loss_dict = {}
        
        # Coarse loss
        color_coarse = render_dict['color_coarse']
        loss_coarse = self.criterion(color_coarse, target_image)
        loss_dict['loss_coarse'] = loss_coarse.item()
        
        total_loss = loss_coarse
        
        # Fine loss (if available)
        if use_fine and 'color_fine' in render_dict:
            color_fine = render_dict['color_fine']
            loss_fine = self.criterion(color_fine, target_image)
            loss_dict['loss_fine'] = loss_fine.item()
            
            # Equal weight to coarse and fine (as in original paper)
            total_loss = total_loss + loss_fine
        
        loss_dict['loss_total'] = total_loss.item()
        
        return total_loss, loss_dict
    
    def train_step(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        target_rgb: torch.Tensor,
        near: float = 2.0,
        far: float = 6.0,
        num_samples_coarse: int = 64,
        num_samples_fine: int = 128,
        perturb: bool = True,
        white_background: bool = False,
    ) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            rays_o (torch.Tensor): Ray origins, shape [batch, 3]
            rays_d (torch.Tensor): Ray directions, shape [batch, 3]
            target_rgb (torch.Tensor): Target RGB colors, shape [batch, 3]
            near (float): Near plane
            far (float): Far plane
            num_samples_coarse (int): Number of coarse samples
            num_samples_fine (int): Number of fine samples
            perturb (bool): Perturb samples
            white_background (bool): Use white background
        
        Returns:
            Dict[str, float]: Loss components
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Render rays
        render_dict = render_rays(
            rays_o, rays_d,
            self.model,
            self.pos_encoder, self.dir_encoder,
            near, far,
            num_samples_coarse, num_samples_fine,
            perturb=perturb,
            white_background=white_background,
        )
        
        # Compute loss
        loss, loss_dict = self.compute_loss(render_dict, target_rgb, use_fine=True)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Update learning rate
        self.scheduler.step()
        
        self.global_step += 1
        self.train_losses.append(loss_dict['loss_total'])
        
        return loss_dict
    
    @torch.no_grad()
    def validate(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        target_rgb: torch.Tensor,
        near: float = 2.0,
        far: float = 6.0,
        num_samples_coarse: int = 64,
        num_samples_fine: int = 128,
        white_background: bool = False,
    ) -> Dict[str, Any]:
        """
        Validation step (no gradient computation).
        
        Returns:
            Dict with validation metrics and rendered outputs
        """
        self.model.eval()
        
        render_dict = render_rays(
            rays_o, rays_d,
            self.model,
            self.pos_encoder, self.dir_encoder,
            near, far,
            num_samples_coarse, num_samples_fine,
            perturb=False,
            white_background=white_background,
        )
        
        # Compute loss
        loss, loss_dict = self.compute_loss(render_dict, target_rgb, use_fine=True)
        
        self.val_losses.append(loss_dict['loss_total'])
        
        # Add rendering outputs
        render_dict.update(loss_dict)
        
        return render_dict
    
    def save_checkpoint(self, path: str, optimizer_state: bool = True):
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'global_step': self.global_step,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        
        if optimizer_state:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str, load_optimizer: bool = True):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
