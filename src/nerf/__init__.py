"""
NeRF: Neural Radiance Fields for View Synthesis

A PyTorch implementation of the neural radiance field model from:
    Tancik et al., "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis" (2020)
    https://www.matthewtancik.com/nerf

Core modules:
    - encoding: Positional (Fourier) feature encoding
    - networks: NeRF network architecture (MLP with skip connections)
    - rays: Ray generation and sampling (stratified + hierarchical)
    - rendering: Volumetric rendering and alpha compositing
    - trainer: Training loop with photometric loss
"""

from .encoding import PositionalEncoding, IdentityEncoding
from .networks import NeRFNetwork, HierarchicalNeRF
from .rays import get_rays, stratified_sample, hierarchical_sample, get_directions_from_rays
from .rendering import volume_rendering, render_rays

__all__ = [
    'PositionalEncoding',
    'IdentityEncoding',
    'NeRFNetwork',
    'HierarchicalNeRF',
    'get_rays',
    'stratified_sample',
    'hierarchical_sample',
    'get_directions_from_rays',
    'volume_rendering',
    'render_rays',
]
