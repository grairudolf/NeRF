"""
Configuration for NeRF training and rendering.

This file defines all hyperparameters and settings used in training.
Modify these values to experiment with different architectures and training regimes.

References:
    Tancik et al., "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis" (2020)
"""

# ============================================================================
# NETWORK ARCHITECTURE
# ============================================================================

# Positional Encoding (Fourier Features)
num_freqs_pos = 10          # Frequency levels for position encoding (L)
num_freqs_dir = 4           # Frequency levels for direction encoding

# NeRF MLP Architecture
hidden_dim = 256            # Number of hidden units per layer
num_layers = 8              # Total number of hidden layers
skip_layers = [4]           # Indices where positional encoding is concatenated

# ============================================================================
# SAMPLING AND RENDERING
# ============================================================================

near = 2.0                  # Near plane distance
far = 6.0                   # Far plane distance

num_samples_coarse = 64     # Number of stratified samples
num_samples_fine = 128      # Number of hierarchical samples

lindisp = False             # If True, sample linearly in disparity (1/depth)
perturb_train = True        # Perturb samples during training
perturb_val = False         # Don't perturb during validation/rendering

# ============================================================================
# TRAINING
# ============================================================================

learning_rate = 5e-4        # Initial learning rate (Adam optimizer)
lr_decay_factor = 0.9999    # Exponential decay: lr *= decay_factor each step

batch_size = 4096           # Rays per batch
num_epochs = 50             # Number of training epochs
checkpoint_interval = 5     # Save checkpoint every N epochs

# ============================================================================
# DATASET
# ============================================================================

downsampling = 1            # Downsample images by this factor (1 = no downsampling)
white_background = False    # Use white background for rendered images

# Image dimensions (auto-detected from data, but can be overridden)
# image_height = 800
# image_width = 800

# ============================================================================
# OPTIMIZATION
# ============================================================================

# Early stopping (optional)
use_early_stopping = False
early_stopping_patience = 10
early_stopping_threshold = 1e-4

# Gradient clipping
grad_clip = None            # If not None, clip gradients to this norm

# Weight decay
weight_decay = 0.0          # L2 regularization (0 = no regularization)

# ============================================================================
# RENDERING / NOVEL VIEW SYNTHESIS
# ============================================================================

num_render_poses = 60       # Number of poses for video rendering
render_radius = 4.0         # Radius of circular orbit for rendering

# ============================================================================
# VALIDATION
# ============================================================================

val_interval = 1            # Validate every N epochs
num_val_images = 5          # Number of validation images to render

# Metrics
compute_psnr = True         # Compute Peak Signal-to-Noise Ratio
compute_ssim = False        # Compute Structural Similarity Index
compute_lpips = False       # Compute LPIPS perceptual distance

# ============================================================================
# LOGGING AND OUTPUT
# ============================================================================

log_interval = 10           # Print training loss every N iterations
save_images_interval = 100  # Save rendered images every N training steps

# Output directories
output_dir = './outputs'
checkpoint_dir = './outputs/checkpoints'
renders_dir = './outputs/renders'
logs_dir = './outputs/logs'

# ============================================================================
# DEVICE AND PRECISION
# ============================================================================

device = 'cuda'             # 'cuda' or 'cpu'
dtype = 'float32'           # 'float32' or 'float64' (float16 may have numerical issues)

# ============================================================================
# EXPERIMENTAL SETTINGS
# ============================================================================

# View-dependent RGB
use_view_direction = True   # Include view direction in network input

# Positional encoding variations
include_input_in_encoding = True  # Include raw input in encoded features
log_space_encoding = True         # Sample frequencies in log space (2^0, 2^1, ...)

# Regularization
use_density_smoothing = False     # Add regularization to encourage smooth density
density_smoothing_weight = 0.01   # Weight for density smoothing loss

# ============================================================================
# ABLATION STUDY FLAGS
# ============================================================================

# Disable hierarchical sampling (use only coarse network)
use_hierarchical_sampling = True

# Use only fine network (skip coarse)
use_only_fine = False

# Disable skip connections
use_skip_connections = True

# ============================================================================
# REPRODUCIBILITY
# ============================================================================

seed = 42                   # Random seed for reproducibility
