import torch
import torch.nn.functional as F
import numpy as np

class Renderer:
    def __init__(self, num_samples, near, far):
        self.num_samples = num_samples
        self.near = near
        self.far = far

    def render(self, rays, model):
        # Sample points along the rays
        z_vals = self.sample_along_rays(rays)
        points = rays[:, None, :3] + rays[:, None, 3:] * z_vals[..., None]

        # Query the NeRF model
        rgb, density = model(points)

        # Perform alpha compositing
        return self.alpha_composite(rgb, density, z_vals)

    def sample_along_rays(self, rays):
        # Stratified sampling along the rays
        z_vals = torch.linspace(self.near, self.far, self.num_samples)
        z_vals = z_vals + torch.rand_like(z_vals) * (self.far - self.near) / self.num_samples
        return z_vals

    def alpha_composite(self, rgb, density, z_vals):
        # Compute alpha values
        alpha = 1 - torch.exp(-density * (z_vals[..., 1:] - z_vals[..., :-1]))
        alpha = torch.cat([alpha, torch.ones_like(alpha[..., :1])], dim=-1)

        # Compute weights
        weights = alpha * torch cumprod(1 - alpha + 1e-10, dim=-1)

        # Composite RGB values
        rgb_composite = torch.sum(weights[..., None] * rgb, dim=-2)
        return rgb_composite

    def render_scene(self, rays, model):
        return self.render(rays, model)