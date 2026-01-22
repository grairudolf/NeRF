import torch
import numpy as np
import imageio
import os
from nerf.models import NeRF
from nerf.renderer import render_rays
from nerf.utils import load_model, load_camera_poses

def render_scene(model_path, camera_poses, output_dir, num_images=100):
    # Load the trained NeRF model
    model = load_model(model_path)
    model.eval()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for i, pose in enumerate(camera_poses):
        if i >= num_images:
            break
        
        # Render the image from the current camera pose
        rendered_image = render_rays(model, pose)

        # Save the rendered image
        image_path = os.path.join(output_dir, f'rendered_{i:03d}.png')
        imageio.imwrite(image_path, (rendered_image.numpy() * 255).astype(np.uint8))
        print(f'Saved rendered image to {image_path}')

if __name__ == "__main__":
    # Example usage
    model_path = 'path/to/trained/model.pth'
    camera_poses = load_camera_poses('path/to/camera_poses.txt')
    output_dir = 'path/to/output/directory'

    render_scene(model_path, camera_poses, output_dir)