import os
import numpy as np
import imageio
import torch
from torchvision import transforms

class NeRFDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = self.load_images()
        self.camera_params = self.load_camera_params()

    def load_images(self):
        images = []
        for filename in sorted(os.listdir(self.image_dir)):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(self.image_dir, filename)
                image = imageio.imread(img_path)
                images.append(image)
        return np.array(images)

    def load_camera_params(self):
        # Placeholder for loading camera intrinsics and extrinsics
        # This should be replaced with actual loading logic
        return {
            'intrinsics': np.array([[1000, 0, 512],
                                     [0, 1000, 512],
                                     [0, 0, 1]]),
            'extrinsics': np.array([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 5],
                                    [0, 0, 0, 1]])
        }

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        camera_param = self.camera_params
        if self.transform:
            image = self.transform(image)
        return image, camera_param

def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])