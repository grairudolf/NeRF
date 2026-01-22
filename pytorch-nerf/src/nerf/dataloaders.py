import torch
from torch.utils.data import DataLoader, Dataset

class NeRFDataset(Dataset):
    def __init__(self, images, poses, intrinsics):
        self.images = images
        self.poses = poses
        self.intrinsics = intrinsics

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        pose = self.poses[idx]
        intrinsic = self.intrinsics[idx]
        return image, pose, intrinsic

def create_dataloader(images, poses, intrinsics, batch_size=32, shuffle=True):
    dataset = NeRFDataset(images, poses, intrinsics)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)