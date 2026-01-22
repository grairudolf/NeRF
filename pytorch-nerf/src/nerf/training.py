import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from nerf.losses import photometric_loss
from nerf.datasets import NeRFDataset
from nerf.models import NeRFModel
from nerf.utils import save_checkpoint, load_checkpoint

class NeRFTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NeRFModel().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.dataset = NeRFDataset(config['dataset_path'])
        self.dataloader = DataLoader(self.dataset, batch_size=config['batch_size'], shuffle=True)

    def train(self):
        self.model.train()
        for epoch in range(self.config['num_epochs']):
            for i, (rays, target_rgb) in enumerate(self.dataloader):
                rays = rays.to(self.device)
                target_rgb = target_rgb.to(self.device)

                self.optimizer.zero_grad()
                rgb_pred, density = self.model(rays)
                loss = photometric_loss(rgb_pred, target_rgb)
                loss.backward()
                self.optimizer.step()

                if i % self.config['log_interval'] == 0:
                    print(f'Epoch [{epoch}/{self.config["num_epochs"]}], Step [{i}/{len(self.dataloader)}], Loss: {loss.item():.4f}')

            if epoch % self.config['checkpoint_interval'] == 0:
                save_checkpoint(self.model, self.optimizer, epoch, self.config['checkpoint_dir'])

    def evaluate(self):
        self.model.eval()
        # Evaluation logic can be added here

    def load_model(self, checkpoint_path):
        load_checkpoint(checkpoint_path, self.model, self.optimizer)

    def save_model(self, checkpoint_path):
        save_checkpoint(self.model, self.optimizer, self.config['num_epochs'], checkpoint_path)