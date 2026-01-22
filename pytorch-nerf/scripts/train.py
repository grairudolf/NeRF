import os
import yaml
import torch
from torch.utils.data import DataLoader
from nerf.datasets import NeRFDataset
from nerf.models import NeRFModel
from nerf.training import train_nerf
from nerf.utils import set_seed

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    set_seed(config['seed'])

    dataset = NeRFDataset(config['dataset'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    model = NeRFModel(config['model'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    train_nerf(model, dataloader, optimizer, config['num_epochs'])

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), '../configs/default.yaml')
    main(config_path)