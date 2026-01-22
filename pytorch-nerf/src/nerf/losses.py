import torch
import torch.nn as nn

class Losses:
    def __init__(self):
        self.mse_loss = nn.MSELoss()

    def photometric_loss(self, predicted_rgb, target_rgb):
        return self.mse_loss(predicted_rgb, target_rgb)

    def total_loss(self, predicted_rgb, target_rgb):
        return self.photometric_loss(predicted_rgb, target_rgb)