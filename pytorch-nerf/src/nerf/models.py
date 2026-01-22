import torch
import torch.nn as nn
import torch.nn.functional as F
from nerf.positional_encoding import positional_encoding

class NeRFModel(nn.Module):
    def __init__(self, num_layers=8, num_units=256, num_views=3):
        super(NeRFModel, self).__init__()
        self.num_layers = num_layers
        self.num_units = num_units
        
        # Input layers for (x, y, z) and view direction
        self.input_layer = nn.Linear(3 + num_views, num_units)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(num_units, num_units) for _ in range(num_layers - 1)]
        )
        
        # Output layers for RGB and density
        self.rgb_layer = nn.Linear(num_units, 3)
        self.density_layer = nn.Linear(num_units, 1)

    def forward(self, x, view_dir):
        # Apply positional encoding
        x_encoded = positional_encoding(x)
        view_dir_encoded = positional_encoding(view_dir)
        
        # Concatenate encoded positions and view directions
        input_data = torch.cat([x_encoded, view_dir_encoded], dim=-1)
        
        # Pass through the input layer
        x = F.relu(self.input_layer(input_data))
        
        # Pass through hidden layers
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        
        # Output RGB and density
        rgb = torch.sigmoid(self.rgb_layer(x))
        density = F.relu(self.density_layer(x))
        
        return rgb, density