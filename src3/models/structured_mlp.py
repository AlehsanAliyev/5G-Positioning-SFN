import torch
import torch.nn as nn


class StructuredMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(256, 128, 64), dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
        self.net = nn.Sequential(*layers)
        self.coord_head = nn.Linear(prev_dim, 2)

    def forward(self, x):
        x = self.net(x)
        coords = self.coord_head(x)
        log_var = torch.zeros(len(x), 1, device=x.device)
        return coords, log_var
