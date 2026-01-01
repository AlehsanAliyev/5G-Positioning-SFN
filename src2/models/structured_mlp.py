# import torch
# import torch.nn as nn
#
# class StructuredMLP(nn.Module):
#     def __init__(self, input_dim, hidden_dim=128):
#         super(StructuredMLP, self).__init__()
#
#         # Shared trunk (learns complex signal relationships)
#         self.backbone = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.2)
#         )
#
#         # Head 1: Coordinate regression (x, y)
#         self.coord_head = nn.Linear(hidden_dim, 2)
#
#         # Head 2: Predict log(σ²) = uncertainty (Foliadis-style)
#         self.uncertainty_head = nn.Linear(hidden_dim, 1)
#
#     def forward(self, x):
#         h = self.backbone(x)
#         coords = self.coord_head(h)
#         log_var = self.uncertainty_head(h)  # log σ²
#         return coords, log_var
#
# #
# # model = StructuredMLP(input_dim=40)
# # coords, log_var = model(torch.randn(1, 40))  # fake batch
import torch
import torch.nn as nn

class StructuredMLP(nn.Module):
    def __init__(self, input_dim, output_uncertainty=False):
        super().__init__()
        self.output_uncertainty = output_uncertainty

        # 🧠 Expanded hidden layers
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Coordinate output (lon, lat)
        self.coord_head = nn.Linear(64, 2)

        # Optional uncertainty output (log-variance for fusion)
        self.uncertainty_head = nn.Linear(64, 1) if output_uncertainty else None

    def forward(self, x):
        x = self.net(x)
        coords = self.coord_head(x)
        log_var = (
            self.uncertainty_head(x)
            if self.output_uncertainty
            else torch.zeros(len(x), 1, device=x.device)
        )
        return coords, log_var
