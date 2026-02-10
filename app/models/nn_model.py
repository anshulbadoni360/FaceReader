"""PyTorch model definitions."""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim),
            nn.BatchNorm1d(dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class AUMultiTaskModel(nn.Module):
    def __init__(self, n_features: int = 40, n_aus: int = 20, 
                 hidden_dims: tuple = (256, 256, 128), dropout: float = 0.3):
        super().__init__()
        
        layers = []
        in_dim = n_features
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim), 
                nn.BatchNorm1d(h_dim), 
                nn.GELU(), 
                nn.Dropout(dropout)
            ]
            in_dim = h_dim
        
        layers.extend([ResidualBlock(in_dim, dropout), ResidualBlock(in_dim, dropout)])
        self.backbone = nn.Sequential(*layers)
        
        self.cls_head = nn.Sequential(
            nn.Linear(in_dim, 64), nn.GELU(), 
            nn.Dropout(dropout * 0.5), nn.Linear(64, n_aus)
        )
        self.reg_head = nn.Sequential(
            nn.Linear(in_dim, 64), nn.GELU(), 
            nn.Dropout(dropout * 0.5), nn.Linear(64, n_aus), nn.ReLU()
        )

    def forward(self, x):
        h = self.backbone(x)
        return self.cls_head(h), self.reg_head(h)
