import torch
import torch.nn as nn

class MLP_NS(nn.Module):
    def __init__(self):
        super(MLP_NS, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 3)
        )

    def forward(self, input_tensor):
        return self.net(input_tensor)