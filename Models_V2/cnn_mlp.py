import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .base_model import BaseModel
from .layers import CNN

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(32*4*4, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 8)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CNN_MLP(BaseModel):
    def __init__(self, lr, beta1, beta2, epsilon):
        super(CNN_MLP, self).__init__()
        self.conv = CNN(16, 32)
        self.mlp = MLP()
        self.optimizer = optim.Adam(self.parameters(), lr=lr, betas=(beta1, beta2), eps=epsilon)

    def compute_loss(self, output, target, _):
        pred = output[0]
        loss = F.cross_entropy(pred, target)
        return loss

    def forward(self, x):
        x = self.conv(x.view(-1, 16, 80, 80))
        score = self.mlp(x.view(-1, 32*4*4))
        return score, None