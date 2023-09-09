import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .base_model import BaseModel

class conv_module(nn.Module):
    def __init__(self):
        super(conv_module, self).__init__()
        self.conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.batch_norm3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.batch_norm4 = nn.BatchNorm2d(32)
        self.relu4 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(self.batch_norm1(x))
        x = self.conv2(x)
        x = self.relu2(self.batch_norm2(x))
        x = self.conv3(x)
        x = self.relu3(self.batch_norm3(x))
        x = self.conv4(x)
        x = self.relu4(self.batch_norm4(x))
        return x.view(-1, 32*4*4)

class mlp_module(nn.Module):
    def __init__(self):
        super(mlp_module, self).__init__()
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
        self.conv = conv_module()
        self.mlp = mlp_module()
        self.optimizer = optim.Adam(self.parameters(), lr=lr, betas=(beta1, beta2), eps=epsilon)

    def compute_loss(self, output, target, _):
        pred = output[0]
        loss = F.cross_entropy(pred, target)
        return loss

    def forward(self, x):
        features = self.conv(x.view(-1, 16, 80, 80))
        score = self.mlp(features)
        return score, None