import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .base_model import BaseModel
from .layers import CNN


class lstm_module(nn.Module):
    def __init__(self):
        super(lstm_module, self).__init__()
        self.lstm = nn.LSTM(input_size=8*4*4+9, hidden_size=96, num_layers=1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(96, 8)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        hidden, _ = self.lstm(x)
        score = self.fc(hidden[-1, :, :])
        return score


class CNN_LSTM(BaseModel):
    def __init__(self, lr, beta1, beta2, epsilon):
        super(CNN_LSTM, self).__init__()
        self.conv = CNN(1, 8)
        self.lstm = lstm_module()
        self.register_buffer("tags", torch.tensor(self.build_tags(), dtype=torch.float))
        self.optimizer = optim.Adam(self.parameters(), lr=lr, betas=(beta1, beta2), eps=epsilon)

    def build_tags(self):
        tags = np.zeros((16, 9))
        tags[:8, :8] = np.eye(8)
        tags[8:, 8] = 1
        return tags

    def compute_loss(self, output, target, _):
        pred = output[0]
        loss = F.cross_entropy(pred, target)
        return loss

    def forward(self, x):
        batch = x.shape[0]
        features = self.conv(x.view(-1, 1, 80, 80))
        features = torch.cat([features.view(-1, 16, 8*4*4), self.tags.unsqueeze(0).expand(batch, -1, -1)], dim=-1)
        score = self.lstm(features)
        return score, None