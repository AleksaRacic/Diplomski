import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .base_model import BaseModel
from .layers import CNN

class TagPannels(torch.nn.Module):
    '''
        Panel tagging.
        Resulting Tensor is of shape [batch_size, num_panels, NUM_RAVEN_PANELS + input.shape[-1=]]
        Tensor:
        batch_size x [
        [...1, 0, 0, 0, 0, 0, 0, 0, 0],
        [...0, 1, 0, 0, 0, 0, 0, 0, 0],
        [...0, 0, 1, 0, 0, 0, 0, 0, 0],
        [...0, 0, 0, 1, 0, 0, 0, 0, 0],
        [...0, 0, 0, 0, 1, 0, 0, 0, 0],
        [...0, 0, 0, 0, 0, 1, 0, 0, 0],
        [...0, 0, 0, 0, 0, 0, 1, 0, 0],
        [...0, 0, 0, 0, 0, 0, 0, 1, 0],
        [...0, 0, 0, 0, 0, 0, 0, 0, 1],
        [...0, 0, 0, 0, 0, 0, 0, 0, 1],
        [...0, 0, 0, 0, 0, 0, 0, 0, 1],
        [...0, 0, 0, 0, 0, 0, 0, 0, 1],
        [...0, 0, 0, 0, 0, 0, 0, 0, 1],
        [...0, 0, 0, 0, 0, 0, 0, 0, 1],
        [...0, 0, 0, 0, 0, 0, 0, 0, 1],
        [...0, 0, 0, 0, 0, 0, 0, 0, 1]
        ]
    '''
    def __init__(self):
        super(TagPannels, self).__init__()

    def forward(self, panel_embeddings: torch.Tensor) -> torch.Tensor:
        batch_size, number_panels, _ = panel_embeddings.shape
        tags = torch.zeros((number_panels, 9), device=panel_embeddings.device).type_as(panel_embeddings)
        tags[:9, :9] = torch.eye(9, device=panel_embeddings.device).type_as(panel_embeddings)
        tags[9:, 8] = torch.ones(7, device=panel_embeddings.device).type_as(panel_embeddings)
        tags = tags.expand((batch_size, -1, -1))
        return torch.cat([panel_embeddings, tags], dim=2)

class CombineContextPanelsPairs(torch.nn.Module):
    '''
        Combining panel_embeddings so that we have all posible pairs and conctenating them along the last dimension.
    '''
    def __init__(self):
        super(CombineContextPanelsPairs, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_context_panels, panel_embedding_length = x.size()
        return torch.cat([
            x.unsqueeze(1).repeat(1, num_context_panels, 1, 1),
            x.unsqueeze(2).repeat(1, 1, num_context_panels, 1)
        ], dim=3).view(batch_size, num_context_panels ** 2, 2 * panel_embedding_length)

class LinearBn(nn.Module):
    '''
        Linear layer with batch normalization and relu activation.
    '''
    def __init__(self, in_dim: int, out_dim: int):
        super(LinearBn, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        tensor_shape_before_bn = x.shape
        x = x.flatten(0, -2)
        x = self.bn(x)
        x = x.view(tensor_shape_before_bn)
        x = self.relu(x)
        return x

class DeepLinearLayerG(nn.Module):
    '''
        Function that finds relations between each pair of panels.
    '''
    def __init__(self, in_channels: int, out_channels: int):
        super(DeepLinearLayerG, self).__init__()
        self.fc = nn.Sequential(
            LinearBn(in_channels, out_channels),
            LinearBn(out_channels, out_channels),
            LinearBn(out_channels, out_channels),
            nn.Dropout(0.5),
            nn.Linear(out_channels, out_channels)
        )
    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x = self.fc(x)
        x = x.sum(dim=1)
        return x

class GroupContextPanelsWithPairs(torch.nn.Module):
    '''
        Combining candidate panel embeddings with context panel embeddings.
    '''
    def forward(self, context_panels_embeddings: torch.Tensor, cnadidate_panel_embedding: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            context_panels_embeddings,
            cnadidate_panel_embedding.unsqueeze(1).repeat(1, 8, 1)
        ], dim=2)

class DeepLinearLayerF(nn.Module):
    '''
        Function that evaluates each choice panel.
    '''
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.5):
        super(DeepLinearLayerF, self).__init__()
        self.fc = nn.Sequential(
            LinearBn(in_channels, 256),
            LinearBn(256, 256),
            nn.Dropout(dropout),
            nn.Linear(256, out_channels)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        return x

class WildRelationalNetwork(BaseModel):
    def __init__(self, lr, beta1, beta2, eps, tag_panels=True):
        super(WildRelationalNetwork, self).__init__()
        self.cnn_num_filters = 32
        self.cnn = CNN(in_channels=1, out_channels=self.cnn_num_filters)
        self.cnn_w_h = 4
        self.cnn_output_size = self.cnn_num_filters * self.cnn_w_h * self.cnn_w_h
        if tag_panels:
            self.tag_panels = TagPannels()
            self.cnn_output_size += 9
        else:
            self.tag_panels = nn.Identity()
            
        self.group_context_panels = CombineContextPanelsPairs()
        self.group_with_answers = GroupContextPanelsWithPairs()

        self.rn_embeding_size = 256
        self.embed_cnn_output = nn.Linear(self.cnn_output_size, self.rn_embeding_size)
        self.g_function = DeepLinearLayerG(self.rn_embeding_size * 2, 256)
        self.f_function = DeepLinearLayerF(256, 13)
        self.meta_beta = 0
        self.norm = torch.nn.LayerNorm(256)

        self.optimizer = optim.Adam(self.parameters(), lr=lr, betas=(beta1, beta2), eps=eps)

    def compute_loss(self, output, target, meta_target):
        pred, meta_pred = output[0], output[1]
        target_loss = F.cross_entropy(pred, target)
        meta_pred = torch.chunk(meta_pred, chunks=12, dim=1)
        meta_target = torch.chunk(meta_target, chunks=12, dim=1)
        if self.meta_beta == 0:
            meta_target_loss = 0
            for idx in range(0, 12):
                meta_target_loss += F.binary_cross_entropy(F.sigmoid(meta_pred[idx]), meta_target[idx])
        loss = target_loss + self.meta_beta * (meta_target_loss / 12)
        return loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_panels, height, width = x.size()
        x = x.view(batch_size * num_panels, 1 , height, width)
        x = self.cnn(x)
        x = x.view(batch_size, num_panels, self.cnn_num_filters * self.cnn_w_h * self.cnn_w_h)
        x = self.tag_panels(x)

        x = self.embed_cnn_output(x)

        context_panels = x[:, :8, :]
        candidate_panel = x[:, 8:, :]

        context_pairs = self.group_context_panels(context_panels)
        context_pairs_g_out = self.g_function(context_pairs)
        f_out = torch.zeros(batch_size, 8, 13, device=x.device)
        for i in range(8):
            context_solution_pairs = self.group_with_answers(context_panels, candidate_panel[:, i, :])
            context_solution_pairs_g_out = self.g_function(context_solution_pairs)
            relations = context_pairs_g_out + context_solution_pairs_g_out
            relations = self.norm(relations)
            f_out[:, i, :] = self.f_function(relations).squeeze()
 
        pred = torch.softmax(f_out[:,:,12], dim=1)
        meta_pred = torch.sum(f_out[:,:,0:12], dim=1)
        return pred, meta_pred