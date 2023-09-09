import torch

from .layers.cnn import CNN

from .layers.combiners import CombineContextPanelsPairs, GroupContextPanelsWithPairs,  TagPannels, CombineContextPanelsTriples, CombineContextPanelsWithTriples
from .layers.fully_connected import DeepLinearLayerF, DeepLinearLayerG

NUM_CHANNELS = 32

class WildRelationNetworkPairs(torch.nn.Module):

    def __init__(self):
        super(WildRelationNetworkPairs, self).__init__()
        self.cnn = CNN()
        self.tagging = TagPannels()
        self.group_context_panels = CombineContextPanelsPairs()
        self.group_with_answers = GroupContextPanelsWithPairs()

        self.pannel_embedding_size = NUM_CHANNELS * 9 * 9
        self.pannel_embedding_tuple_size = 2 * (self.pannel_embedding_size + 9)

        self.out_channels = 256

        self.g_function = DeepLinearLayerG(in_channels=self.pannel_embedding_tuple_size, out_channels= self.out_channels)
        self.f_function = DeepLinearLayerF(in_channels= self.out_channels, out_channels=1, dropout=0.5)
        self.norm = torch.nn.LayerNorm(self.out_channels)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        batch_size, num_panels, height, width = x.size()
        x = x.view(batch_size * num_panels, 1 , height, width)
        x = self.cnn(x)
        x = x.view(batch_size, num_panels, self.pannel_embedding_size)
        context_panels = x[:, :8, :]
        solution_panels = x[:, 8:, :]
        context_pairs = self.group_context_panels(context_panels)
        context_pairs_g_out = self.g_function(context_pairs)
        f_out = torch.zeros(batch_size, 8, device=x.device)
        for i in range(8):
            context_solution_pairs = self.group_with_answers(context_panels, solution_panels[:, i, :])
            context_solution_pairs_g_out = self.g_function(context_solution_pairs)
            relations = context_pairs_g_out + context_solution_pairs_g_out
            relations = self.norm(relations)
            f_out[:, i] = self.f_function(relations).squeeze()
        return torch.softmax(f_out, dim = 1)

class WildRelationNetworkTriplets(WildRelationNetworkPairs):
    def __init__(self):
        super(WildRelationNetworkTriplets, self).__init__()
        
        self.group_context_panels = CombineContextPanelsTriples()
        self.group_with_answers = CombineContextPanelsWithTriples()

        self.pannel_embedding_size = NUM_CHANNELS * 9 * 9
        self.pannel_embedding_tuple_size = 3 * (self.pannel_embedding_size + 9)

        self.g_function = DeepLinearLayerG(in_channels=self.pannel_embedding_tuple_size, out_channels=self.pannel_embedding_tuple_size)
        self.f_function = DeepLinearLayerF(in_channels=self.pannel_embedding_tuple_size, out_channels=1, dropout=0.5)
        self.norm = torch.nn.LayerNorm(self.pannel_embedding_tuple_size)