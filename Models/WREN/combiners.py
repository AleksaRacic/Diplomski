import torch

NUM_RAVEN_PANELS = 9

class TagPannels(torch.nn.Module):
    '''
        Panel tagging.
        Resulting Tensor is of shape [batch_size, num_panels, NUM_RAVEN_PANELS]
        Tensor:
        batch_size x [
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1]
        ]
    '''

    def __init__(self):
        super(TagPannels, self).__init__()

    def forward(self, panel_embeddings: torch.Tensor) -> torch.Tensor:
        batch_size, number_panels, _ = panel_embeddings.shape
        tags = torch.zeros((number_panels, 9), device=panel_embeddings.device).type_as(panel_embeddings) #TODO 16 je hardkodovano - treba da bude velicina num_panels, a 9 je velicina rejvenove matrice
        tags[:9, :9] = torch.eye(9, device=panel_embeddings.device).type_as(panel_embeddings)
        tags[9:, 8] = torch.ones(7, device=panel_embeddings.device).type_as(panel_embeddings)
        tags = tags.expand((batch_size, -1, -1))
        return torch.cat([panel_embeddings, tags], dim=2)


class CombineContextPanels(torch.nn.Module):
    '''
        Combining panel_embeddings so that we have all posible pairs and conctenating them along the last dimension.
    '''
    def __init__(self):
        super(CombineContextPanels, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_context_panels, panel_embedding_length = x.size()
        return torch.cat([
            x.unsqueeze(1).repeat(1, num_context_panels, 1, 1),
            x.unsqueeze(2).repeat(1, 1, num_context_panels, 1)
        ], dim=3).view(batch_size, num_context_panels ** 2, 2 * panel_embedding_length)


class GroupContextPanelsWith(torch.nn.Module):
    '''
        Combining candidate panel embeddings with context panel embeddings.
    '''
    def forward(self, context_panels_embeddings: torch.Tensor, cnadidate_panel_embedding: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            context_panels_embeddings,
            cnadidate_panel_embedding.unsqueeze(1).repeat(1, 8, 1)
        ], dim=2)