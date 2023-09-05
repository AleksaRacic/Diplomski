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

        self.g_function = DeepLinearLayerG(in_channels=self.pannel_embedding_tuple_size, out_channels=self.pannel_embedding_tuple_size)
        self.f_function = DeepLinearLayerF(in_channels=self.pannel_embedding_tuple_size, out_channels=1, dropout=0.5)
        self.norm = torch.nn.LayerNorm(self.pannel_embedding_tuple_size)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        batch_size, num_panels, height, width = x.size() #izvlacimo dimenzije ulaznog tensora
        x = x.view(batch_size * num_panels, 1 , height, width) # menjamo velicinu tensora x kako bismo ga pustili kroz KNN
        x = self.cnn(x)
        x = x.view(batch_size, num_panels, self.pannel_embedding_size) #Menjamo oblik posle KNN-a, velicina 32 oznacava num channels u KNN-u a 9*9 su dimenzije
        #interpretacija slika posle KNN-a, koje su hiperparametri TODO izvuci ove dimenzije iz KNN-a, a ne hardkodovati
        x = self.tagging(x) #oznacavanje choice i context panela
        context_panels = x[:, :8, :]
        solution_panels = x[:, 8:, :]
        context_pairs = self.group_context_panels(context_panels)
        context_pairs_g_out = self.g_function(context_pairs) # dobijamo ocenu relacije izmedju svaka 2 context panela
        f_out = torch.zeros(batch_size, 8, device=x.device) #formiramo izlazni tensor
        for i in range(8):
            context_solution_pairs = self.group_with_answers(context_panels, solution_panels[:, i, :])
            context_solution_pairs_g_out = self.g_function(context_solution_pairs)# dobijamo ocenu relacije izmedju svaka 2 context i choice panela
            relations = context_pairs_g_out + context_solution_pairs_g_out #spajamo tensore sa ocenama choice-choice i choice-context TODO izracunaj fc za sve kombinacije pa zatim ih sve saberi
            relations = self.norm(relations)# vrsimo normalizaciju na izlazne verovatnoce
            f_out[:, i] = self.f_function(relations).squeeze() # primenjujemo funkciju f i dajemo ocenu svakom choice panelu koja predstavlja koliko je taj panel dobar izbor
        return torch.softmax(f_out, dim = 1)# Softmax po svakom redu, tj na sve odgovore u jednom test primeru

class WildRelationNetworkTriplets(WildRelationNetworkPairs):
    def __init__(self):
        super(WildRelationNetworkPairs, self).__init__()
        
        self.group_context_panels = CombineContextPanelsTriples()
        self.group_with_answers = CombineContextPanelsWithTriples()

        self.pannel_embedding_size = NUM_CHANNELS * 9 * 9
        self.pannel_embedding_tuple_size = 3 * (self.pannel_embedding_size + 9)

        self.g_function = DeepLinearLayerG(in_channels=self.pannel_embedding_tuple_size, out_channels=self.pannel_embedding_tuple_size)
        self.f_function = DeepLinearLayerF(in_channels=self.pannel_embedding_tuple_size, out_channels=1, dropout=0.5)
        self.norm = torch.nn.LayerNorm(self.pannel_embedding_tuple_size)