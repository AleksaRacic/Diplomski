import torch

from .cnn import CNN

from .combiners import CombineContextPanels, GroupContextPanelsWith,  TagPannels
from .fully_connected import DeepLinearLayerF, DeepLinearLayerG

class Wild_Relation_Network(torch.nn.Module):

    def __init__(self):
        super(Wild_Relation_Network, self).__init__()
        self.cnn = CNN()
        self.tagovanje = TagPannels()
        self.group_context_panels = CombineContextPanels()
        self.group_with_answers = GroupContextPanelsWith()
        self.g_function = DeepLinearLayerG()
        self.f_function = DeepLinearLayerF()
        self.norm = torch.nn.LayerNorm([512])

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        batch_size, num_panels, height, width = x.size() #izvlacimo dimenzije ulaznog tensora
        x = x.view(batch_size * num_panels, 1 , height, width) # menjamo velicinu tensora x kako bismo ga pustili kroz KNN
        x = self.cnn(x)
        x = x.view(batch_size, num_panels, 32 * 9 * 9) #Menjamo oblik posle KNN-a, velicina 32 oznacava num channels u KNN-u a 9*9 su dimenzije
        #interpretacija slika posle KNN-a, koje su hiperparametri TODO izvuci ove dimenzije iz KNN-a, a ne hardkodovati
        x = self.tagovanje(x) #oznacavanje choice i context panela
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