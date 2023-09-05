from torch import nn, Tensor

class LinearBn(nn.Module):

    def __init__(self, in_dim: int, out_dim: int):
        super(LinearBn, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor):
        x = self.linear(x)
        tensor_shape_before_bn = x.shape
        x = x.flatten(0, -2)
        x = self.bn(x)
        x = x.view(tensor_shape_before_bn)
        x = self.relu(x)
        return x


class DeepLinearLayerG(nn.Module):
    """Funkcija koja trazi relacije izmedju svakog para panela."""
    def __init__(self):
        super(DeepLinearLayerG, self).__init__()
        self.fc = nn.Sequential(
            LinearBn(5202, 512), #ulaz je 5202 jer je to velicina izlaza iz KNN-a
            LinearBn(512, 512),
            LinearBn(512, 512),
            LinearBn(512, 512)
        )
    def forward(self, x: Tensor) ->Tensor:
        x = self.fc(x)
        x = x.sum(dim=1) # sumiramo vrednost po prvoj dimenziji TODO ovo odvoji u combiner feature
        return x

class DeepLinearLayerF(nn.Module):
    """Funkcija koja ocenjuje svaki choice panel"""
    def __init__(self):
        super(DeepLinearLayerF, self).__init__()
        self.mlp = nn.Sequential(
            LinearBn(512, 256),
            LinearBn(256, 256),
            #.Dropout(0.5),
            nn.Linear(256, 1)
        )
    def forward(self, x: Tensor) -> Tensor:
        x = self.mlp(x)
        return x